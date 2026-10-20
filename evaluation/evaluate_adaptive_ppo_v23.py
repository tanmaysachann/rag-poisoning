"""Audit the frozen PPO bypass and paired V23 containment replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from detect.snapshot_containment_v23 import SnapshotContainment


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    old_path = Path("results/ppo_validation_snapshot_v22_evaluation.json")
    new_path = Path("results/ppo_validation_snapshot_v23_containment_evaluation.json")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    if len(old["cases"]) != 75 or len(new["cases"]) != 75:
        raise ValueError("Expected identical 75-question rollouts")
    for a, b in zip(old["cases"], new["cases"]):
        if any(a[key] != b[key] for key in ("qid", "wrong_answer", "final_document")):
            raise ValueError("PPO replay changed its attack")
        if ([{key: step[key] for key in ("operation", "position", "payload")}
             for step in a["trace"]] !=
            [{key: step[key] for key in ("operation", "position", "payload")}
             for step in b["trace"]]):
            raise ValueError("PPO replay changed its attack")
    snapshot_path = Path("data/benchmark_snapshot_v22")
    scanner = SnapshotContainment(snapshot_path)
    try:
        clean = [scanner.assess(row["text"]) for row in jsonl(
            Path("data/benchmark/validation/corpus.jsonl"))]
    finally:
        scanner.close()
    fixed_path = Path("results/edit_rollout_validation_fixed_substitution_snapshot_v22.summary.json")
    random_path = Path("results/edit_rollout_validation_random_snapshot_v22.summary.json")
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    for baseline in (fixed, random):
        if baseline["cases"] != 75 or baseline["seed"] != 20261020:
            raise ValueError("Baseline schedule mismatch")
    report = {
        "status": "one-seed adaptive development replay, not a blind deployment claim",
        "train_episodes": 200,
        "train_questions": 350,
        "validation_questions": 75,
        "ppo_checkpoint_sha256": sha(Path("artifacts/poison_policy_ppo_snapshot_v22.pt")),
        "v22_replay_sha256": sha(old_path),
        "v23_replay_sha256": sha(new_path),
        "fixed_summary_sha256": sha(fixed_path),
        "random_summary_sha256": sha(random_path),
        "validation_snapshot_manifest_sha256": sha(snapshot_path / "manifest.json"),
        "v22_ppo": old["summary"],
        "v23_frozen_ppo_replay": new["summary"],
        "v22_fixed": fixed,
        "v22_random": random,
        "v23_clean_exact_matches": sum(item["exact_snapshot_match"] for item in clean),
        "v23_clean_integrity_flags": sum(item["near_copy_score"] >= 0.9 for item in clean),
        "v23_clean_total": len(clean),
        "limits": [
            "V23 was designed after inspecting the V22 PPO failure.",
            "The policy was frozen for the V23 replay, not retrained against V23.",
            "The RAG answerer is a deterministic extractive local benchmark, not a hosted LLM.",
            "Clean utility requires answer-level replay, beyond exact-source flags.",
            "One PPO seed and one attack surface cannot establish generalization.",
        ],
    }
    output = Path("results/adaptive_ppo_v23.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "v22_defended_successes": sum(
            row["terminal"]["defended_attack_success"] for row in old["cases"]),
        "v23_defended_successes": sum(
            row["terminal"]["defended_attack_success"] for row in new["cases"]),
        "v23_clean_flags": report["v23_clean_integrity_flags"],
        "v23_clean_total": report["v23_clean_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
