"""Audit two PPO attack rounds and frozen-defender replay under source identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same_attacks(before: dict, after: dict) -> None:
    if len(before["cases"]) != 75 or len(after["cases"]) != 75:
        raise ValueError("Expected 75 validation attacks")
    for a, b in zip(before["cases"], after["cases"]):
        if any(a[key] != b[key] for key in ("qid", "wrong_answer", "final_document")):
            raise ValueError("Frozen policy produced a different document")
        actions = lambda row: [(step["operation"], step["position"], step["payload"])
                               for step in row["trace"]]
        if actions(a) != actions(b):
            raise ValueError("Frozen policy changed actions")


def outcome(path: Path) -> dict:
    data = load(path)
    return {
        "report_sha256": sha(path),
        "raw_successes": sum(row["terminal"]["attack_success"] for row in data["cases"]),
        "defended_successes": sum(
            row["terminal"]["defended_attack_success"] for row in data["cases"]),
        "detected_attacks": sum(
            row["terminal"]["attack_doc_detected"] is True for row in data["cases"]),
        "cases": len(data["cases"]),
    }


def main() -> None:
    rounds = []
    configs = [
        ("20261020", "snapshot_v22", "snapshot_v23_containment",
         "artifacts/poison_policy_ppo_snapshot_v22.pt"),
        ("20261021", "snapshot_v23_retrained", "snapshot_v24_identity_replay",
         "artifacts/poison_policy_ppo_snapshot_v23.pt"),
    ]
    for seed, vulnerable_name, hardened_name, checkpoint_name in configs:
        before_path = Path(f"results/ppo_validation_{vulnerable_name}_evaluation.json")
        if hardened_name == "snapshot_v24_identity_replay":
            after_path = Path("results/ppo_validation_snapshot_v24_identity_replay.json")
        else:
            after_path = Path(f"results/ppo_validation_{hardened_name}_evaluation.json")
        before, after = load(before_path), load(after_path)
        same_attacks(before, after)
        baseline_name = ("snapshot_v22" if seed == "20261020"
                         else "snapshot_v23_seed20261021")
        fixed_path = Path(
            f"results/edit_rollout_validation_fixed_substitution_{baseline_name}.summary.json")
        random_path = Path(f"results/edit_rollout_validation_random_{baseline_name}.summary.json")
        fixed, random = load(fixed_path), load(random_path)
        if any(item["seed"] != int(seed) or item["cases"] != 75 for item in (fixed, random)):
            raise ValueError("Baseline schedule mismatch")
        rounds.append({
            "seed": int(seed), "training_episodes": 200,
            "checkpoint_sha256": sha(Path(checkpoint_name)),
            "frozen_policy_against_training_defender": outcome(before_path),
            "same_policy_against_next_defender": outcome(after_path),
            "fixed_baseline_defended_successes": round(
                fixed["defended_attack_success_rate"] * fixed["cases"]),
            "random_baseline_defended_successes": round(
                random["defended_attack_success_rate"] * random["cases"]),
            "fixed_baseline_report_sha256": sha(fixed_path),
            "random_baseline_report_sha256": sha(random_path),
        })
    clean_path = Path("results/clean_snapshot_v24_identity.json")
    clean = load(clean_path)
    report = {
        "status": "two adaptive development rounds, each with one PPO seed",
        "training_questions_per_round": 350,
        "validation_questions": 75,
        "rounds": rounds,
        "clean_answer_replay": {
            "report_sha256": sha(clean_path),
            "same_answers": clean["same_answers"],
            "questions": clean["questions"],
            "clean_quarantines": clean["clean_quarantines"],
            "retrieved_clean_candidates": clean["retrieved_clean_candidates"],
        },
        "limits": [
            "The PPO policy was frozen only within each paired defender replay.",
            "The V23 and identity changes were made after inspecting attack failures.",
            "PPO used the local extractive RAG answer path and known-document replacement.",
            "Exact identity checking requires an independently pinned pre-attack ID/hash map.",
            "New inserted documents and approved source updates need separate handling.",
            "One seed per round and reused development questions do not prove generalization.",
        ],
    }
    output = Path("results/adaptive_ppo_v24.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rounds": rounds,
                      "clean_answer_replay": report["clean_answer_replay"]}, indent=2))


if __name__ == "__main__":
    main()
