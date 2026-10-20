"""Test the same near-copy rule on saved PPO edits and their frozen source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from detect.snapshot_near_duplicate import SnapshotNearDuplicate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    snapshot_path = Path("data/benchmark_snapshot_v22")
    snapshot_manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    corpus_path = Path("data/benchmark/validation/corpus.jsonl")
    if snapshot_manifest["source_corpus_sha256"] != sha(corpus_path):
        raise ValueError("Benchmark source changed")
    queries = {row["qid"]: row for row in jsonl(
        Path("data/benchmark/validation/queries.jsonl"))}
    corpus = {row["doc_id"]: row["text"] for row in jsonl(corpus_path)}
    learned_path = Path("results/product_v22_ppo_stress.cases.jsonl")
    learned_report = json.loads(Path("results/product_v22_ppo_stress.json").read_text(encoding="utf-8"))
    if learned_report["case_rows_sha256"] != sha(learned_path):
        raise ValueError("Learned PPO case file changed")
    learned_cases = jsonl(learned_path)
    threshold = 0.9  # Reuse the BEIR clean-calibrated product setting.
    scanner = SnapshotNearDuplicate(snapshot_path)
    clean = {}
    cases = []
    try:
        for seed in (42, 43, 44):
            policy_path = Path(f"results/ppo_validation_defender_seed{seed}.json")
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            for row in policy["cases"]:
                qid = row["qid"]
                reference = corpus[queries[qid]["support_doc_ids"][0]]
                if qid not in clean:
                    clean[qid] = scanner.assess(reference)
                integrity = scanner.assess(row["final_document"])
                cases.append({
                    "seed": seed, "qid": qid,
                    "attack_success": row["terminal"]["attack_success"],
                    "integrity_flag": integrity["near_copy_score"] >= threshold,
                    "integrity": integrity,
                })
    finally:
        scanner.close()
    if len(cases) != len(learned_cases) or any(
        (row["qid"] != old["qid"] or row["attack_success"] != old["attack_success"])
        for row, old in zip(cases, learned_cases)
    ):
        raise ValueError("PPO/learned case alignment mismatch")
    successful = [row for row in cases if row["attack_success"]]
    combined = [row["integrity_flag"] or old["v21_learned_flag"]
                for row, old in zip(cases, learned_cases)]
    report = {
        "status": "cross-corpus development stress test; no answer replay",
        "snapshot_manifest_sha256": sha(snapshot_path / "manifest.json"),
        "ppo_learned_case_sha256": sha(learned_path),
        "threshold": threshold,
        "clean_exact_matches": sum(value["exact_snapshot_match"] for value in clean.values()),
        "clean_total": len(clean),
        "clean_integrity_flags": sum(value["near_copy_score"] >= threshold
                                     for value in clean.values()),
        "attack_attempts": len(cases),
        "integrity_flags": sum(row["integrity_flag"] for row in cases),
        "combined_flags": sum(combined),
        "successful_attempts": len(successful),
        "integrity_flags_among_successes": sum(row["integrity_flag"] for row in successful),
        "combined_flags_among_successes": sum(
            flag for row, flag in zip(cases, combined) if row["attack_success"]),
        "limits": [
            "The frozen corpus is a benchmark source, not a reviewed truth set.",
            "A near-copy alert detects changed source text; benign edits can alert too.",
            "The PPO policy was trained against V3, not the new product.",
            "Flags on saved successful rollouts are not measured prevented answers.",
            "The 0/75 clean result tests unmodified corpus passages, not legitimate updates.",
        ],
    }
    output = Path("results/product_v22_ppo_snapshot.json")
    case_path = output.with_suffix(".cases.jsonl")
    with case_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row, flag in zip(cases, combined):
            sink.write(json.dumps({**row, "combined_flag": flag}) + "\n")
    report["case_rows_sha256"] = sha(case_path)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "limits"}, indent=2))


if __name__ == "__main__":
    main()
