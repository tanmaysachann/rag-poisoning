"""Stress-test the learned branch on saved PPO document-edit rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from detect.product_v22 import PoisonDocumentScanner


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    benchmark = Path("data/benchmark/validation")
    queries = {row["qid"]: row for row in rows(benchmark / "queries.jsonl")}
    corpus = {row["doc_id"]: row["text"] for row in rows(benchmark / "corpus.jsonl")}
    policy_paths = [Path(f"results/ppo_validation_defender_seed{seed}.json")
                    for seed in (42, 43, 44)]
    scanner = PoisonDocumentScanner(use_snapshot=False)
    cases = []
    clean = {}
    try:
        for path in policy_paths:
            policy = json.loads(path.read_text(encoding="utf-8"))
            if policy["summary"]["cases"] != len(policy["cases"]):
                raise ValueError(f"PPO case mismatch: {path}")
            for item in policy["cases"]:
                qid = item["qid"]
                query = queries[qid]
                reference = corpus[query["support_doc_ids"][0]]
                if qid not in clean:
                    clean[qid] = scanner.assess(query["question"], reference, reference)
                assessment = scanner.assess(
                    query["question"], reference, item["final_document"])
                cases.append({
                    "seed_report": path.name,
                    "qid": qid,
                    "edited": item["final_document"] != reference,
                    "attack_success": item["terminal"]["attack_success"],
                    "v3_detected": item["terminal"]["attack_doc_detected"],
                    "v21_learned_flag": assessment["learned"]["flag"],
                    "v21_learned_score": assessment["learned"]["fusion_score"],
                    "clean_reference_flag": clean[qid]["learned"]["flag"],
                })
        output = Path("results/product_v22_ppo_stress.json")
        case_path = output.with_suffix(".cases.jsonl")
        with case_path.open("w", encoding="utf-8", newline="\n") as sink:
            for row in cases:
                sink.write(json.dumps(row) + "\n")
        successful = [row for row in cases if row["attack_success"]]
        report = {
            "status": "cross-corpus stress test on saved PPO attacker rollouts",
            "ppo_reports_sha256": {path.name: sha(path) for path in policy_paths},
            "attempts": len(cases),
            "edited_attempts": sum(row["edited"] for row in cases),
            "v21_learned_flags": sum(row["v21_learned_flag"] for row in cases),
            "v3_detected": sum(row["v3_detected"] is True for row in cases),
            "attack_successes": len(successful),
            "v21_flags_among_successes": sum(row["v21_learned_flag"] for row in successful),
            "clean_reference_flags": sum(row["learned"]["flag"] for row in clean.values()),
            "clean_reference_total": len(clean),
            "case_rows_sha256": sha(case_path),
            "limits": [
                "The PPO attacker was trained on MS MARCO against the older V3 defender.",
                "V21 was trained on BEIR NQ and its clean calibration does not transfer automatically.",
                "This scores saved final documents; no new answer replay or PPO retraining occurred.",
                "A flag on a successful attack is not a measured prevented answer.",
                "The supplied support passage is a benchmark reference, not independently authenticated.",
            ],
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in {"ppo_reports_sha256", "limits"}}, indent=2))
    finally:
        scanner.close()


if __name__ == "__main__":
    main()
