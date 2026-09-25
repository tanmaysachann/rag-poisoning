"""Score a frozen research detector on a named, previously generated attack set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.research_detector import ResearchDetector


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--strategy", choices=("random", "greedy", "stealth"), required=True)
    parser.add_argument("--detector", type=Path, default=ARTIFACTS_DIR / "research_detector_hashing.joblib")
    parser.add_argument("--attacks", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_benchmark(args.benchmark)
    detector = joblib.load(args.detector)
    if not isinstance(detector, ResearchDetector):
        raise ValueError("Research detector artifact has the wrong type")
    attack_path = args.attacks or RESULTS_DIR / f"attack_baseline_{args.split}_{args.strategy}.jsonl"
    attacks = _read_jsonl(attack_path)
    if any(row["split"] != args.split or row["strategy"] != args.strategy for row in attacks):
        raise ValueError("Attack records do not match the requested split and strategy")
    corpus = {row["doc_id"]: row for row in _read_jsonl(args.benchmark / args.split / "corpus.jsonl")}
    queries = {row["qid"]: row for row in _read_jsonl(args.benchmark / args.split / "queries.jsonl")}
    if set(queries) != {row["qid"] for row in attacks}:
        raise ValueError("Attack records must cover exactly the selected split")

    scored: list[dict] = []
    for qid, query in queries.items():
        clean = corpus[query["support_doc_ids"][0]]
        score = detector.score(query["question"], clean["text"])
        scored.append({"qid": qid, "label": 0, "risk_score": score["risk_score"],
                       "prediction": int(score["decision"] == "quarantine")})
    for attack in attacks:
        score = detector.score(attack["question"], attack["text"])
        scored.append({"qid": attack["qid"], "label": 1, "risk_score": score["risk_score"],
                       "prediction": int(score["decision"] == "quarantine"),
                       "attack_success": attack["attack_success"],
                       "top5_rank": attack["top5_rank"]})
    labels = np.asarray([row["label"] for row in scored])
    predictions = np.asarray([row["prediction"] for row in scored])
    risks = np.asarray([row["risk_score"] for row in scored])
    successful = [row for row in scored if row.get("attack_success")]
    metrics = {
        "split": args.split,
        "strategy": args.strategy,
        "threshold_source": "validation greedy clean cases; frozen from training run",
        "threshold": detector.threshold,
        "clean_cases": int(sum(labels == 0)),
        "attack_cases": int(sum(labels == 1)),
        "successful_undefended_attacks": len(successful),
        "false_positive_rate": float(np.mean(predictions[labels == 0])),
        "attack_recall": float(recall_score(labels, predictions)),
        "successful_attack_recall": float(np.mean([row["prediction"] for row in successful])) if successful else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, risks)),
        "pr_auc": float(average_precision_score(labels, risks)),
    }
    output = args.output or RESULTS_DIR / f"research_detector_{args.split}_{args.strategy}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
