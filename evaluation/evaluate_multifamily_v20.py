"""Compare the V20 six-family scorer with its V9 warm start on held-out questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from detect.research_detector import threshold_for_fpr
from detect.train_multifamily_v20 import read_verified_rows


def score_rows(scorer: ReferenceConditionedScorer, rows: list[dict],
               batch_size: int) -> np.ndarray:
    result = []
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        result.extend(scorer.score([
            {"query": row["query"], "reference": row["reference"],
             "candidate": row["candidate"]} for row in chunk
        ], batch_size=batch_size))
        print(f"scored {start + len(chunk)}/{len(rows)}", flush=True)
    values = np.asarray(result, dtype=np.float64)
    if len(values) != len(rows) or not np.isfinite(values).all():
        raise ValueError("Invalid score output")
    return values


def summarize(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    groups = defaultdict(list)
    by_question = defaultdict(list)
    for row, score in zip(rows, scores):
        family = row["family"]
        flag = bool(score >= threshold)
        groups[family].append(flag)
        if row["label"] == "poison":
            by_question[family, row["query_id"]].append(flag)
    clean = groups["clean"] + groups["reference"]
    attacks = [flag for family in groups if family not in {"clean", "reference"}
               for flag in groups[family]]
    return {
        "threshold": float(threshold),
        "clean_flags": sum(clean), "clean_total": len(clean),
        "clean_fpr": sum(clean) / len(clean),
        "reference_flags": sum(groups["reference"]),
        "reference_total": len(groups["reference"]),
        "poison_caught": sum(attacks), "poison_total": len(attacks),
        "poison_recall": sum(attacks) / len(attacks),
        "families": {
            family: {"caught": sum(flags), "total": len(flags),
                     "recall": sum(flags) / len(flags),
                     "questions_with_any_caught": sum(
                         any(values) for (name, _), values in by_question.items()
                         if name == family
                     ),
                     "questions": sum(name == family for name, _ in by_question)}
            for family, flags in groups.items() if family not in {"clean", "reference"}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/multifamily_v20"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/multifamily_v20"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/reference_detector_v9"))
    parser.add_argument("--output", type=Path, default=Path("results/multifamily_v20.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = read_verified_rows(args.data, "calibration", manifest)
    evaluation = read_verified_rows(args.data, "evaluation", manifest)
    training = read_verified_rows(args.data, "train", manifest)
    groups = [{row["query_id"] for row in part}
              for part in (training, calibration, evaluation)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Question leakage")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "protocol": "question-disjoint PoisonArena development, six published families",
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "models": {}, "split_counts": manifest["splits"],
        "limitations": manifest["limitations"] + [
            "Attack provenance labels do not prove answer compromise.",
            "The reference is the released BEIR qrel, unavailable as a trusted live oracle.",
        ],
    }
    case_rows = []
    for name, model_path in (("v9_warmstart", args.baseline), ("v20", args.model)):
        started = time.perf_counter()
        scorer = ReferenceConditionedScorer(model_path, device)
        cal_scores = score_rows(scorer, calibration, args.batch_size)
        eval_scores = score_rows(scorer, evaluation, args.batch_size)
        cal_clean = np.asarray([row["label"] != "poison" for row in calibration])
        labels = np.asarray([row["label"] == "poison" for row in evaluation])
        thresholds = {str(int(rate * 100)): threshold_for_fpr(cal_scores[cal_clean], rate)
                      for rate in (0.01, 0.02)}
        report["models"][name] = {
            "weights_sha256": hashlib.sha256(
                (model_path / "model.safetensors").read_bytes()
            ).hexdigest(),
            "calibration_clean_total": int(cal_clean.sum()),
            "calibration_clean_flags": {
                rate: int(np.sum(cal_scores[cal_clean] >= threshold))
                for rate, threshold in thresholds.items()
            },
            "evaluation_roc_auc": float(roc_auc_score(labels, eval_scores)),
            "evaluation_pr_auc": float(average_precision_score(labels, eval_scores)),
            "at_clean_fpr_target": {
                rate: summarize(evaluation, eval_scores, threshold)
                for rate, threshold in thresholds.items()
            },
            "scoring_seconds": round(time.perf_counter() - started, 2),
        }
        for row, score in zip(evaluation, eval_scores):
            case_rows.append({
                "model": name, "query_id": row["query_id"],
                "candidate_id": row["candidate_id"], "label": row["label"],
                "family": row["family"], "score": float(score),
                "flag_at_1pct": bool(score >= thresholds["1"]),
                "flag_at_2pct": bool(score >= thresholds["2"]),
            })
        del scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case_path = args.output.with_suffix(".cases.jsonl")
    with case_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in case_rows:
            sink.write(json.dumps(row) + "\n")
    report["case_rows_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": {
        name: value["at_clean_fpr_target"] for name, value in report["models"].items()
    }}, indent=2), flush=True)


if __name__ == "__main__":
    main()
