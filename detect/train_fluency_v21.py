"""Ablate clean-trained character fluency in the six-family signal detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from detect.char_fluency import CharFluency
from detect.research_detector import ResearchDetector, threshold_for_fpr
from detect.train_multifamily_v20 import read_verified_rows
from detect.train_multisignal_v20 import extract
from evaluation.evaluate_multifamily_v20 import summarize


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=7, learning_rate=0.05,
        l2_regularization=1.0, class_weight="balanced", random_state=20261020,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/multifamily_v20"))
    parser.add_argument("--v9-data", type=Path, default=Path("data/reference_detector_v9"))
    parser.add_argument("--v3", type=Path,
                        default=Path("artifacts/research_detector_v3_hashing.joblib"))
    parser.add_argument("--model-output", type=Path,
                        default=Path("artifacts/fluency_v21.joblib"))
    parser.add_argument("--report", type=Path, default=Path("results/fluency_v21.json"))
    args = parser.parse_args()
    if args.model_output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite V21 model or report")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v9_manifest = json.loads((args.v9_data / "manifest.json").read_text(encoding="utf-8"))
    v9_rows_path = args.v9_data / "train.jsonl"
    if hashlib.sha256(v9_rows_path.read_bytes()).hexdigest() != v9_manifest["train_sha256"]:
        raise ValueError("V9 clean text changed")
    clean_texts = [row["candidate"] for line in v9_rows_path.open(encoding="utf-8")
                   if (row := json.loads(line))["label"] != "poison"]
    fluency = CharFluency().fit(clean_texts)
    rows = {split: read_verified_rows(args.data, split, manifest)
            for split in ("train", "calibration", "evaluation")}
    qids = {split: {row["query_id"] for row in part} for split, part in rows.items()}
    if any(qids[a] & qids[b] for a in qids for b in qids if a != b):
        raise ValueError("Question leakage")
    started = time.perf_counter()
    detector = joblib.load(args.v3)
    if not isinstance(detector, ResearchDetector):
        raise TypeError("Unexpected V3 artifact")
    signals = {split: extract(part, detector) for split, part in rows.items()}
    extra = {split: np.vstack([fluency.features(row["candidate"]) for row in part])
             for split, part in rows.items()}
    targets = {split: np.asarray([row["label"] == "poison" for row in part], dtype=np.int8)
               for split, part in rows.items()}
    clean_mask = targets["calibration"] == 0
    reports = {}
    cases = []
    enhanced = None
    enhanced_threshold = None
    for name, x in (
        ("base_signals", signals),
        ("with_char_fluency", {split: np.hstack([signals[split], extra[split]])
                               for split in rows}),
    ):
        fitted = model().fit(x["train"], targets["train"])
        cal_scores = fitted.predict_proba(x["calibration"])[:, 1]
        eval_scores = fitted.predict_proba(x["evaluation"])[:, 1]
        cutoffs = {str(int(rate * 100)): threshold_for_fpr(
            cal_scores[clean_mask], rate) for rate in (0.01, 0.02)}
        reports[name] = {
            "calibration_clean_total": int(clean_mask.sum()),
            "calibration_clean_flags": {
                rate: int(np.sum(cal_scores[clean_mask] >= threshold))
                for rate, threshold in cutoffs.items()},
            "roc_auc": float(roc_auc_score(targets["evaluation"], eval_scores)),
            "pr_auc": float(average_precision_score(targets["evaluation"], eval_scores)),
            "evaluation_at_fpr_target": {
                rate: summarize(rows["evaluation"], eval_scores, threshold)
                for rate, threshold in cutoffs.items()},
        }
        for row, score in zip(rows["evaluation"], eval_scores):
            cases.append({"model": name, "query_id": row["query_id"],
                          "candidate_id": row["candidate_id"], "family": row["family"],
                          "label": row["label"], "score": float(score),
                          "flag_at_1pct": bool(score >= cutoffs["1"]),
                          "flag_at_2pct": bool(score >= cutoffs["2"])})
        if name == "with_char_fluency":
            enhanced = fitted
            enhanced_threshold = cutoffs["1"]
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": enhanced, "fluency": fluency,
                 "v3_artifact_sha256": hashlib.sha256(args.v3.read_bytes()).hexdigest(),
                 "threshold": enhanced_threshold,
                 "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
                args.model_output)
    report = {
        "status": "exploratory development; architecture chosen after prior failure analysis",
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "v9_clean_data_sha256": v9_manifest["train_sha256"],
        "char_lm_clean_documents": len(clean_texts),
        "v3_artifact_sha256": hashlib.sha256(args.v3.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model_output.read_bytes()).hexdigest(),
        "models": reports, "runtime_seconds": round(time.perf_counter() - started, 2),
        "limitations": manifest["limitations"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    case_path = args.report.with_suffix(".cases.jsonl")
    with case_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in cases:
            sink.write(json.dumps(row) + "\n")
    report["case_rows_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: item["evaluation_at_fpr_target"]["1"]
                      for name, item in reports.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
