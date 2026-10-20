"""Train a compact six-family detector from the frozen V3 signal extractor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detect.research_detector import FEATURE_NAMES, ResearchDetector, threshold_for_fpr
from detect.train_multifamily_v20 import read_verified_rows
from evaluation.evaluate_multifamily_v20 import summarize


def extract(rows: list[dict], detector: ResearchDetector) -> np.ndarray:
    features = []
    for index, row in enumerate(rows, 1):
        signal = detector.features(row["query"], row["candidate"])
        v3_probability = detector.classifier.predict_proba(signal.reshape(1, -1))[0, 1]
        features.append(np.append(signal, v3_probability))
        if index % 500 == 0:
            print(f"features {index}/{len(rows)}", flush=True)
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (len(rows), len(FEATURE_NAMES) + 1) or not np.isfinite(result).all():
        raise ValueError("Invalid signal matrix")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/multifamily_v20"))
    parser.add_argument("--v3", type=Path,
                        default=Path("artifacts/research_detector_v3_hashing.joblib"))
    parser.add_argument("--model-output", type=Path,
                        default=Path("artifacts/multisignal_v20.joblib"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/multisignal_v20.json"))
    args = parser.parse_args()
    if args.model_output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite V20 model or report")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {split: read_verified_rows(args.data, split, manifest)
            for split in ("train", "calibration", "evaluation")}
    qids = {split: {row["query_id"] for row in part} for split, part in rows.items()}
    if any(qids[a] & qids[b] for a in qids for b in qids if a != b):
        raise ValueError("Question leakage")
    started = time.perf_counter()
    detector = joblib.load(args.v3)
    if not isinstance(detector, ResearchDetector):
        raise TypeError("Unexpected V3 artifact")
    x = {split: extract(part, detector) for split, part in rows.items()}
    y = {split: np.asarray([row["label"] == "poison" for row in part], dtype=np.int8)
         for split, part in rows.items()}
    candidates = {
        "v3_frozen": None,
        "weighted_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=3000, random_state=20261020,
            ),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=7, learning_rate=0.05,
            l2_regularization=1.0, class_weight="balanced",
            random_state=20261020,
        ),
    }
    report = {"data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "v3_artifact_sha256": hashlib.sha256(args.v3.read_bytes()).hexdigest(),
              "features": [*FEATURE_NAMES, "v3_risk"], "models": {},
              "selection": "highest calibration macro-family recall at <=1% clean FPR; "
                           "ties prefer simpler model",
              "limitations": manifest["limitations"] + [
                  "The 29 evaluation questions were previously inspected in V10/V14.",
                  "Source labels do not measure downstream answer compromise.",
              ]}
    case_rows = []
    selected = None
    selected_merit = -1.0
    for name, model in candidates.items():
        if model is not None:
            model.fit(x["train"], y["train"])
        scores = {
            split: (x[split][:, -1] if model is None else
                    model.predict_proba(x[split])[:, 1])
            for split in ("calibration", "evaluation")
        }
        clean_mask = y["calibration"] == 0
        cutoffs = {str(int(rate * 100)): threshold_for_fpr(
            scores["calibration"][clean_mask], rate)
            for rate in (0.01, 0.02)}
        calibration_1 = summarize(rows["calibration"], scores["calibration"], cutoffs["1"])
        merit = float(np.mean([value["recall"] for value in
                               calibration_1["families"].values()]))
        if merit > selected_merit:
            selected_merit = merit
            selected = name
        report["models"][name] = {
            "calibration_at_1pct": calibration_1,
            "evaluation_roc_auc": float(roc_auc_score(y["evaluation"], scores["evaluation"])),
            "evaluation_pr_auc": float(average_precision_score(
                y["evaluation"], scores["evaluation"])),
            "evaluation_at_fpr_target": {
                rate: summarize(rows["evaluation"], scores["evaluation"], cutoff)
                for rate, cutoff in cutoffs.items()
            },
        }
        for row, score in zip(rows["evaluation"], scores["evaluation"]):
            case_rows.append({"model": name, "query_id": row["query_id"],
                              "candidate_id": row["candidate_id"],
                              "label": row["label"], "family": row["family"],
                              "score": float(score),
                              "flag_at_1pct": bool(score >= cutoffs["1"]),
                              "flag_at_2pct": bool(score >= cutoffs["2"])})
    report["selected_model"] = selected
    report["selected_calibration_macro_recall"] = selected_merit
    report["runtime_seconds"] = round(time.perf_counter() - started, 2)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": candidates[selected],
                 "feature_names": [*FEATURE_NAMES, "v3_risk"],
                 "threshold": report["models"][selected]["calibration_at_1pct"]["threshold"],
                 "data_manifest_sha256": report["data_manifest_sha256"],
                 "v3_artifact_sha256": report["v3_artifact_sha256"]}, args.model_output)
    report["selected_model_sha256"] = hashlib.sha256(args.model_output.read_bytes()).hexdigest()
    case_path = args.report.with_suffix(".cases.jsonl")
    with case_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in case_rows:
            sink.write(json.dumps(row) + "\n")
    report["case_rows_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_model": selected,
                      "evaluations": {name: value["evaluation_at_fpr_target"]["1"]
                                      for name, value in report["models"].items()}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
