"""Explore clean-calibrated score fusion between the two V20 detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from detect.research_detector import ResearchDetector, threshold_for_fpr
from detect.train_multifamily_v20 import read_verified_rows
from detect.train_multisignal_v20 import extract
from evaluation.evaluate_multifamily_v20 import score_rows, summarize


def clean_tail_probability(scores: np.ndarray, clean_reference: np.ndarray) -> np.ndarray:
    """Smoothed empirical upper-tail probability from calibration-clean scores."""
    sorted_clean = np.sort(np.asarray(clean_reference, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    rank_below = np.searchsorted(sorted_clean, scores, side="left")
    return (len(sorted_clean) - rank_below + 1) / (len(sorted_clean) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/multifamily_v20"))
    parser.add_argument("--reference-model", type=Path,
                        default=Path("artifacts/multifamily_v20"))
    parser.add_argument("--signal-model", type=Path,
                        default=Path("artifacts/multisignal_v20.joblib"))
    parser.add_argument("--v3", type=Path,
                        default=Path("artifacts/research_detector_v3_hashing.joblib"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/fusion_v20.json"))
    args = parser.parse_args()
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = read_verified_rows(args.data, "calibration", manifest)
    evaluation = read_verified_rows(args.data, "evaluation", manifest)
    cal_clean = np.asarray([row["label"] != "poison" for row in calibration])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    scorer = ReferenceConditionedScorer(
        args.reference_model, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    neural_cal = score_rows(scorer, calibration, 24)
    neural_eval = score_rows(scorer, evaluation, 24)
    signal_artifact = joblib.load(args.signal_model)
    v3 = joblib.load(args.v3)
    if not isinstance(v3, ResearchDetector):
        raise TypeError("Unexpected V3 artifact")
    if signal_artifact["v3_artifact_sha256"] != hashlib.sha256(args.v3.read_bytes()).hexdigest():
        raise ValueError("Signal model's V3 feature extractor changed")
    signal_cal = signal_artifact["model"].predict_proba(extract(calibration, v3))[:, 1]
    signal_eval = signal_artifact["model"].predict_proba(extract(evaluation, v3))[:, 1]
    neural_tail_cal = clean_tail_probability(neural_cal, neural_cal[cal_clean])
    neural_tail_eval = clean_tail_probability(neural_eval, neural_cal[cal_clean])
    signal_tail_cal = clean_tail_probability(signal_cal, signal_cal[cal_clean])
    signal_tail_eval = clean_tail_probability(signal_eval, signal_cal[cal_clean])
    # High risk when either independently calibrated model sees an extreme score.
    fusion_cal = -np.log(np.minimum(neural_tail_cal, signal_tail_cal))
    fusion_eval = -np.log(np.minimum(neural_tail_eval, signal_tail_eval))
    thresholds = {str(int(rate * 100)): threshold_for_fpr(
        fusion_cal[cal_clean], rate) for rate in (0.01, 0.02)}
    report = {
        "method": "maximum clean-calibrated upper-tail surprise of MiniLM and gradient boosting",
        "status": "exploratory development: component evaluation on these questions was inspected first",
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "reference_model_sha256": hashlib.sha256(
            (args.reference_model / "model.safetensors").read_bytes()).hexdigest(),
        "signal_model_sha256": hashlib.sha256(args.signal_model.read_bytes()).hexdigest(),
        "calibration_clean_total": int(cal_clean.sum()),
        "calibration_clean_flags": {
            rate: int(np.sum(fusion_cal[cal_clean] >= threshold))
            for rate, threshold in thresholds.items()},
        "evaluation_at_fpr_target": {
            rate: summarize(evaluation, fusion_eval, threshold)
            for rate, threshold in thresholds.items()},
        "limitations": manifest["limitations"] + [
            "No new blind attack collection or downstream answer replay was evaluated.",
            "Fusion was constructed after component scores on the development questions were seen.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases_path = args.output.with_suffix(".cases.jsonl")
    with cases_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row, a, b, risk in zip(evaluation, neural_eval, signal_eval, fusion_eval):
            sink.write(json.dumps({
                "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                "family": row["family"], "label": row["label"],
                "minilm_risk": float(a), "signal_risk": float(b),
                "fusion_risk": float(risk),
                "flag_at_1pct": bool(risk >= thresholds["1"]),
                "flag_at_2pct": bool(risk >= thresholds["2"]),
            }) + "\n")
    report["case_rows_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation_at_fpr_target"], indent=2), flush=True)


if __name__ == "__main__":
    main()
