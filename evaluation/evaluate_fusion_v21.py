"""Explore clean-calibrated MiniLM plus fluency-signal fusion on V20 splits."""

from __future__ import annotations

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
from evaluation.evaluate_fusion_v20 import clean_tail_probability
from evaluation.evaluate_multifamily_v20 import score_rows, summarize


def main() -> None:
    data = Path("data/multifamily_v20")
    output = Path("results/fusion_v21.json")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = read_verified_rows(data, "calibration", manifest)
    evaluation = read_verified_rows(data, "evaluation", manifest)
    clean = np.asarray([row["label"] != "poison" for row in calibration])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    neural_model_path = Path("artifacts/multifamily_v20")
    scorer = ReferenceConditionedScorer(
        neural_model_path, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    neural_cal = score_rows(scorer, calibration, 24)
    neural_eval = score_rows(scorer, evaluation, 24)
    v3_path = Path("artifacts/research_detector_v3_hashing.joblib")
    v3 = joblib.load(v3_path)
    if not isinstance(v3, ResearchDetector):
        raise TypeError("Unexpected V3 artifact")
    signal_path = Path("artifacts/fluency_v21.joblib")
    signal = joblib.load(signal_path)
    if signal["v3_artifact_sha256"] != hashlib.sha256(v3_path.read_bytes()).hexdigest():
        raise ValueError("V3 signal extractor changed")

    def signal_scores(rows: list[dict]) -> np.ndarray:
        matrix = np.hstack([
            extract(rows, v3),
            np.vstack([signal["fluency"].features(row["candidate"]) for row in rows]),
        ])
        return signal["model"].predict_proba(matrix)[:, 1]

    signal_cal = signal_scores(calibration)
    signal_eval = signal_scores(evaluation)
    fusion_cal = -np.log(np.minimum(
        clean_tail_probability(neural_cal, neural_cal[clean]),
        clean_tail_probability(signal_cal, signal_cal[clean]),
    ))
    fusion_eval = -np.log(np.minimum(
        clean_tail_probability(neural_eval, neural_cal[clean]),
        clean_tail_probability(signal_eval, signal_cal[clean]),
    ))
    thresholds = {str(int(rate * 100)): threshold_for_fpr(fusion_cal[clean], rate)
                  for rate in (0.01, 0.02)}
    report = {
        "method": "maximum clean-calibrated upper-tail surprise of MiniLM and fluency-signal model",
        "status": "exploratory development after V20 evaluation inspection",
        "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "neural_model_sha256": hashlib.sha256(
            (neural_model_path / "model.safetensors").read_bytes()).hexdigest(),
        "signal_model_sha256": hashlib.sha256(signal_path.read_bytes()).hexdigest(),
        "calibration_clean_total": int(clean.sum()),
        "calibration_clean_flags": {
            rate: int(np.sum(fusion_cal[clean] >= threshold))
            for rate, threshold in thresholds.items()},
        "evaluation_at_fpr_target": {
            rate: summarize(evaluation, fusion_eval, threshold)
            for rate, threshold in thresholds.items()},
        "limitations": manifest["limitations"] + [
            "The development questions and component results were inspected before this fusion.",
            "Source labels are not downstream answer outcomes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    case_path = output.with_suffix(".cases.jsonl")
    with case_path.open("w", encoding="utf-8", newline="\n") as sink:
        for row, neural, signal_score, fusion in zip(
            evaluation, neural_eval, signal_eval, fusion_eval
        ):
            sink.write(json.dumps({
                "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                "family": row["family"], "label": row["label"],
                "neural_score": float(neural), "signal_score": float(signal_score),
                "fusion_score": float(fusion),
                "flag_at_1pct": bool(fusion >= thresholds["1"]),
                "flag_at_2pct": bool(fusion >= thresholds["2"]),
            }) + "\n")
    report["case_rows_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation_at_fpr_target"], indent=2), flush=True)


if __name__ == "__main__":
    main()
