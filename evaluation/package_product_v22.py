"""Freeze clean calibration for the learned V21 scanner component."""

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
from evaluation.evaluate_multifamily_v20 import score_rows


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = Path("artifacts/product_v22_calibration.joblib")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    data = Path("data/multifamily_v20")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_verified_rows(data, "calibration", manifest)
    clean = np.asarray([row["label"] != "poison" for row in rows])
    neural_path = Path("artifacts/multifamily_v20")
    signal_path = Path("artifacts/fluency_v21.joblib")
    v3_path = Path("artifacts/research_detector_v3_hashing.joblib")
    report_path = Path("results/fusion_v21.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["data_manifest_sha256"] != sha(manifest_path) or
            report["neural_model_sha256"] != sha(neural_path / "model.safetensors") or
            report["signal_model_sha256"] != sha(signal_path)):
        raise ValueError("Fusion report input hashes changed")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    scorer = ReferenceConditionedScorer(
        neural_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    neural = score_rows(scorer, rows, 24)
    v3 = joblib.load(v3_path)
    signal = joblib.load(signal_path)
    if not isinstance(v3, ResearchDetector) or signal["v3_artifact_sha256"] != sha(v3_path):
        raise ValueError("V3 signal artifact mismatch")
    features = np.hstack([
        extract(rows, v3),
        np.vstack([signal["fluency"].features(row["candidate"]) for row in rows]),
    ])
    signal_scores = signal["model"].predict_proba(features)[:, 1]
    fusion = -np.log(np.minimum(
        clean_tail_probability(neural, neural[clean]),
        clean_tail_probability(signal_scores, signal_scores[clean]),
    ))
    threshold = threshold_for_fpr(fusion[clean], 0.01)
    if not np.isclose(threshold, report["evaluation_at_fpr_target"]["1"]["threshold"]):
        raise ValueError("Recomputed fusion threshold differs from evaluated threshold")
    artifact = {
        "schema": "product-v22-calibration",
        "data_manifest_sha256": sha(manifest_path),
        "neural_model_sha256": sha(neural_path / "model.safetensors"),
        "signal_model_sha256": sha(signal_path),
        "v3_model_sha256": sha(v3_path),
        "fusion_report_sha256": sha(report_path),
        "neural_clean_scores": np.sort(neural[clean]),
        "signal_clean_scores": np.sort(signal_scores[clean]),
        "fusion_threshold_1pct": threshold,
        "snapshot_threshold": 0.9,
        "snapshot_manifest_sha256": sha(Path("data/trusted_snapshot_v12/manifest.json")),
        "calibration_clean_total": int(clean.sum()),
        "calibration_fusion_clean_flags": int(np.sum(fusion[clean] >= threshold)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    print(json.dumps({
        "artifact": str(output), "sha256": sha(output),
        "calibration_clean_total": artifact["calibration_clean_total"],
        "fusion_threshold_1pct": threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
