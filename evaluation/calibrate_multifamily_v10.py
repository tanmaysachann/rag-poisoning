"""Freeze a v10 threshold using only the original clean v7 calibration cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from detect.research_detector import threshold_for_fpr
from evaluation.evaluate_external_rag_v7 import read_cases, summary
from evaluation.evaluate_reference_detector_v9 import clean_scores, score_cases


def main() -> None:
    data = Path("data/external_rag_v7")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    cases = read_cases(data / "calibration.jsonl",
                       manifest["file_sha256"]["calibration.jsonl"])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model_path = Path("artifacts/multifamily_v10")
    scorer = ReferenceConditionedScorer(
        model_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scores, seconds, pairs = score_cases(cases, scorer, 24)
    threshold = threshold_for_fpr(clean_scores(cases, scores), 0.01)
    report = {"protocol": "research_protocols/multifamily_detector_v10.md",
              "training_manifest_sha256": hashlib.sha256(
                  Path("data/multifamily_v10/manifest.json").read_bytes()).hexdigest(),
              "calibration_manifest_sha256": hashlib.sha256(
                  (data / "manifest.json").read_bytes()).hexdigest(),
              "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
              "model_weights_sha256": hashlib.sha256(
                  (model_path / "model.safetensors").read_bytes()).hexdigest(),
              "threshold": threshold, "target_clean_fpr": 0.01,
              "calibration": summary(cases, scores, threshold),
              "pairs": pairs, "seconds": round(seconds, 2)}
    output = Path("results/multifamily_v10_calibration.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
