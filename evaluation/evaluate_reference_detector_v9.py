"""Calibrate and replay v9 on the known, question-disjoint external archive.

This is development evidence, not a fresh sealed benchmark. AK is an unseen
training family but was already inspected in earlier v7/v8 experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from detect.reference_conditioned_v9 import ReferenceConditionedScorer
from detect.research_detector import threshold_for_fpr
from evaluation.evaluate_external_rag_v7 import read_cases, summary


def score_cases(cases: list[dict], scorer: ReferenceConditionedScorer,
                batch_size: int) -> tuple[dict, float, int]:
    rows: dict[tuple[str, str], dict] = {}
    for case in cases:
        for retrieval in case["retrievals"].values():
            for candidate in retrieval["candidates"]:
                key = (case["query_id"], candidate["id"])
                row = {"query": case["query"], "reference": case["reference_text"],
                       "candidate": candidate["text"]}
                if key in rows and rows[key] != row:
                    raise ValueError(f"Candidate changed across retrievals: {key}")
                rows[key] = row
    scores = {}
    keys = list(rows)
    started = time.perf_counter()
    for start in range(0, len(keys), 384):
        part = keys[start:start + 384]
        result = scorer.score([rows[key] for key in part], batch_size=batch_size)
        scores.update((key, float(value)) for key, value in zip(part, result))
        print(f"scored {start + len(part)}/{len(keys)} candidate pairs", flush=True)
    return scores, time.perf_counter() - started, len(keys)


def clean_scores(cases: list[dict], scores: dict) -> np.ndarray:
    return np.asarray([scores[case["query_id"], candidate["id"]]
                       for case in cases
                       for candidate in case["retrievals"]["clean"]["candidates"]],
                      dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/external_rag_v7"))
    parser.add_argument("--model", type=Path,
                        default=Path("artifacts/reference_detector_v9"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/reference_detector_v9_replay.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = read_cases(args.data / "calibration.jsonl",
                             manifest["file_sha256"]["calibration.jsonl"])
    replay = read_cases(args.data / "test.jsonl",
                        manifest["file_sha256"]["test.jsonl"])
    if {case["query_id"] for case in calibration} & {
            case["query_id"] for case in replay}:
        raise ValueError("Calibration and replay questions overlap")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ReferenceConditionedScorer(args.model, device)
    cal_scores, cal_seconds, cal_pairs = score_cases(calibration, scorer, args.batch_size)
    threshold = threshold_for_fpr(clean_scores(calibration, cal_scores), 0.01)
    cal_summary = summary(calibration, cal_scores, threshold)
    print(f"Frozen threshold {threshold:.9f}; clean calibration flags "
          f"{cal_summary['clean_flags']}/{cal_summary['clean_total']}", flush=True)
    del cal_scores
    test_scores, test_seconds, test_pairs = score_cases(replay, scorer, args.batch_size)
    test_summary = summary(replay, test_scores, threshold)
    v7 = json.loads(Path("results/external_rag_v7.json").read_text(encoding="utf-8"))
    report = {"protocol": "research_protocols/reference_detector_v9.md",
              "evaluation_status": "development replay; v7/v8 aggregates already exposed",
              "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "training_manifest_sha256": hashlib.sha256(
                  Path("data/reference_detector_v9/manifest.json").read_bytes()).hexdigest(),
              "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "model_weights_sha256": hashlib.sha256(
                  (args.model / "model.safetensors").read_bytes()).hexdigest(),
              "device": str(device), "threshold": threshold,
              "target_clean_fpr": 0.01,
              "calibration": cal_summary, "development_replay": test_summary,
              "scoring": {"calibration_pairs": cal_pairs,
                          "calibration_seconds": round(cal_seconds, 2),
                          "test_pairs": test_pairs, "test_seconds": round(test_seconds, 2),
                          "milliseconds_per_test_pair": round(1000 * test_seconds / test_pairs, 2)},
              "frozen_v3_comparison": v7["models"]["frozen_v3_content_detector"]["test"],
              "limitation": "Training labels are provenance and the selected reference is a BEIR qrel oracle proxy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
