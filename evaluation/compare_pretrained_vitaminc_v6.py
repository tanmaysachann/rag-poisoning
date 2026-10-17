"""Development-only comparison with the VitaminC authors' pretrained ALBERT.

The sealed test split is not read. This is a model-selection diagnostic after
the locally trained MiniLM failed the 1% false-quarantine development gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.research_detector import threshold_for_fpr
from detect.train_factual_verifier_v6 import LABEL_TO_ID, predict, read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/factual_verifier_v6"))
    parser.add_argument("--model", type=Path,
                        default=Path("artifacts/vendor/albert-base-vitaminc"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/factual_verifier_v6_pretrained_dev.json"))
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for name in ("validation", "calibration"):
        path = args.data / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][path.name]:
            raise ValueError(f"Development data changed: {path}")
    validation = read_rows(args.data / "validation.jsonl")
    calibration = read_rows(args.data / "calibration.jsonl")
    if {row["page"] for row in validation} & {row["page"] for row in calibration}:
        raise ValueError("Validation and calibration pages overlap")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).to(device)
    if {int(key): value for key, value in model.config.id2label.items()} != {
        index: label for label, index in LABEL_TO_ID.items()
    }:
        raise ValueError("Unexpected pretrained model label order")
    started = time.perf_counter()
    # The source model's processor orders the pair as evidence, then claim.
    val_prob = predict(model, tokenizer, validation, device, 16, 256, evidence_first=True)
    cal_prob = predict(model, tokenizer, calibration, device, 16, 256, evidence_first=True)
    val_scores = val_prob[:, LABEL_TO_ID["REFUTES"]].astype(np.float64)
    cal_scores = cal_prob[:, LABEL_TO_ID["REFUTES"]].astype(np.float64)
    val_labels = np.asarray([row["label"] for row in validation])
    cal_labels = np.asarray([row["label"] for row in calibration])
    by_label_threshold = {
        label: threshold_for_fpr(val_scores[val_labels == label], 0.01)
        for label in ("SUPPORTS", "NOT ENOUGH INFO")
    }
    threshold = max(by_label_threshold.values())
    report = {
        "model": "tals/albert-base-vitaminc",
        "task": "development-only trusted-evidence factual verification",
        "model_weight_sha256": hashlib.sha256((args.model / "pytorch_model.bin").read_bytes()).hexdigest(),
        "validation_calibration_threshold": threshold,
        "threshold_fpr_target_per_non_refuted_class": 0.01,
        "calibration_rates": {
            label: {
                "count": int(np.sum(cal_scores[cal_labels == label] >= threshold)),
                "total": int(np.sum(cal_labels == label)),
                "rate": float(np.mean(cal_scores[cal_labels == label] >= threshold)),
            } for label in LABEL_TO_ID
        },
        "validation_accuracy": float(np.mean(
            val_prob.argmax(axis=1) == np.asarray([LABEL_TO_ID[x] for x in val_labels])
        )),
        "calibration_accuracy": float(np.mean(
            cal_prob.argmax(axis=1) == np.asarray([LABEL_TO_ID[x] for x in cal_labels])
        )),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitation": "Same dataset family used in this published model's training; not an independent RAG-poison evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
