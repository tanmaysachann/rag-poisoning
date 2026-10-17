"""Evaluate the frozen factual verifier on untouched VitaminC test pages.

This reports evidence contradiction, not RAG poison detection or answer success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detect.train_factual_verifier_v6 import LABELS, LABEL_TO_ID, predict, read_rows
from evaluation.metrics import wilson_interval


def _rate(count: int, total: int) -> dict:
    low, high = wilson_interval(count, total)
    return {"count": count, "total": total, "rate": count / total,
            "wilson_95": {"lower": low, "upper": high}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/factual_verifier_v6"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/factual_verifier_v6"))
    parser.add_argument("--training-report", type=Path,
                        default=Path("results/factual_verifier_v6_training.json"))
    parser.add_argument("--output", type=Path, default=Path("results/factual_verifier_v6_test.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    manifest_path = args.data / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != training["data_manifest_sha256"]:
        raise ValueError("Dataset manifest changed after training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_path = args.data / "test.jsonl"
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != manifest["file_sha256"]["test.jsonl"]:
        raise ValueError("Frozen test file changed")
    rows = read_rows(test_path)
    train_pages = {row["page"] for row in read_rows(args.data / "train.jsonl")}
    dev_pages = {row["page"] for name in ("validation", "calibration")
                 for row in read_rows(args.data / f"{name}.jsonl")}
    if {row["page"] for row in rows} & (train_pages | dev_pages):
        raise ValueError("Test page appears in train/development data")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).to(device)
    started = time.perf_counter()
    probabilities = predict(model, tokenizer, rows, device, args.batch_size,
                            int(training["max_length"]))
    elapsed = time.perf_counter() - started
    truth = np.asarray([LABEL_TO_ID[row["label"]] for row in rows])
    predicted = probabilities.argmax(axis=1)
    refute_scores = probabilities[:, LABEL_TO_ID["REFUTES"]].astype(np.float64)
    quarantined = refute_scores >= float(training["quarantine_threshold"])
    by_label = {}
    for label in LABELS:
        mask = truth == LABEL_TO_ID[label]
        by_label[label] = {
            "classification_accuracy": _rate(int(np.sum(predicted[mask] == truth[mask])), int(np.sum(mask))),
            "quarantine_rate": _rate(int(np.sum(quarantined[mask])), int(np.sum(mask))),
        }
    report = {
        "task": "trusted-evidence claim verification; not RAG poison detection",
        "test_data_sha256": manifest["file_sha256"]["test.jsonl"],
        "training_report_sha256": hashlib.sha256(args.training_report.read_bytes()).hexdigest(),
        "model_weight_sha256": hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest(),
        "cases": len(rows), "distinct_test_pages": len({row["page"] for row in rows}),
        "device": str(device), "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(LABELS)))).tolist(),
        "threshold_source": "calibration only", "quarantine_threshold": training["quarantine_threshold"],
        "by_label": by_label,
        "elapsed_seconds": round(elapsed, 2),
        "mean_scoring_ms_per_pair": round(1000 * elapsed / len(rows), 3),
        "limitations": [
            "VitaminC labels factual relation to supplied evidence, not attacker intent.",
            "Trusted evidence is supplied directly; live retrieval of such evidence is not evaluated.",
            "This is one source family with Wikipedia revision language, not unseen RAG attack families.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
