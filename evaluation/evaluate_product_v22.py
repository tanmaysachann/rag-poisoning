"""Combine frozen learned and snapshot alerts into a reproducible dev report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from detect.train_multifamily_v20 import read_verified_rows
from evaluation.evaluate_multifamily_v20 import summarize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cases(path: Path, expected_hash: str) -> list[dict]:
    if digest(path) != expected_hash:
        raise ValueError(f"Case file changed: {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    data = Path("data/multifamily_v20")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_verified_rows(data, "evaluation", manifest)
    learned_path = Path("results/fusion_v21.json")
    integrity_path = Path("results/snapshot_near_duplicate_v22.json")
    learned_report = json.loads(learned_path.read_text(encoding="utf-8"))
    integrity_report = json.loads(integrity_path.read_text(encoding="utf-8"))
    learned = read_cases(learned_path.with_suffix(".cases.jsonl"),
                         learned_report["case_rows_sha256"])
    integrity = read_cases(integrity_path.with_suffix(".cases.jsonl"),
                           integrity_report["case_rows_sha256"])
    calibration_path = Path("artifacts/product_v22_calibration.joblib")
    calibration = joblib.load(calibration_path)
    if len(rows) != len(learned) or len(rows) != len(integrity):
        raise ValueError("Case count mismatch")
    if calibration["fusion_report_sha256"] != digest(learned_path):
        raise ValueError("Calibration and learned report mismatch")
    if calibration["snapshot_manifest_sha256"] != integrity_report["snapshot_manifest_sha256"]:
        raise ValueError("Snapshot mismatch")
    for row, a, b in zip(rows, learned, integrity):
        expected = (row["query_id"], row["candidate_id"], row["family"], row["label"])
        if any(tuple(part[key] for key in ("query_id", "candidate_id", "family", "label"))
               != expected for part in (a, b)):
            raise ValueError("Case alignment mismatch")
    flags = np.asarray([a["flag_at_1pct"] or b["flag_at_1pct"]
                        for a, b in zip(learned, integrity)], dtype=np.float64)
    both = np.asarray([a["flag_at_1pct"] and b["flag_at_1pct"]
                       for a, b in zip(learned, integrity)], dtype=np.float64)
    report = {
        "product": "V22 local poisoned-document research scanner",
        "status": "exploratory development, not blind or live serving",
        "data_manifest_sha256": digest(manifest_path),
        "learned_report_sha256": digest(learned_path),
        "integrity_report_sha256": digest(integrity_path),
        "calibration_artifact_sha256": digest(calibration_path),
        "calibration_clean_total": calibration["calibration_clean_total"],
        "calibration_clean_flags": calibration["calibration_fusion_clean_flags"] +
            integrity_report["calibration_clean_flags"]["1"],
        "learned_only": learned_report["evaluation_at_fpr_target"]["1"],
        "integrity_only": integrity_report["evaluation_at_fpr_target"]["1"],
        "combined": summarize(rows, flags, 0.5),
        "both_alerts": summarize(rows, both, 0.5),
        "limits": [
            "Question-disjoint development rows, but these questions were previously inspected.",
            "Labels identify published attack passages, not downstream answer compromise.",
            "The reference must be authenticated by the caller; BEIR qrels alone do not do that.",
            "The integrity branch needs a separate, pinned pre-attack snapshot.",
            "An exact corpus match does not establish truth or safety.",
        ],
    }
    output = Path("results/product_v22.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["combined"], indent=2))


if __name__ == "__main__":
    main()
