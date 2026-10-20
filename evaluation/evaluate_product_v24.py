"""Audit V24 learned-plus-containment decisions on six-family development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from detect.train_multifamily_v20 import read_verified_rows
from evaluation.evaluate_multifamily_v20 import summarize


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = Path("data/multifamily_v20")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_verified_rows(data, "evaluation", manifest)
    learned_path = Path("results/fusion_v21.json")
    learned_report = json.loads(learned_path.read_text(encoding="utf-8"))
    containment_path = Path("results/snapshot_containment_v23.json")
    containment_report = json.loads(containment_path.read_text(encoding="utf-8"))
    calibration_path = Path("artifacts/product_v24_calibration.joblib")
    calibration = joblib.load(calibration_path)
    if (calibration["fusion_report_sha256"] != sha(learned_path) or
            calibration["containment_report_sha256"] != sha(containment_path)):
        raise ValueError("Calibration evidence changed")
    learned_cases_path = learned_path.with_suffix(".cases.jsonl")
    containment_cases_path = containment_path.with_suffix(".cases.jsonl")
    if (sha(learned_cases_path) != learned_report["case_rows_sha256"] or
            sha(containment_cases_path) != containment_report["case_rows_sha256"]):
        raise ValueError("Case evidence changed")
    learned = [json.loads(line) for line in learned_cases_path.open(encoding="utf-8")]
    containment = [json.loads(line) for line in containment_cases_path.open(encoding="utf-8")]
    if len(rows) != len(learned) or len(rows) != len(containment):
        raise ValueError("Case count mismatch")
    flags = []
    for row, a, b in zip(rows, learned, containment):
        key = ("query_id", "candidate_id", "family", "label")
        if tuple(row[k] for k in key) != tuple(a[k] for k in key) or tuple(
            row[k] for k in key) != tuple(b[k] for k in key):
            raise ValueError("Case alignment mismatch")
        flags.append(a["flag_at_1pct"] or b["flag"])
    tp = sum(flag and row["label"] == "poison" for row, flag in zip(rows, flags))
    fp = sum(flag and row["label"] != "poison" for row, flag in zip(rows, flags))
    fn = sum(not flag and row["label"] == "poison" for row, flag in zip(rows, flags))
    tn = sum(not flag and row["label"] != "poison" for row, flag in zip(rows, flags))
    report = {
        "product": "V24 local source-label and integrity scanner",
        "status": "exploratory development; adaptive hardening after PPO failures",
        "data_manifest_sha256": sha(manifest_path),
        "learned_report_sha256": sha(learned_path),
        "containment_report_sha256": sha(containment_path),
        "calibration_artifact_sha256": sha(calibration_path),
        "calibration_clean_total": calibration["calibration_clean_total"],
        "calibration_clean_flags": (calibration["calibration_fusion_clean_flags"] +
                                    containment_report["calibration_clean_flags"]),
        "learned_only": learned_report["evaluation_at_fpr_target"]["1"],
        "containment_only": containment_report["evaluation"],
        "combined": summarize(rows, np.asarray(flags, dtype=np.float64), 0.5),
        "classification": {
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "binary_accuracy": (tp + tn) / len(rows),
            "precision_at_this_attack_prevalence": tp / (tp + fp),
            "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
            "majority_class_accuracy": max(tp + fn, tn + fp) / len(rows),
        },
        "identity_rule": "optional exact expected SHA-256 for a known source ID; no such authenticated hash in these rows",
        "limits": [
            "PoisonArena questions had previously been inspected; this is not blind.",
            "Source labels are not independent factual or answer-impact judgments.",
            "The reference and optional expected digest require external authentication.",
            "The snapshot is a pinned corpus, not proof of factual truth.",
            "No six-family downstream answer replay was done for this scanner.",
        ],
    }
    output = Path("results/product_v24.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["combined"], indent=2))


if __name__ == "__main__":
    main()
