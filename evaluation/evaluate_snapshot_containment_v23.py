"""Evaluate containment hardening after PPO exposed inserted-prefix evasion."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from detect.research_detector import threshold_for_fpr
from detect.snapshot_containment_v23 import SnapshotContainment
from detect.train_multifamily_v20 import read_verified_rows
from evaluation.evaluate_multifamily_v20 import summarize


def main() -> None:
    data = Path("data/multifamily_v20")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    calibration = read_verified_rows(data, "calibration", manifest)
    evaluation = read_verified_rows(data, "evaluation", manifest)
    started = time.perf_counter()
    scanner = SnapshotContainment(Path("data/trusted_snapshot_v12"))
    try:
        assessments = []
        for index, row in enumerate(calibration + evaluation, 1):
            assessments.append(scanner.assess(row["candidate"]))
            if index % 250 == 0:
                print(f"snapshot containment {index}/{len(calibration)+len(evaluation)}", flush=True)
    finally:
        scanner.close()
    cal = np.asarray([item["near_copy_score"] for item in assessments[:len(calibration)]])
    scores = np.asarray([item["near_copy_score"] for item in assessments[len(calibration):]])
    clean = np.asarray([row["label"] != "poison" for row in calibration])
    threshold = max(0.9, threshold_for_fpr(cal[clean], 0.01))
    output = Path("results/snapshot_containment_v23.json")
    cases = output.with_suffix(".cases.jsonl")
    with cases.open("w", encoding="utf-8", newline="\n") as sink:
        for row, assessment in zip(evaluation, assessments[len(calibration):]):
            sink.write(json.dumps({
                "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                "family": row["family"], "label": row["label"],
                **assessment, "flag": assessment["near_copy_score"] >= threshold,
            }) + "\n")
    report = {
        "status": "adaptive development after PPO insertion bypass; not blind",
        "method": "opening/middle/ending FTS retrieval and substantial source containment",
        "snapshot_manifest_sha256": hashlib.sha256(
            Path("data/trusted_snapshot_v12/manifest.json").read_bytes()).hexdigest(),
        "threshold": threshold,
        "calibration_clean_flags": int(np.sum(cal[clean] >= threshold)),
        "calibration_clean_total": int(clean.sum()),
        "evaluation": summarize(evaluation, scores, threshold),
        "case_rows_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limits": [
            "The source snapshot is pinned, not proof of factual truth.",
            "A benign document quoting a full source passage could alert.",
            "These development questions had already been inspected.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation"], indent=2), flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']}", flush=True)


if __name__ == "__main__":
    main()
