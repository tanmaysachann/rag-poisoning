"""Evaluate altered-copy integrity evidence against a pinned BEIR NQ snapshot."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from detect.research_detector import threshold_for_fpr
from detect.snapshot_near_duplicate import SnapshotNearDuplicate
from detect.train_multifamily_v20 import read_verified_rows
from evaluation.evaluate_multifamily_v20 import summarize


def main() -> None:
    data = Path("data/multifamily_v20")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    calibration = read_verified_rows(data, "calibration", manifest)
    evaluation = read_verified_rows(data, "evaluation", manifest)
    snapshot_path = Path("data/trusted_snapshot_v12")
    started = time.perf_counter()
    scanner = SnapshotNearDuplicate(snapshot_path)
    try:
        assessments = [scanner.assess(row["candidate"]) for row in calibration + evaluation]
    finally:
        scanner.close()
    cal_assess = assessments[:len(calibration)]
    eval_assess = assessments[len(calibration):]
    cal_scores = np.asarray([item["near_copy_score"] for item in cal_assess])
    eval_scores = np.asarray([item["near_copy_score"] for item in eval_assess])
    clean_cal = np.asarray([row["label"] != "poison" for row in calibration])
    thresholds = {
        str(int(rate * 100)): max(0.9, threshold_for_fpr(cal_scores[clean_cal], rate))
        for rate in (0.01, 0.02)
    }
    report = {
        "method": "changed near-copy of a separate hash-pinned 2.68M-passage snapshot",
        "status": "exploratory development; snapshot source is not authenticated truth",
        "snapshot_manifest_sha256": hashlib.sha256(
            (snapshot_path / "manifest.json").read_bytes()).hexdigest(),
        "calibration_clean_total": int(clean_cal.sum()),
        "calibration_clean_flags": {
            rate: int(np.sum(cal_scores[clean_cal] >= threshold))
            for rate, threshold in thresholds.items()
        },
        "calibration_clean_nonexact": int(sum(
            not item["exact_snapshot_match"] for item, clean
            in zip(cal_assess, clean_cal) if clean)),
        "evaluation_at_fpr_target": {
            rate: summarize(evaluation, eval_scores, threshold)
            for rate, threshold in thresholds.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitations": manifest["limitations"] + [
            "This integrity check requires a separate, pre-attack pinned corpus.",
            "A changed close copy may be benign; exact matches may still be false or malicious.",
            "The development questions were inspected during detector development.",
        ],
    }
    by_family = defaultdict(list)
    for row, assessment in zip(evaluation, eval_assess):
        by_family[row["family"]].append(assessment)
    report["family_coverage"] = {
        family: {
            "exact_matches": sum(a["exact_snapshot_match"] for a in items),
            "no_neighbor": sum(a["status"] == "no_snapshot_neighbor" for a in items),
            "max_score": max(a["near_copy_score"] for a in items),
        } for family, items in sorted(by_family.items())
    }
    output = Path("results/snapshot_near_duplicate_v22.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = output.with_suffix(".cases.jsonl")
    with cases.open("w", encoding="utf-8", newline="\n") as sink:
        for row, assessment in zip(evaluation, eval_assess):
            sink.write(json.dumps({
                "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                "family": row["family"], "label": row["label"],
                **assessment,
                "flag_at_1pct": assessment["near_copy_score"] >= thresholds["1"],
                "flag_at_2pct": assessment["near_copy_score"] >= thresholds["2"],
            }) + "\n")
    report["case_rows_sha256"] = hashlib.sha256(cases.read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation_at_fpr_target"], indent=2), flush=True)
    print(f"elapsed_seconds={report['elapsed_seconds']}", flush=True)


if __name__ == "__main__":
    main()
