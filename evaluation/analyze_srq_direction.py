"""Post-hoc validation diagnostic for SRQ score direction and components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import RESULTS_DIR
from detect.research_detector import threshold_for_fpr


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "slm_srq_validation_50_75.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "slm_srq_direction_diagnostic.json")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.summary.with_suffix(".cases.jsonl"))
    groups = {name: [row for row in rows if row["group"] == name]
              for name in ("calibration_clean", "validation_clean", "validation_greedy", "validation_stealth")}
    if len(groups["calibration_clean"]) != summary["calibration_clean"] or any(
        len(groups[name]) != summary["validation_clean"]
        for name in ("validation_clean", "validation_greedy", "validation_stealth")
    ):
        raise ValueError("SRQ summary and cases disagree")
    report = {
        "source": str(args.summary), "validation_cases_per_family": summary["validation_clean"],
        "target_train_fpr": 0.05,
        "note": "Exploratory, post-hoc validation analysis across six score/direction choices; do not report a selected direction as an untouched test result",
        "signals": {},
    }
    for field in ("raw_srq", "query_contribution", "response_contribution"):
        report["signals"][field] = {}
        for direction, factor in (("high", 1), ("low", -1)):
            train = np.asarray([factor * row[field] for row in groups["calibration_clean"]])
            clean = np.asarray([factor * row[field] for row in groups["validation_clean"]])
            threshold = threshold_for_fpr(train, 0.05)
            families = {}
            for family in ("greedy", "stealth"):
                attack = np.asarray([factor * row[field] for row in groups[f"validation_{family}"]])
                families[family] = {
                    "recall": float(np.mean(attack >= threshold)),
                    "roc_auc": float(roc_auc_score(
                        [0] * len(clean) + [1] * len(attack),
                        np.concatenate([clean, attack]),
                    )),
                }
            report["signals"][field][direction] = {
                "threshold": float(threshold),
                "clean_validation_fpr": float(np.mean(clean >= threshold)),
                "families": families,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
