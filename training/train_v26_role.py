"""Train and calibrate the V26 structural role-boundary classifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data.build_v26_role_dev import ROOT, SOURCE, cases
from detect.role_boundary_v26 import features


ARTIFACT = ROOT / "artifacts/role_boundary_v26.joblib"
REPORT = ROOT / "results/v26_role_development.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train() -> dict:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("V26 role classifier is already trained")
    rows = cases()
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "calibration"]
    model = make_pipeline(DictVectorizer(sparse=False), LogisticRegression(max_iter=1000, C=0.5))
    model.fit([features(row["reference"], row["candidate"]) for row in train_rows],
              [row["role_attack"] for row in train_rows])
    scores = model.predict_proba([features(row["reference"], row["candidate"]) for row in val_rows])[:, 1]
    negatives = [float(score) for row, score in zip(val_rows, scores) if not row["role_attack"]]
    positives = [float(score) for row, score in zip(val_rows, scores) if row["role_attack"]]
    # Threshold is calibrated only on the grouped NFCorpus development split.
    # A zero-false-alarm operating point is required on these hard negatives.
    candidates = sorted({0.5, *(float(value) for value in scores)})
    operating = []
    for threshold in candidates:
        fp = sum(value >= threshold for value in negatives)
        tp = sum(value >= threshold for value in positives)
        if fp == 0:
            operating.append((tp, -threshold, threshold))
    if not operating:
        raise RuntimeError("No zero-false-alarm threshold exists")
    threshold = max(operating)[2]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, ARTIFACT)
    report = {
        "protocol": "v26-role-boundary-nfcorpus-grouped-development-v1",
        "source_sha256": sha(SOURCE), "builder_sha256": sha(ROOT / "data/build_v26_role_dev.py"),
        "features_sha256": sha(ROOT / "detect/role_boundary_v26.py"),
        "train_questions": 80, "calibration_questions": 40,
        "train_rows": len(train_rows), "calibration_rows": len(val_rows),
        "calibration_positive": len(positives), "calibration_negative": len(negatives),
        "threshold": threshold,
        "role_recall": {"flagged": sum(value >= threshold for value in positives), "total": len(positives)},
        "clean_false_alarm": {"flagged": sum(value >= threshold for value in negatives), "total": len(negatives)},
        "positive_score_min": min(positives), "negative_score_max": max(negatives),
        "artifact_sha256": sha(ARTIFACT),
        "limits": ["Synthetic role attacks and quoted-markup controls on inspected NFCorpus questions.",
                   "Threshold selected on 40 grouped development questions; FiQA was not scored."],
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
