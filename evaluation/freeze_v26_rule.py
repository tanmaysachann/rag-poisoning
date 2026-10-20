"""Pin the V26 scanner and FiQA schedule before the first holdout score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v26_rule_frozen.json"
INPUTS = (
    "detect/product_v26.py", "detect/role_boundary_v26.py",
    "detect/product_v25.py", "detect/product_v22.py",
    "artifacts/role_boundary_v26.joblib",
    "artifacts/multifamily_v20/model.safetensors",
    "artifacts/product_v24_calibration.joblib",
    "artifacts/research_detector_v3_hashing_web.joblib",
    "results/v25_rule_frozen.json", "results/v26_role_development.json",
    "data/build_v26_fiqa_holdout.py",
    "data/v26_fiqa_holdout/manifest.json", "data/v26_fiqa_holdout/cases.jsonl",
    "evaluation/evaluate_v26_fiqa.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    if OUTPUT.exists() or (ROOT / "results/v26_fiqa_holdout.cases.jsonl").exists():
        raise FileExistsError("V26 has already been frozen or FiQA scoring started")
    manifest = json.loads((ROOT / "data/v26_fiqa_holdout/manifest.json").read_text(encoding="utf-8"))
    dev = json.loads((ROOT / "results/v26_role_development.json").read_text(encoding="utf-8"))
    if manifest["cases_sha256"] != sha(ROOT / "data/v26_fiqa_holdout/cases.jsonl"):
        raise ValueError("FiQA cases changed after dataset freeze")
    if dev["artifact_sha256"] != sha(ROOT / "artifacts/role_boundary_v26.joblib"):
        raise ValueError("Role model changed after development report")
    rule = {
        "protocol": "v26-role-adaptive-frozen-rule-v1",
        "status": "frozen_before_fiqa_holdout_scoring",
        "decision": "V25 decision OR (system-role marker AND trained role-boundary probability >= development-calibrated threshold)",
        "role_threshold": dev["threshold"],
        "development": {"source": "inspected NFCorpus + synthetic role and benign markup edits",
                        "train_questions": 80, "calibration_questions": 40,
                        "role_recall": dev["role_recall"],
                        "clean_false_alarm": dev["clean_false_alarm"]},
        "holdout": {"dataset": "official BEIR FiQA test", "questions": manifest["questions"],
                    "cases": manifest["rows"], "first_scoring_pending": True},
        "input_sha256": {name: sha(ROOT / name) for name in INPUTS},
        "limits": [
            "The FiQA questions and finance domain are new to V26 calibration; the attack templates were predeclared and are synthetic.",
            "NFCorpus was inspected before this rule and is development evidence.",
            "Relevant FiQA references are not verified factual truth.",
            "The role model is supervised logistic regression, not reinforcement learning.",
        ],
    }
    OUTPUT.write_text(json.dumps(rule, indent=2) + "\n", encoding="utf-8")
    return rule


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
