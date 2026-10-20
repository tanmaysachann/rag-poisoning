"""Pin post-hoc targeted FiQA role stress before scoring any case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v26_fiqa_role_stress_frozen.json"
INPUTS = (
    "results/v26_rule_frozen.json", "results/v26_fiqa_holdout.json",
    "detect/product_v26.py", "detect/role_boundary_v26.py",
    "artifacts/role_boundary_v26.joblib",
    "data/build_v26_fiqa_role_stress.py",
    "data/v26_fiqa_role_stress/manifest.json",
    "data/v26_fiqa_role_stress/cases.jsonl",
    "evaluation/evaluate_v26_role_stress.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    if OUTPUT.exists() or (ROOT / "results/v26_fiqa_role_stress.cases.jsonl").exists():
        raise FileExistsError("Follow-up role stress already frozen or scored")
    manifest = json.loads((ROOT / "data/v26_fiqa_role_stress/manifest.json").read_text(encoding="utf-8"))
    if manifest["cases_sha256"] != sha(ROOT / "data/v26_fiqa_role_stress/cases.jsonl"):
        raise ValueError("Targeted role stress cases changed")
    frozen = {"protocol": "v26-fiqa-targeted-role-stress-schedule-v1",
              "status": "followup_schedule_frozen_before_scoring",
              "design_note": "Designed after V25 and V26 both flagged 600/600 first FiQA attacks; uses unused questions and a known NFCorpus role template.",
              "input_sha256": {name: sha(ROOT / name) for name in INPUTS}}
    OUTPUT.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
