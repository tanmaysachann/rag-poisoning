"""Pin the V25 decision rule and inspected development evidence before holdout scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from detect.product_v25 import FUSION_THRESHOLD, REFERENCE_MARGIN, V3_THRESHOLD
from evaluation.develop_v25_rule import domain, metrics


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = (
        "detect/product_v25.py", "detect/product_v22.py",
        "artifacts/multifamily_v20/model.safetensors",
        "artifacts/product_v24_calibration.joblib",
        "artifacts/research_detector_v3_hashing_web.joblib",
        "results/v25_poisonarena_dev_scores.jsonl",
        "results/v24_blind_scifact.cases.jsonl",
        "data/v25_nfcorpus_holdout/manifest.json",
        "data/v25_nfcorpus_holdout/cases.jsonl",
    )
    domains = dict((domain(
        "poisonarena", ROOT / "results/v25_poisonarena_dev_scores.jsonl"),
        domain("scifact", ROOT / "results/v24_blind_scifact.cases.jsonl"),
    ))

    def rule(row: dict) -> bool:
        return bool(row["containment_flag"] or row["v3_risk"] >= V3_THRESHOLD
                    or (row["fusion_score"] >= FUSION_THRESHOLD
                        and row["fusion_score"] - row["reference_fusion_score"]
                        >= REFERENCE_MARGIN))

    report = {
        "protocol": "v25-reference-adaptive-frozen-rule-v1",
        "status": "frozen_before_nfcorpus_holdout_scoring",
        "decision": "containment OR authenticated-known-ID-mismatch OR v3-risk>=0.75 OR (v24-fusion>=4.272025739459554 AND fusion-minus-reference>=0.5)",
        "v3_threshold": V3_THRESHOLD,
        "fusion_threshold": FUSION_THRESHOLD,
        "reference_margin": REFERENCE_MARGIN,
        "development": {name: metrics(rows, rule) for name, rows in domains.items()},
        "input_sha256": {name: sha(ROOT / name) for name in paths},
        "limits": [
            "Both development sets were previously inspected when this rule was chosen.",
            "NFCorpus case contents were generated and pinned, but no model scores were read before this freeze.",
            "Synthetic source labels do not establish factual truth or RAG answer impact.",
            "The reference requires external authentication; new document admission remains a separate policy.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(freeze(ROOT / "results/v25_rule_frozen.json"), indent=2))
