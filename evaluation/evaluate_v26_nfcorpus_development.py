"""Revisit the inspected NFCorpus V25 role failure with frozen V26 signal.

Eighty questions supplied V26 role-model training examples; the other forty
were used for its threshold calibration. All numbers here are development
evidence and must not be presented as a new blind holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from data.build_v26_role_dev import cases
from detect.role_boundary_v26 import features


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v26_nfcorpus_development_replay.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = ROOT / "data/v25_nfcorpus_holdout/cases.jsonl"
    trace = ROOT / "results/v25_nfcorpus_holdout.cases.jsonl"
    originals = [json.loads(line) for line in source.open(encoding="utf-8")]
    scores = [json.loads(line) for line in trace.open(encoding="utf-8")]
    if len(originals) != len(scores) or len(originals) != 840:
        raise ValueError("V25 NFCorpus evidence inventory changed")
    development = cases()
    split = {row["query_id"]: row["split"] for row in development}
    payload = joblib.load(ROOT / "artifacts/role_boundary_v26.joblib")
    rows = []
    for row, prior in zip(originals, scores):
        if (row["query_id"], row["candidate_id"], hashlib.sha256(row["candidate"].encode()).hexdigest()) != (
                prior["query_id"], prior["candidate_id"], prior["candidate_sha256"]):
            raise ValueError("V25 trace differs from NFCorpus cases")
        signal = features(row["reference"], row["candidate"])
        risk = float(payload["model"].predict_proba([signal])[0, 1])
        role = bool(signal["system_marker"] and risk >= payload["threshold"])
        rows.append({"query_id": row["query_id"], "family": row["family"],
                     "label": row["label"], "split": split[row["query_id"]],
                     "v25": prior["v25_flag"], "role": role,
                     "v26": prior["v25_flag"] or role})
    def tally(subset: list[dict], key: str) -> dict:
        return {"flagged": sum(row[key] for row in subset), "total": len(subset)}
    report = {
        "protocol": "v26-replay-on-inspected-v25-nfcorpus-development-v1",
        "status": "development_replay_not_blind",
        "inputs_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            source, trace, ROOT / "artifacts/role_boundary_v26.joblib",
            ROOT / "results/v26_rule_frozen.json")},
        "role_delimiter": {part: {version: tally([
            row for row in rows if row["family"] == "role_delimiter"
            and (part == "all" or row["split"] == part)], version)
            for version in ("v25", "role", "v26")}
            for part in ("train", "calibration", "all")},
        "original_clean": {version: tally([
            row for row in rows if row["label"] != "poison"], version)
            for version in ("v25", "role", "v26")},
        "limits": ["NFCorpus was inspected before V26 design and supplies role-model training and calibration.",
                   "This measures source labels, not factual correctness or answer quality."],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
