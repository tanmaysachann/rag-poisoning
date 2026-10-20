"""Cache V3 risks on inspected PoisonArena development cases for V25 design.

This does not evaluate an untouched test set or modify the deployed scanner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "data/multifamily_v20/evaluation.jsonl",
    "results/fusion_v21.cases.jsonl",
    "results/snapshot_containment_v23.cases.jsonl",
    "artifacts/research_detector_v3_hashing_web.joblib",
)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source, fusion, containment = [read(ROOT / name) for name in INPUTS[:3]]
    if not len(source) == len(fusion) == len(containment) == 1066:
        raise ValueError("PoisonArena development row inventory changed")
    detector = joblib.load(ROOT / INPUTS[3])
    digest = {name: sha(ROOT / name) for name in INPUTS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for index, (row, learned, integrity) in enumerate(zip(source, fusion, containment), 1):
            key = ("query_id", "candidate_id", "family", "label")
            if (tuple(row[field] for field in key) != tuple(learned[field] for field in key)
                    or tuple(row[field] for field in key) != tuple(integrity[field] for field in key)):
                raise ValueError("Development score alignment failed")
            risk = detector.score(row["query"], row["candidate"])
            record = {
                "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                "family": row["family"], "label": row["label"],
                "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                "fusion_score": learned["fusion_score"],
                "containment_flag": integrity["flag"],
                "v3_risk": risk["risk_score"], "v3_flag": risk["decision"] == "quarantine",
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                print(f"Cached {index}/{len(source)} development scores", flush=True)
    (output.with_suffix(".inputs.json")).write_text(json.dumps({
        "protocol": "v25-inspected-development-v3-risk-cache-v1",
        "input_sha256": digest, "rows_sha256": sha(output), "rows": len(source),
        "status": "previously_inspected_development_only",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cache(ROOT / "results/v25_poisonarena_dev_scores.jsonl")
