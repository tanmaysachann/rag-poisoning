"""Paired V25/V26 scoring of a separately pinned post-hoc FiQA role stress."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from detect.product_v26 import RoleAdaptiveScanner


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v26_fiqa_role_stress"
FROZEN = ROOT / "results/v26_fiqa_role_stress_frozen.json"
TRACE = ROOT / "results/v26_fiqa_role_stress.cases.jsonl"
OUTPUT = ROOT / "results/v26_fiqa_role_stress.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def tally(rows: list[dict], field: str) -> dict:
    flagged = sum(row[field] for row in rows)
    return {"flagged": flagged, "total": len(rows), "rate": flagged / len(rows)}


def evaluate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["status"] != "followup_schedule_frozen_before_scoring":
        raise ValueError("Targeted stress schedule was not frozen")
    for name, digest in frozen["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Targeted stress input changed: {name}")
    rows = read(DATA / "cases.jsonl")
    if len(rows) != 360 or len({row["query_id"] for row in rows}) != 120:
        raise ValueError("Targeted stress inventory changed")
    prior = read(TRACE) if TRACE.exists() else []
    if len(prior) > len(rows):
        raise ValueError("Trace exceeds schedule")
    for row, score in zip(rows, prior):
        if (row["query_id"], row["candidate_id"], hashlib.sha256(row["candidate"].encode()).hexdigest()) != (
                score["query_id"], score["candidate_id"], score["candidate_sha256"]):
            raise ValueError("Trace differs from pinned schedule")
    scanner = RoleAdaptiveScanner(root=ROOT)
    try:
        with TRACE.open("a", encoding="utf-8", newline="\n") as sink:
            for index, row in enumerate(rows[len(prior):], len(prior) + 1):
                result = scanner.assess(row["query"], row["reference"], row["candidate"])
                score = {
                    "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                    "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                    "family": row["family"], "label": row["label"],
                    "v25_flag": bool(result["v3"]["flag"] or result["reference_adaptive"]["flag"]
                                     or result["integrity"].get("flag", False) or result["identity"]["flag"]),
                    "role_flag": result["role_boundary"]["flag"],
                    "v26_flag": result["decision"] == "review",
                }
                sink.write(json.dumps(score, ensure_ascii=False) + "\n")
                sink.flush()
                prior.append(score)
                if index % 30 == 0:
                    print(f"Scored {index}/{len(rows)} pinned follow-up stress cases", flush=True)
    finally:
        scanner.close()
    for name, digest in frozen["input_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Input changed during stress scoring: {name}")
    report = {
        "protocol": "v26-fiqa-post-hoc-targeted-role-stress-v1",
        "status": "followup_targeted_not_blind",
        "schedule_sha256": sha(FROZEN), "trace_sha256": sha(TRACE),
        "questions": 120, "cases": 360,
        "results": {family: {version: tally([r for r in prior if r["family"] == family], field)
                             for version, field in (("v25", "v25_flag"),
                                                    ("v26", "v26_flag"),
                                                    ("role_only", "role_flag"))}
                    for family in ("reference", "quoted_markup", "role_delimiter")},
        "limits": [
            "Designed after inspecting the first FiQA aggregate result; this is targeted stress, not blind model-selection evidence.",
            "The role template was already used in NFCorpus development and is not a novel attack family.",
            "Original FiQA references are qrel-relevant, not verified factual truth.",
            "Synthetic source labels do not measure downstream answer accuracy.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
