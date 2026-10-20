"""First paired V25/V26 scoring of pinned FiQA source-label cases."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from detect.product_v26 import RoleAdaptiveScanner


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/v26_rule_frozen.json"
DATASET = ROOT / "data/v26_fiqa_holdout"
OUTPUT = ROOT / "results/v26_fiqa_holdout.json"
TRACE = ROOT / "results/v26_fiqa_holdout.cases.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def rate(rows: list[dict], key: str) -> dict:
    flagged = sum(bool(row[key]) for row in rows)
    return {"flagged": flagged, "total": len(rows),
            "rate": flagged / len(rows) if rows else None}


def ci(rows: list[dict], key: str, label: str, *, draws: int = 2000) -> list[float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    qids = sorted(groups)
    rng = random.Random(20261020)
    values = []
    for _ in range(draws):
        sample = [item for _ in qids for item in groups[rng.choice(qids)] if item["label"] == label]
        values.append(rate(sample, key)["rate"])
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]


def evaluate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_fiqa_holdout_scoring":
        raise ValueError("V26 rule was not frozen before FiQA scoring")
    for name, expected in frozen["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen V26 input changed: {name}")
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    rows = read(DATASET / "cases.jsonl")
    if len(rows) != manifest["rows"] or len({r["query_id"] for r in rows}) != manifest["questions"]:
        raise ValueError("FiQA inventory changed")
    prior = read(TRACE) if TRACE.exists() else []
    for source, record in zip(rows, prior):
        if (record["query_id"], record["candidate_id"], record["candidate_sha256"]) != (
                source["query_id"], source["candidate_id"],
                hashlib.sha256(source["candidate"].encode()).hexdigest()):
            raise ValueError("Scoring trace differs from the frozen FiQA schedule")
    started = time.perf_counter()
    scanner = RoleAdaptiveScanner(root=ROOT)
    try:
        with TRACE.open("a", encoding="utf-8", newline="\n") as sink:
            for index, row in enumerate(rows[len(prior):], len(prior) + 1):
                result = scanner.assess(row["query"], row["reference"], row["candidate"])
                record = {
                    "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                    "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                    "label": row["label"], "family": row["family"],
                    "v25_flag": result["version"] == "v26-development" and (
                        result["v3"]["flag"] or result["reference_adaptive"]["flag"]
                        or result["integrity"].get("flag", False) or result["identity"]["flag"]),
                    "role_flag": result["role_boundary"]["flag"],
                    "role_risk": result["role_boundary"]["risk"],
                    "v26_flag": result["decision"] == "review",
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                prior.append(record)
                if index % 40 == 0:
                    print(f"Scored {index}/{len(rows)} frozen FiQA cases", flush=True)
    finally:
        scanner.close()
    for name, expected in frozen["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed during scoring: {name}")
    metrics = {}
    for version, key in (("v25", "v25_flag"), ("v26", "v26_flag"), ("role_only", "role_flag")):
        metrics[version] = {
            "attack_recall": rate([r for r in prior if r["label"] == "poison"], key),
            "clean_false_alarm": rate([r for r in prior if r["label"] != "poison"], key),
            "original_clean_false_alarm": rate([r for r in prior if r["label"] in {"reference", "clean_other"}], key),
            "benign_markup_false_alarm": rate([r for r in prior if r["label"] == "benign_markup"], key),
            "question_cluster_95pct_ci": {
                "attack_recall": ci(prior, key, "poison"),
                "benign_markup_false_alarm": ci(prior, key, "benign_markup"),
            },
        }
    families = {family: {version: rate([r for r in prior if r["family"] == family], key)
                         for version, key in (("v25", "v25_flag"), ("v26", "v26_flag"),
                                              ("role_only", "role_flag"))}
                for family in ["reference", "clean_other", *manifest["families"]]}
    report = {
        "protocol": "v26-paired-fiqa-holdout-v1", "status": "first_scoring_after_rule_freeze",
        "rule_sha256": sha(FROZEN), "holdout_manifest_sha256": sha(DATASET / "manifest.json"),
        "trace_sha256": sha(TRACE), "questions": manifest["questions"], "cases": len(prior),
        "metrics": metrics, "families": families,
        "elapsed_seconds_this_run": round(time.perf_counter() - started, 1),
        "limits": ["Fixed synthetic attacks on original FiQA passages, not arbitrary attacks.",
                   "Source labels measure authored attack intent, not factual correctness.",
                   "FiQA qrels measure relevance, not answer truth.",
                   "This is document screening; answer-path replay is reported separately."],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
