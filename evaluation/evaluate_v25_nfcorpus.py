"""One-pass frozen V24/V25 paired source-label evaluation on NFCorpus."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from detect.product_v25 import ReferenceAdaptiveScanner


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/v25_rule_frozen.json"
DATASET = ROOT / "data/v25_nfcorpus_holdout"
OUTPUT = ROOT / "results/v25_nfcorpus_holdout.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def count(rows: list[dict], key: str, poison: bool) -> dict:
    subset = [row for row in rows if (row["label"] == "poison") == poison]
    flagged = sum(bool(row[key]) for row in subset)
    return {"flagged": flagged, "total": len(subset),
            "rate": flagged / len(subset)}


def cluster_ci(rows: list[dict], key: str, poison: bool,
               *, seed: int = 20261020, draws: int = 2000) -> list[float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    qids = sorted(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [item for _ in qids for item in groups[rng.choice(qids)]]
        values.append(count(sample, key, poison)["rate"])
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]


def evaluate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_nfcorpus_holdout_scoring":
        raise ValueError("V25 rule was not frozen before holdout scoring")
    if any(sha(ROOT / name) != expected for name, expected in frozen["input_sha256"].items()):
        raise ValueError("A frozen rule, model, development trace, or holdout input changed")
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != "v25-nfcorpus-frozen-source-label-holdout-v1":
        raise ValueError("Unexpected holdout protocol")
    rows = read(DATASET / "cases.jsonl")
    if len(rows) != manifest["rows"] or len({row["query_id"] for row in rows}) != manifest["questions"]:
        raise ValueError("Holdout inventory changed")
    trace = OUTPUT.with_suffix(".cases.jsonl")
    prior = read(trace) if trace.exists() else []
    if len(prior) > len(rows):
        raise ValueError("Existing trace exceeds holdout inventory")
    for source, record in zip(rows, prior):
        if (record["query_id"], record["candidate_id"], record["candidate_sha256"]) != (
                source["query_id"], source["candidate_id"],
                hashlib.sha256(source["candidate"].encode()).hexdigest()):
            raise ValueError("Existing trace is not the frozen holdout schedule")
    start = time.perf_counter()
    scanner = ReferenceAdaptiveScanner(root=ROOT)
    try:
        with trace.open("a", encoding="utf-8", newline="\n") as sink:
            for index, row in enumerate(rows[len(prior):], len(prior) + 1):
                result = scanner.assess(row["query"], row["reference"], row["candidate"])
                baseline_flag = (result["learned"]["flag"] or result["integrity"].get("flag", False)
                                 or result["identity"]["flag"])
                record = {
                    "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                    "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                    "label": row["label"], "family": row["family"],
                    "v24_flag": bool(baseline_flag),
                    "v25_flag": result["decision"] == "review",
                    "v3_flag": result["v3"]["risk"] >= scanner.v24.v3.threshold,
                    "v3_risk": result["v3"]["risk"],
                    "fusion_score": result["learned"]["fusion_score"],
                    "reference_fusion_score": result["reference_adaptive"]["reference_fusion_score"],
                    "containment_flag": result["integrity"].get("flag", False),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                prior.append(record)
                if index % 70 == 0:
                    print(f"Scored {index}/{len(rows)} untouched NFCorpus cases", flush=True)
    finally:
        scanner.close()
    if any(sha(ROOT / name) != expected for name, expected in frozen["input_sha256"].items()):
        raise ValueError("Frozen input changed during holdout scoring")
    metrics = {
        name: {
            "attack_recall": count(prior, key, True),
            "clean_false_alarm": count(prior, key, False),
            "question_cluster_95pct_ci": {
                "attack_recall": cluster_ci(prior, key, True),
                "clean_false_alarm": cluster_ci(prior, key, False),
            },
        }
        for name, key in (("v24", "v24_flag"), ("v25", "v25_flag"),
                          ("v3_detector", "v3_flag"))
    }
    families = {
        family: {name: count([row for row in prior if row["family"] == family], key, True)
                 for name, key in (("v24", "v24_flag"), ("v25", "v25_flag"))}
        for family in manifest["families"]
    }
    report = {
        "protocol": "v25-paired-nfcorpus-holdout-v1",
        "status": "first_scoring_after_rule_freeze",
        "rule_sha256": sha(FROZEN),
        "holdout_manifest_sha256": sha(DATASET / "manifest.json"),
        "trace_sha256": sha(trace),
        "questions": manifest["questions"], "cases": len(prior),
        "metrics": metrics, "families": families,
        "elapsed_seconds_this_run": round(time.perf_counter() - start, 1),
        "limits": [
            "Five fixed-template synthetic new-document attacks; not arbitrary adversarial text.",
            "BEIR qrels choose relevant references and do not verify factual truth.",
            "The reference is a benchmark oracle here; deployment needs independent authentication.",
            "This measures document screening, not end-to-end answer correctness.",
            "Do not tune V25 on these cases and then call this an untouched holdout.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
