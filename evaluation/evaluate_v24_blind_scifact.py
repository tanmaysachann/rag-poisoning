"""Evaluate the unchanged V24 scanner on the frozen SciFact transfer set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from data.build_v24_blind_scifact import sha


ROOT = Path(__file__).resolve().parents[1]


def _rate(rows: list[dict], label: str, key: str) -> dict:
    chosen = [row for row in rows if (row["label"] == "poison") == (label == "poison")]
    flagged = sum(row[key] for row in chosen)
    return {"flagged": flagged, "total": len(chosen),
            "rate": flagged / len(chosen) if chosen else None}


def _cluster_interval(rows: list[dict], label: str, key: str,
                      *, seed: int = 20261020, draws: int = 1000) -> list[float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    names = sorted(groups)
    randomizer = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [item for _ in names for item in groups[randomizer.choice(names)]]
        estimates.append(_rate(sample, label, key)["rate"])
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]]


def evaluate(dataset: Path, output: Path) -> dict:
    dataset, output = dataset.resolve(), output.resolve()
    manifest_path, cases_path = dataset / "manifest.json", dataset / "cases.jsonl"
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen["protocol"] != "v24-blind-scifact-source-label-transfer-v1":
        raise ValueError("Unexpected frozen protocol")
    expected = {
        "cases_sha256": cases_path,
        "builder_sha256": ROOT / "data/build_v24_blind_scifact.py",
        "model_sha256": ROOT / "artifacts/multifamily_v20/model.safetensors",
        "calibration_sha256": ROOT / "artifacts/product_v24_calibration.joblib",
    }
    if any(sha(path) != frozen[field] for field, path in expected.items()):
        raise ValueError("Frozen data, generator, or model changed")
    rows = [json.loads(line) for line in cases_path.open(encoding="utf-8")]
    if len(rows) != frozen["rows"] or len({row["query_id"] for row in rows}) != frozen["questions"]:
        raise ValueError("Frozen case inventory changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Evaluation report already exists: {output}")
    trace_path = output.with_suffix(".cases.jsonl")
    prior = []
    if trace_path.exists():
        prior = [json.loads(line) for line in trace_path.open(encoding="utf-8") if line.strip()]
        if len(prior) > len(rows):
            raise ValueError("Existing trace has too many rows")
        for source, result in zip(rows, prior):
            if (result["query_id"], result["candidate_id"], result["candidate_sha256"]) != (
                    source["query_id"], source["candidate_id"],
                    hashlib.sha256(source["candidate"].encode()).hexdigest()):
                raise ValueError("Existing trace does not match the frozen dataset")
    from detect.product_v22 import PoisonDocumentScanner
    start = time.perf_counter()
    scanner = PoisonDocumentScanner(root=ROOT, integrity_mode="containment")
    try:
        with trace_path.open("a", encoding="utf-8", newline="\n") as sink:
            for index, row in enumerate(rows[len(prior):], len(prior) + 1):
                result = scanner.assess(row["query"], row["reference"], row["candidate"])
                base = scanner.v3.score(row["query"], row["candidate"])
                record = {
                    "query_id": row["query_id"], "candidate_id": row["candidate_id"],
                    "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                    "label": row["label"], "family": row["family"],
                    "v24_flag": result["decision"] == "review",
                    "v3_flag": base["decision"] == "quarantine",
                    "learned_flag": result["learned"]["flag"],
                    "containment_flag": result["integrity"]["flag"],
                    "fusion_score": result["learned"]["fusion_score"],
                    "v3_risk": base["risk_score"],
                    "snapshot_match_method": result["integrity"].get("match_method"),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                prior.append(record)
                if index % 100 == 0:
                    print(f"Scored {index}/{len(rows)} frozen cases", flush=True)
    finally:
        scanner.close()
    metrics = {
        name: {"attack_recall": _rate(prior, "poison", key),
               "clean_false_alarm": _rate(prior, "clean", key),
               "question_cluster_95pct_ci": {
                   "attack_recall": _cluster_interval(prior, "poison", key),
                   "clean_false_alarm": _cluster_interval(prior, "clean", key),
               }}
        for name, key in (("v24", "v24_flag"), ("v3_detector", "v3_flag"),
                          ("learned_only", "learned_flag"),
                          ("containment_only", "containment_flag"))
    }
    families = {
        name: {version: _rate([row for row in prior if row["family"] == name],
                             "poison", key)
               for version, key in (("v24", "v24_flag"), ("v3_detector", "v3_flag"))}
        for name in frozen["families"]
    }
    report = {
        "protocol": frozen["protocol"], "status": "frozen_cross_domain_synthetic_transfer",
        "dataset_manifest_sha256": sha(manifest_path),
        "cases_sha256": sha(trace_path),
        "model_sha256": frozen["model_sha256"],
        "calibration_sha256": frozen["calibration_sha256"],
        "questions": frozen["questions"], "cases": len(prior),
        "metrics": metrics, "attack_families": families,
        "elapsed_seconds_this_run": round(time.perf_counter() - start, 1),
        "limits": [
            "Synthetic fixed-template source labels; no independent factual review.",
            "SciFact relevance qrels are not a truth oracle for scientific claims.",
            "New inserted documents have no authenticated known-ID hash.",
            "This measures passage screening, not downstream RAG answer correctness.",
            "No tuning on these cases is allowed after the freeze.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/v24_blind_scifact")
    parser.add_argument("--output", type=Path, default=ROOT / "results/v24_blind_scifact.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.output), indent=2))
