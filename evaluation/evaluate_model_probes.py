"""Measure real MiniLM encoder activations/attention against validation attacks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from detect.model_probes import MiniLMModelProbes, PROBE_VERSION, ProbeReference
from detect.research_detector import threshold_for_fpr


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _measure(probe, queries: list[dict], docs: dict[int, dict], *, label: str):
    hidden, attention = [], []
    for index, query in enumerate(queries, 1):
        vector, concentration = probe.extract(query["question"], docs[query["support_doc_ids"][0]]["text"])
        hidden.append(vector)
        attention.append(concentration)
        if index % 50 == 0:
            print(f"{label}: {index}/{len(queries)}", flush=True)
    return np.asarray(hidden), np.asarray(attention)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--reference-count", type=int, default=100)
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument("--limit", type=int, default=75)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "model_probes_validation.json")
    args = parser.parse_args()
    if args.reference_count < 5 or args.calibration_count < 1 or args.limit < 1:
        raise ValueError("Reference, calibration, and evaluation counts must be positive")
    validate_benchmark(args.benchmark)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    torch.set_num_threads(1)
    probe = MiniLMModelProbes(local_files_only=True)
    train_queries = _read_jsonl(args.benchmark / "train" / "queries.jsonl")
    train_docs = {row["doc_id"]: row for row in _read_jsonl(args.benchmark / "train" / "corpus.jsonl")}
    if args.reference_count + args.calibration_count > len(train_queries):
        raise ValueError("Reference and calibration counts exceed training queries")
    ref_hidden, ref_attn = _measure(
        probe, train_queries[:args.reference_count], train_docs, label="reference"
    )
    reference = ProbeReference.fit(ref_hidden, ref_attn)
    calibration_queries = train_queries[args.reference_count:args.reference_count + args.calibration_count]
    cal_hidden, cal_attn = _measure(probe, calibration_queries, train_docs, label="calibration")
    cal_scores = [reference.score(vector, attention) for vector, attention in zip(cal_hidden, cal_attn)]
    keys = ("layer3_mahalanobis", "attention_deviation")
    thresholds = {
        key: threshold_for_fpr(np.asarray([row[key] for row in cal_scores]), args.target_fpr)
        for key in keys
    }
    queries = _read_jsonl(args.benchmark / "validation" / "queries.jsonl")[:args.limit]
    docs = {row["doc_id"]: row for row in _read_jsonl(args.benchmark / "validation" / "corpus.jsonl")}
    clean_hidden, clean_attn = _measure(probe, queries, docs, label="validation clean")
    clean_scores = [reference.score(vector, attention) for vector, attention in zip(clean_hidden, clean_attn)]
    results = {}
    cases = []
    for strategy in ("greedy", "stealth"):
        attacks = {row["qid"]: row for row in _read_jsonl(RESULTS_DIR / f"attack_baseline_validation_{strategy}.jsonl")}
        if set(attacks) != {query["qid"] for query in _read_jsonl(args.benchmark / "validation" / "queries.jsonl")}:
            raise ValueError(f"Incomplete {strategy} validation attacks")
        attack_scores = []
        for index, query in enumerate(queries, 1):
            vector, concentration = probe.extract(query["question"], attacks[query["qid"]]["text"])
            score = reference.score(vector, concentration)
            attack_scores.append(score)
            cases.append({"qid": query["qid"], "strategy": strategy,
                          "clean": clean_scores[index - 1], "attack": score})
            if index % 50 == 0:
                print(f"{strategy} attack: {index}/{len(queries)}", flush=True)
        family = {}
        for key in keys:
            clean = np.asarray([row[key] for row in clean_scores])
            attack = np.asarray([row[key] for row in attack_scores])
            family[key] = {
                "train_calibrated_threshold": thresholds[key],
                "validation_clean_fpr": float(np.mean(clean >= thresholds[key])),
                "validation_attack_recall": float(np.mean(attack >= thresholds[key])),
                "validation_roc_auc": float(roc_auc_score(
                    np.r_[np.zeros(len(clean)), np.ones(len(attack))], np.r_[clean, attack]
                )),
            }
        results[strategy] = family
    summary = {
        "probe_version": PROBE_VERSION,
        "model": probe.model_name,
        "model_revision": probe.model_revision,
        "encoder_internal_not_generator": True,
        "reference_train_cases": args.reference_count,
        "calibration_train_cases": args.calibration_count,
        "validation_cases_per_family": len(queries),
        "target_train_fpr": args.target_fpr,
        "families": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
