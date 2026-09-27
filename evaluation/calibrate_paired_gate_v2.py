"""Calibrate paired gate on full-context v2 training decisions only."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from evaluation.evaluate_paired_gate_v2 import paired_decision
from generation.sentence_ranker import SentenceRanker, _contains_alias


def threshold_for_clean_rejection(cosines: list[float | None], target_rate: float) -> tuple[float, int]:
    if not cosines or not 0 <= target_rate < 1:
        raise ValueError("Need nonempty scores and target_rate in [0, 1)")
    missing = sum(value is None for value in cosines)
    allowed = math.floor(target_rate * len(cosines))
    values = sorted(value for value in cosines if value is not None)
    if not values:
        raise ValueError("No query has a usable paired answer")
    if missing > allowed:
        return float(values[0]), missing
    # The gate keeps score >= threshold; at most `allowed - missing` scores
    # may lie strictly below this selected order statistic.
    threshold = values[allowed - missing]
    rejected = sum(value is None or value < threshold for value in cosines)
    return float(threshold), rejected


def main() -> None:
    benchmark = ROOT_DIR / "data/benchmark_v2"
    validate(benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(benchmark / "train/corpus.jsonl",
                                artifact_dir=ROOT_DIR / "artifacts/multisupport_v2/train")
    ranker = joblib.load(ROOT_DIR / "artifacts/sentence_ranker_minilm_strict.joblib")
    if not isinstance(ranker, SentenceRanker) or retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("MiniLM ranker required")
    cases = []
    for row in read_jsonl(benchmark / "train/queries.jsonl"):
        docs = retriever.retrieve(row["question"], top_k=5)
        result = paired_decision(row["question"], docs, retriever, ranker, -1.0)
        cases.append({"qid": row["qid"], "answer_cosine": result["answer_cosine"],
                      "source_doc_id": result["source_doc_id"],
                      "peer_doc_id": result["peer_doc_id"],
                      "answer_alias_match": _contains_alias(result["answer"], row["answer_aliases"])})
    threshold, rejected = threshold_for_clean_rejection(
        [row["answer_cosine"] for row in cases], 0.05
    )
    summary = {
        "benchmark": "two-support-msmarco-v2", "split": "train", "cases": len(cases),
        "target_clean_rejection": 0.05, "calibrated_threshold": threshold,
        "train_clean_rejected": rejected,
        "target_feasible": rejected <= int(0.05 * len(cases)),
        "train_clean_alias_match": sum(row["answer_alias_match"] for row in cases),
        "missing_peer_document": sum(row["peer_doc_id"] is None for row in cases),
        "missing_usable_pair_score": sum(row["answer_cosine"] is None for row in cases),
        "peer_group_contract": "non-null source_query_id equality for retrieved IDs via corpus metadata",
        "note": "Threshold calibrated on full retrieval and selected-answer path, not doc-only support pairs.",
    }
    output = ROOT_DIR / "results/paired_gate_v2_train_calibration.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
