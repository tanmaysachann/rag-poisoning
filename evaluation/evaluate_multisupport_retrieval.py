"""Measure two-support retrieval and clean answer selection on v2 development data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from generation.sentence_ranker import SentenceRanker, _contains_alias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data/benchmark_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate(args.benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(
        args.benchmark / args.split / "corpus.jsonl",
        artifact_dir=ROOT_DIR / "artifacts/multisupport_v2" / args.split,
    )
    if retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("Cached MiniLM was not loaded")
    ranker = joblib.load(ROOT_DIR / "artifacts/sentence_ranker_minilm_strict.joblib")
    if not isinstance(ranker, SentenceRanker):
        raise ValueError("Wrong sentence-ranker artifact type")
    queries = read_jsonl(args.benchmark / args.split / "queries.jsonl")
    cases = []
    for row in queries:
        docs = retriever.retrieve(row["question"], top_k=5)
        ids = [doc["doc_id"] for doc in docs]
        support_hits = [doc_id for doc_id in row["support_doc_ids"] if doc_id in ids]
        answer, source, _ = ranker.select(row["question"], docs, retriever)
        cases.append({
            "qid": row["qid"], "question": row["question"],
            "support_doc_ids": row["support_doc_ids"], "retrieved_doc_ids": ids,
            "support_hits": support_hits, "support_count_at_5": len(support_hits),
            "answer": answer, "source_doc_id": source,
            "source_is_support": source in row["support_doc_ids"],
            "alias_match": _contains_alias(answer, row["answer_aliases"]),
            "abstained": source is None,
        })
    n = len(cases)
    summary = {
        "benchmark": "two-support-msmarco-v2", "split": args.split, "cases": n,
        "retrieval_backend": retriever.embedder.model_name,
        "one_or_more_support_at_5": sum(row["support_count_at_5"] >= 1 for row in cases) / n,
        "both_supports_at_5": sum(row["support_count_at_5"] == 2 for row in cases) / n,
        "answer_alias_match": sum(row["alias_match"] for row in cases) / n,
        "answer_source_is_support": sum(row["source_is_support"] for row in cases) / n,
        "abstention_rate": sum(row["abstained"] for row in cases) / n,
        "note": "Development split only. Two passages share an MS MARCO query and are not proven independent sources.",
    }
    output = args.output or ROOT_DIR / f"results/multisupport_v2_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
