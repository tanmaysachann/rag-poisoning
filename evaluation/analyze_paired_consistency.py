"""Measure agreement between two labeled passages without reading v2 test data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from config import ROOT_DIR
from data.validate_multisupport_benchmark import read_jsonl, validate
from generation.sentence_ranker import SentenceRanker, _contains_alias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data/benchmark_v2")
    args = parser.parse_args()
    validate(args.benchmark, ROOT_DIR / "data/benchmark")
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(args.benchmark / args.split / "corpus.jsonl",
                                artifact_dir=ROOT_DIR / "artifacts/multisupport_v2" / args.split)
    ranker = joblib.load(ROOT_DIR / "artifacts/sentence_ranker_minilm_strict.joblib")
    if not isinstance(ranker, SentenceRanker) or retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("MiniLM and the frozen sentence ranker are required")
    docs = {row["doc_id"]: row for row in retriever.documents}
    queries = read_jsonl(args.benchmark / args.split / "queries.jsonl")
    cases = []
    for query in queries:
        pair = [docs[doc_id] for doc_id in query["support_doc_ids"]]
        answers = [ranker.select(query["question"], [doc], retriever)[0] for doc in pair]
        both_answered = all(not answer.startswith("Insufficient relevant evidence") for answer in answers)
        cosine = (float(retriever.encode(answers[0])[0] @ retriever.encode(answers[1])[0])
                  if both_answered else None)
        cases.append({"qid": query["qid"], "support_doc_ids": query["support_doc_ids"],
                      "answers": answers, "both_answered": both_answered,
                      "answer_cosine": cosine,
                      "both_match_alias": all(_contains_alias(answer, query["answer_aliases"]) for answer in answers)})
    valid = [row["answer_cosine"] for row in cases if row["answer_cosine"] is not None]
    summary = {
        "benchmark": "two-support-msmarco-v2", "split": args.split, "cases": len(cases),
        "both_passages_answered": sum(row["both_answered"] for row in cases),
        "both_passages_exact_alias_match": sum(row["both_match_alias"] for row in cases),
        "cosine_median_when_both_answered": float(np.median(valid)) if valid else None,
        "cosine_p05_when_both_answered": float(np.quantile(valid, 0.05)) if valid else None,
        "cosine_p25_when_both_answered": float(np.quantile(valid, 0.25)) if valid else None,
        "note": "Two answer-bearing passages share an MS MARCO query; sentence cosine is not factual entailment.",
    }
    output = ROOT_DIR / f"results/paired_consistency_v2_{args.split}.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
