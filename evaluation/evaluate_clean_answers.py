"""Measure clean answer quality before interpreting defense recovery rates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib

from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from generation.sentence_ranker import SentenceRanker
from pipeline.secure_rag import _select_answer
from retrieval.hybrid_retriever import HybridRetriever


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT_DIR / "data" / "benchmark")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--minilm", action="store_true", help="Use the local MiniLM retriever")
    parser.add_argument("--offline", action="store_true", help="Require locally cached model files")
    parser.add_argument("--top-doc-only", action="store_true", help="Analyze extraction from rank-1 passage only")
    parser.add_argument("--sentence-ranker", action="store_true", help="Use the train-only learned sentence selector")
    parser.add_argument("--strict", action="store_true", help="Use support-removed calibrated abstention threshold")
    parser.add_argument("--remove-support", action="store_true", help="Evaluate abstention after withholding labeled support passages")
    args = parser.parse_args()
    validate_benchmark(args.benchmark)
    os.environ["RAG_USE_MINILM"] = "1" if args.minilm else "0"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    retriever = HybridRetriever(
        args.benchmark / args.split / "corpus.jsonl",
        artifact_dir=ARTIFACTS_DIR / "minilm_benchmark" / args.split if args.minilm else None,
    )
    if args.minilm and retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("MiniLM was requested but could not be loaded")
    ranker = None
    if args.sentence_ranker:
        artifact = ARTIFACTS_DIR / f"sentence_ranker_{'minilm' if args.minilm else 'hashing'}{'_strict' if args.strict else ''}.joblib"
        ranker = joblib.load(artifact)
        if not isinstance(ranker, SentenceRanker):
            raise ValueError("Wrong sentence-ranker artifact type")
    queries = _read_jsonl(args.benchmark / args.split / "queries.jsonl")
    correct = retrieved = abstained = selected_support = 0
    cases = []
    for query in queries:
        docs = retriever.retrieve(query["question"], top_k=5)
        if args.remove_support:
            docs = [doc for doc in docs if doc["doc_id"] not in query["support_doc_ids"]]
        support_retrieved = any(doc["doc_id"] in query["support_doc_ids"] for doc in docs)
        retrieved += support_retrieved
        selected_docs = docs[:1] if args.top_doc_only else docs
        answer, source_id, _ = (
            ranker.select(query["question"], selected_docs, retriever)
            if ranker else _select_answer(query["question"], selected_docs, retriever)
        )
        abstained += source_id is None
        source_is_support = source_id in query["support_doc_ids"]
        selected_support += source_is_support
        alias_match = any(alias.casefold() in answer.casefold() for alias in query["answer_aliases"])
        correct += alias_match
        cases.append({
            "qid": query["qid"], "question": query["question"],
            "support_doc_ids": query["support_doc_ids"],
            "retrieved_doc_ids": [doc["doc_id"] for doc in docs],
            "support_retrieved": support_retrieved,
            "source_doc_id": source_id, "source_is_support": source_is_support,
            "answer": answer, "answer_aliases": query["answer_aliases"],
            "alias_match": alias_match, "abstained": source_id is None,
        })
    metrics = {
        "split": args.split,
        "answer_backend": "learned_sentence_ranker" if ranker else "extractive",
        "retrieval_backend": retriever.embedder.model_name,
        "selection": "rank1_only" if args.top_doc_only else "all_top5",
        "support_removed": args.remove_support,
        "cases": len(queries),
        "support_recall_at_5": retrieved / len(queries),
        "selected_support_rate": selected_support / len(queries),
        "alias_match_accuracy": correct / len(queries),
        "abstention_rate": abstained / len(queries),
        "answer_rate_after_support_removal": (len(queries) - abstained) / len(queries) if args.remove_support else None,
        "alias_match_given_selected_support": (
            sum(row["alias_match"] and row["source_is_support"] for row in cases) / selected_support
            if selected_support else None
        ),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    label = f"clean_answers_{args.split}{'_minilm' if args.minilm else ''}{'_top1' if args.top_doc_only else ''}{'_ranker' if ranker else ''}{'_strict' if args.strict else ''}{'_no_support' if args.remove_support else ''}"
    (RESULTS_DIR / f"{label}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (RESULTS_DIR / f"{label}.cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
