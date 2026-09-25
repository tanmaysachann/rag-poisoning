"""Check abstention on NQ-Open questions against the small MS MARCO validation corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from config import ARTIFACTS_DIR, RESULTS_DIR, ROOT_DIR
from data.validate_benchmark import validate_benchmark
from generation.sentence_ranker import SentenceRanker
from retrieval.hybrid_retriever import HybridRetriever


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    benchmark = ROOT_DIR / "data/benchmark"
    validate_benchmark(benchmark)
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    corpus_path = benchmark / "validation/corpus.jsonl"
    corpus = _read_jsonl(corpus_path)
    retriever = HybridRetriever(corpus_path, artifact_dir=ARTIFACTS_DIR / "minilm_benchmark/validation")
    if retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("Cached MiniLM is required")
    ranker = joblib.load(ARTIFACTS_DIR / "sentence_ranker_minilm_strict.joblib")
    if not isinstance(ranker, SentenceRanker):
        raise ValueError("Wrong ranker artifact type")
    questions = _read_jsonl(benchmark / "nq_challenge.jsonl")
    if len(questions) != 50 or len({row["qid"] for row in questions}) != 50:
        raise ValueError("Expected exactly 50 unique NQ challenge questions")
    cases = []
    for query in questions:
        aliases = [alias.casefold() for alias in query["answer_aliases"]]
        alias_seen = any(alias in doc["text"].casefold() for doc in corpus for alias in aliases)
        candidates = retriever.retrieve(query["question"], top_k=5)
        answer, source_id, _ = ranker.select(query["question"], candidates, retriever)
        cases.append({
            "qid": query["qid"], "question": query["question"],
            "answer_aliases": query["answer_aliases"],
            "answer": answer, "source_doc_id": source_id,
            "retrieved_doc_ids": [doc["doc_id"] for doc in candidates],
            "alias_seen_anywhere_in_validation_corpus": alias_seen,
            "alias_match": any(alias in answer.casefold() for alias in aliases),
            "abstained": source_id is None,
        })
    absent = [row for row in cases if not row["alias_seen_anywhere_in_validation_corpus"]]
    summary = {
        "corpus_split": "validation", "questions": len(cases),
        "source_dataset": "NQ-Open challenge",
        "alias_absent_from_corpus": len(absent),
        "abstention_all": sum(row["abstained"] for row in cases) / len(cases),
        "abstention_when_alias_absent": (
            sum(row["abstained"] for row in absent) / len(absent) if absent else None
        ),
        "alias_match_all": sum(row["alias_match"] for row in cases) / len(cases),
        "note": "Alias absence is a lexical stress test, not proof that no passage entails the answer; not an accuracy claim",
    }
    output = RESULTS_DIR / "nq_challenge_validation_minilm_ranker.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
