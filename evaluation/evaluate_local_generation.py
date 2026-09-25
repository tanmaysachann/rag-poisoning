"""Measure the optional local cited generator on clean validation questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--model-path", type=Path, default=ROOT / "artifacts/models/qwen2_5_0_5b")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Limit and max_new_tokens must be positive")
    weight_path = args.model_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Local Qwen weights are missing: {weight_path}")
    weight_sha256 = _sha256(weight_path)
    os.environ["RAG_LLM_MODEL"] = str(args.model_path.resolve())
    os.environ["RAG_USE_MINILM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from data.validate_benchmark import validate_benchmark
    from generation.local_llm import generate_raw_answer, validate_cited_output
    from retrieval.hybrid_retriever import HybridRetriever

    validate_benchmark(args.benchmark)
    queries = _read_jsonl(args.benchmark / "validation/queries.jsonl")
    random.Random(args.seed).shuffle(queries)
    queries = queries[:args.limit]
    if len(queries) != args.limit:
        raise ValueError("Requested more cases than the validation split contains")
    retriever = HybridRetriever(
        args.benchmark / "validation/corpus.jsonl",
        artifact_dir=ROOT / "artifacts/minilm_benchmark/validation",
    )
    if retriever.embedder.backend != "sentence-transformers":
        raise RuntimeError("Cached MiniLM is required for this generation study")
    cases = []
    for query in queries:
        started = time.perf_counter()
        documents = retriever.retrieve(query["question"], top_k=5)
        retrieval_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        raw_answer = generate_raw_answer(query["question"], documents, max_new_tokens=args.max_new_tokens)
        answer, source_id = validate_cited_output(raw_answer, documents)
        generation_ms = (time.perf_counter() - started) * 1000
        cases.append({
            "qid": query["qid"], "question": query["question"],
            "answer_aliases": query["answer_aliases"],
            "raw_answer": raw_answer, "answer": answer, "source_doc_id": source_id,
            "support_doc_ids": query["support_doc_ids"],
            "retrieved_doc_ids": [doc["doc_id"] for doc in documents],
            "support_retrieved": any(doc["doc_id"] in query["support_doc_ids"] for doc in documents),
            "citation_valid": source_id is not None,
            "source_is_support": source_id in query["support_doc_ids"],
            "alias_match": any(alias.casefold() in answer.casefold() for alias in query["answer_aliases"]),
            "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
        })
    summary = {
        "split": "validation", "cases": len(cases), "seed": args.seed,
        "generator": "Qwen/Qwen2.5-0.5B-Instruct", "model_weight_sha256": weight_sha256,
        "max_new_tokens": args.max_new_tokens,
        "retrieval_backend": retriever.embedder.model_name,
        "support_recall_at_5": sum(row["support_retrieved"] for row in cases) / len(cases),
        "valid_citation_rate": sum(row["citation_valid"] for row in cases) / len(cases),
        "cited_support_rate": sum(row["source_is_support"] for row in cases) / len(cases),
        "alias_match_rate": sum(row["alias_match"] for row in cases) / len(cases),
        "abstention_rate": sum(not row["citation_valid"] for row in cases) / len(cases),
        "median_generation_ms": statistics.median(row["generation_ms"] for row in cases),
        "median_retrieval_ms": statistics.median(row["retrieval_ms"] for row in cases),
        "note": "Clean validation subset; citation ID is checked against retrieved documents, not factual entailment",
    }
    output = args.output or ROOT / f"results/local_generation_validation_{len(cases)}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
