"""Check saved Contriever validation ranks, counts, and pinned input hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark
from evaluation.evaluate_contriever import MODEL_REVISION, MODEL_WEIGHT_SHA256, _jsonl


def verify(root: Path = ROOT_DIR) -> dict:
    benchmark = root / "data/benchmark"
    validate_benchmark(benchmark)
    corpus_path = benchmark / "validation/corpus.jsonl"
    queries_path = benchmark / "validation/queries.jsonl"
    summary_path = root / "results/benchmark_retrieval_contriever_validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = _jsonl(summary_path.with_suffix(".cases.jsonl"))
    queries = {row["qid"]: row for row in _jsonl(queries_path)}
    corpus_ids = {row["doc_id"] for row in _jsonl(corpus_path)}
    if (summary["model_revision"] != MODEL_REVISION
            or summary["weight_sha256"] != MODEL_WEIGHT_SHA256
            or summary["corpus_sha256"] != hashlib.sha256(corpus_path.read_bytes()).hexdigest()
            or summary["queries_sha256"] != hashlib.sha256(queries_path.read_bytes()).hexdigest()
            or len(cases) != len(queries) or {row["qid"] for row in cases} != set(queries)):
        raise ValueError("Contriever validation input or case set differs")
    for row in cases:
        support = set(queries[row["qid"]]["support_doc_ids"])
        if (row["support_doc_ids"] != queries[row["qid"]]["support_doc_ids"]
                or len(row["rrf_top5"]) != 5 or len(row["dense_top5"]) != 5
                or not set(row["rrf_top5"] + row["dense_top5"]).issubset(corpus_ids)
                or row["rrf_support_at_1"] != (row["rrf_top5"][0] in support)
                or row["rrf_support_at_5"] != bool(support.intersection(row["rrf_top5"]))
                or row["dense_support_at_1"] != (row["dense_top5"][0] in support)
                or row["dense_support_at_5"] != bool(support.intersection(row["dense_top5"]))):
            raise ValueError(f"Contriever ranks disagree with labels: {row['qid']}")
    for key in ("rrf_support_at_1", "rrf_support_at_5", "dense_support_at_1", "dense_support_at_5"):
        if summary[key] != sum(row[key] for row in cases) / len(cases):
            raise ValueError(f"Contriever {key} summary differs from cases")
    return {"status": "verified", "validation_cases": len(cases),
            "rrf_support_at_5_count": sum(row["rrf_support_at_5"] for row in cases)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
