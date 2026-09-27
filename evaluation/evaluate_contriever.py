"""Offline CPU Contriever-msmarco retrieval comparison on v1 validation only."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from config import ROOT_DIR, RRF_K
from data.validate_benchmark import validate_benchmark
from retrieval.hybrid_retriever import tokenize


MODEL_ID = "facebook/contriever-msmarco"
MODEL_REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"
MODEL_WEIGHT_SHA256 = "08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode(texts: list[str], tokenizer, model, *, batch_size: int = 8) -> np.ndarray:
    """Masked mean pooling from the official model card, then L2 normalize."""
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[start:start + batch_size], padding=True,
                               truncation=True, max_length=256, return_tensors="pt")
            output = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            rows.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def main() -> None:
    torch.set_num_threads(4)
    benchmark = ROOT_DIR / "data/benchmark"
    validate_benchmark(benchmark)
    corpus_path = benchmark / "validation/corpus.jsonl"
    queries_path = benchmark / "validation/queries.jsonl"
    documents, queries = _jsonl(corpus_path), _jsonl(queries_path)
    model_dir = ROOT_DIR / "artifacts/models/contriever_msmarco"
    weight = model_dir / "pytorch_model.bin"
    if not weight.is_file() or hashlib.sha256(weight.read_bytes()).hexdigest() != MODEL_WEIGHT_SHA256:
        raise ValueError("Pinned Contriever weight is missing or its SHA-256 differs")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to("cpu")
    model_load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    doc_vectors = encode([row["text"] for row in documents], tokenizer, model)
    corpus_encode_ms = (time.perf_counter() - started) * 1000
    bm25 = BM25Okapi([tokenize(row["text"]) for row in documents])
    cases, query_times = [], []
    for query in queries:
        started = time.perf_counter()
        query_vector = encode([query["question"]], tokenizer, model)[0]
        bm25_scores = np.asarray(bm25.get_scores(tokenize(query["question"])), dtype=np.float32)
        dense_scores = doc_vectors @ query_vector
        sparse_order = np.argsort(-bm25_scores, kind="stable")
        dense_order = np.argsort(-dense_scores, kind="stable")
        sparse_ranks = np.empty(len(documents), dtype=np.int32)
        dense_ranks = np.empty(len(documents), dtype=np.int32)
        sparse_ranks[sparse_order] = np.arange(1, len(documents) + 1)
        dense_ranks[dense_order] = np.arange(1, len(documents) + 1)
        fused = 1 / (RRF_K + sparse_ranks) + 1 / (RRF_K + dense_ranks)
        fused_order = np.argsort(-fused, kind="stable")
        query_times.append((time.perf_counter() - started) * 1000)
        top5 = [documents[int(index)]["doc_id"] for index in fused_order[:5]]
        dense_top5 = [documents[int(index)]["doc_id"] for index in dense_order[:5]]
        support = set(query["support_doc_ids"])
        cases.append({
            "qid": query["qid"], "support_doc_ids": query["support_doc_ids"],
            "rrf_top5": top5, "dense_top5": dense_top5,
            "rrf_support_at_1": top5[0] in support,
            "rrf_support_at_5": bool(support.intersection(top5)),
            "dense_support_at_1": dense_top5[0] in support,
            "dense_support_at_5": bool(support.intersection(dense_top5)),
        })
    summary = {
        "split": "validation", "cases": len(cases), "corpus_documents": len(documents),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weight_sha256": MODEL_WEIGHT_SHA256,
        "pooling": "attention-mask mean of last hidden state, L2 normalized",
        "max_length": 256, "device": "cpu", "torch_threads": 4,
        "dimensions": int(doc_vectors.shape[1]),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "model_load_ms": round(model_load_ms, 2),
        "corpus_encode_ms": round(corpus_encode_ms, 2),
        "query_and_search_median_ms": round(float(np.median(query_times)), 2),
        "query_and_search_p95_ms": round(float(np.quantile(query_times, .95)), 2),
        **{name: sum(row[name] for row in cases) / len(cases) for name in (
            "rrf_support_at_1", "rrf_support_at_5", "dense_support_at_1", "dense_support_at_5"
        )},
        "note": "Development split only; this is retrieval support recall, not answer correctness or attack robustness.",
    }
    output = ROOT_DIR / "results/benchmark_retrieval_contriever_validation.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in cases:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
