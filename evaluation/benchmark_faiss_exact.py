"""Compare the current exact dense search with FAISS FlatIP on development data.

The larger corpus is a deterministic, perturbed replication of v1 train and
validation passages. It measures index/search cost, not retrieval quality.
No v1 or v2 test examples are opened.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from threadpoolctl import threadpool_limits

from config import ROOT_DIR
from data.validate_benchmark import validate_benchmark


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _median_ms(values: list[float]) -> float:
    return round(statistics.median(values) * 1000, 3)


def _p95_ms(values: list[float]) -> float:
    return round(float(np.quantile(values, 0.95)) * 1000, 3)


def benchmark(base: np.ndarray, questions: np.ndarray, *, copies: int, seed: int = 42) -> dict:
    if copies < 1 or base.ndim != 2 or questions.ndim != 2 or base.shape[1] != questions.shape[1]:
        raise ValueError("Need positive copies and matching two-dimensional vectors")
    if copies == 1:
        matrix = np.ascontiguousarray(base, dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        matrix = np.tile(base, (copies, 1))
        matrix += rng.normal(0.0, 0.01, matrix.shape).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        matrix = np.ascontiguousarray(matrix)
    queries = np.ascontiguousarray(questions, dtype=np.float32)
    k = min(5, len(matrix))
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        index = faiss.IndexFlatIP(matrix.shape[1])
        started = time.perf_counter()
        index.add(matrix)
        build_seconds = time.perf_counter() - started

        # The first search warms library code and is omitted for both paths.
        _ = matrix @ queries[0]
        _ = index.search(queries[:1], k)
        numpy_times, faiss_times = [], []
        id_order_disagreements, score_disagreements = [], []
        for position, query in enumerate(queries):
            started = time.perf_counter()
            scores = matrix @ query
            exact = np.argsort(-scores, kind="stable")[:k]
            numpy_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            found_scores, found = index.search(query.reshape(1, -1), k)
            faiss_times.append(time.perf_counter() - started)
            if list(map(int, exact)) != list(map(int, found[0])):
                id_order_disagreements.append(position)
            if not np.allclose(scores[exact], found_scores[0], rtol=0, atol=1e-5):
                score_disagreements.append(position)
    return {
        "vectors": len(matrix), "dimensions": matrix.shape[1],
        "float32_matrix_mb": round(matrix.nbytes / 1_000_000, 3),
        "faiss_build_ms": round(build_seconds * 1000, 3),
        "numpy_search_median_ms": _median_ms(numpy_times),
        "numpy_search_p95_ms": _p95_ms(numpy_times),
        "faiss_search_median_ms": _median_ms(faiss_times),
        "faiss_search_p95_ms": _p95_ms(faiss_times),
        "top5_id_order_disagreements": id_order_disagreements,
        "top5_score_disagreements": score_disagreements,
        "queries": len(queries), "copies": copies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results/faiss_exact_scale_validation.json")
    args = parser.parse_args()
    if any(value < 1 for value in args.copies):
        raise ValueError("--copies values must be positive")
    benchmark_root = ROOT_DIR / "data/benchmark"
    validate_benchmark(benchmark_root)
    documents = [row for split in ("train", "validation")
                 for row in _jsonl(benchmark_root / split / "corpus.jsonl")]
    questions = _jsonl(benchmark_root / "validation/queries.jsonl")[:30]
    vectorizer = HashingVectorizer(n_features=384, alternate_sign=False,
                                  norm="l2", ngram_range=(1, 2))
    base = vectorizer.transform([row["text"] for row in documents]).toarray().astype(np.float32)
    query_vectors = vectorizer.transform([row["question"] for row in questions]).toarray().astype(np.float32)
    runs = [benchmark(base, query_vectors, copies=copies) for copies in args.copies]
    if any(row["top5_score_disagreements"] for row in runs):
        raise RuntimeError("FAISS FlatIP and the current NumPy dense search disagree on top-five scores")
    result = {
        "input": "v1 train+validation passages and validation questions only",
        "base_documents": len(documents), "queries": len(questions),
        "embedding": "sklearn HashingVectorizer 384 dimensions, normalized",
        "large_corpus": "seed-42 perturbed replicas; latency only, not retrieval-quality evidence",
        "numpy_version": np.__version__, "faiss_version": importlib.metadata.version("faiss-cpu"),
        "threads_per_search": 1, "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
