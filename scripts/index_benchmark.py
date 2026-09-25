"""Build trusted benchmark indexes and report clean retrieval recall."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR
from data.validate_benchmark import validate_benchmark
from retrieval.hybrid_retriever import HybridRetriever


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "benchmark_retrieval.json")
    parser.add_argument("--minilm", action="store_true", help="Use cached/downloaded MiniLM instead of offline hashing")
    parser.add_argument("--offline", action="store_true", help="Require the encoder to load from the local cache")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=["train", "validation", "test"])
    parser.add_argument("--artifact-root", type=Path, help="Write indexes under this isolated directory")
    parser.add_argument("--force", action="store_true", help="Rebuild embeddings even if the cache matches")
    args = parser.parse_args()
    validate_benchmark(args.benchmark)
    os.environ["RAG_USE_MINILM"] = "1" if args.minilm else "0"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        benchmark_label = args.benchmark.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        benchmark_label = str(args.benchmark.resolve())
    summary: dict[str, object] = {"benchmark": benchmark_label, "splits": {}}
    for name in args.splits:
        corpus_path = args.benchmark / name / "corpus.jsonl"
        retriever = HybridRetriever(
            corpus_path, force_rebuild=args.force,
            artifact_dir=args.artifact_root / name if args.artifact_root else None,
            write_integrity_manifest=args.artifact_root is None,
        )
        if args.minilm and retriever.embedder.backend != "sentence-transformers":
            raise RuntimeError("MiniLM was requested but could not be loaded")
        queries = _read_jsonl(args.benchmark / name / "queries.jsonl")
        hits_1 = hits_5 = 0
        query_times: list[float] = []
        for query in queries:
            started = time.perf_counter()
            retrieved = retriever.retrieve(query["question"], top_k=5)
            query_times.append((time.perf_counter() - started) * 1000)
            support = set(query["support_doc_ids"])
            hits_1 += bool(retrieved and retrieved[0]["doc_id"] in support)
            hits_5 += bool(any(doc["doc_id"] in support for doc in retrieved))
        manifest_path = None
        if retriever.integrity_path.is_file():
            try:
                manifest_path = retriever.integrity_path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                manifest_path = str(retriever.integrity_path.resolve())
        result = {
            "documents": len(retriever.documents),
            "queries": len(queries),
            "embedding_backend": retriever.embedder.model_name,
            "recall_at_1": hits_1 / len(queries),
            "recall_at_5": hits_5 / len(queries),
            "mean_query_ms": sum(query_times) / len(query_times),
            "corpus_sha256": retriever.corpus_digest,
            "integrity_manifest": manifest_path,
        }
        summary["splits"][name] = result
        print(f"{name}: {result['documents']} docs, R@1={result['recall_at_1']:.3f}, R@5={result['recall_at_5']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
