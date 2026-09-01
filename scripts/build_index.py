"""Build the small streamed datasets and cache the clean retrieval index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_CORPUS_PATH
from data.build_corpus import build_corpus_from_msmarco
from data.build_queries import build_queries_from_nq_open
from retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-passages", type=int, default=500)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Reuse existing corpus/query JSONL files and only rebuild cached embeddings.",
    )
    args = parser.parse_args()

    if not args.reuse_data:
        build_corpus_from_msmarco(args.n_passages, args.seed)
        build_queries_from_nq_open(args.n_queries)
    retriever = HybridRetriever(
        BASE_CORPUS_PATH, force_rebuild=True, write_integrity_manifest=True
    )
    print(
        f"Indexed {len(retriever.documents)} clean documents with "
        f"{retriever.embedder.model_name}; embeddings={retriever.embeddings.shape}."
    )


if __name__ == "__main__":
    main()
