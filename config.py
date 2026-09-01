"""Shared paths and runtime configuration for the phase-1 demo."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RESULTS_DIR = ROOT_DIR / "results"

BASE_CORPUS_PATH = DATA_DIR / "corpus.jsonl"
DEMO_CORPUS_PATH = DATA_DIR / "demo_corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
POISONED_DOCS_PATH = DATA_DIR / "poisoned_docs.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60

# Set RAG_USE_LLM=1 to opt into generation. The extractive backend is the
# reliable default for a CPU-only, potentially offline live demonstration.
USE_LLM = os.getenv("RAG_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}


def ensure_project_dirs() -> None:
    """Create directories used for generated data and artifacts."""

    for path in (DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
