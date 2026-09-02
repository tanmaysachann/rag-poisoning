"""BM25 + dense retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import json
import math
import re
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from config import (
    ARTIFACTS_DIR,
    BASE_CORPUS_PATH,
    DEMO_CORPUS_PATH,
    EMBEDDING_MODEL,
    RRF_K,
    ensure_project_dirs,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class _SimpleBM25:
    """Small dependency-free BM25 fallback with rank_bm25-compatible output."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = [list(tokens) for tokens in corpus]
        self.k1 = k1
        self.b = b
        self.lengths = np.asarray([len(tokens) for tokens in self.corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.lengths)) if len(self.lengths) else 1.0
        self.term_freqs = [Counter(tokens) for tokens in self.corpus]
        doc_freq: Counter[str] = Counter()
        for tokens in self.corpus:
            doc_freq.update(set(tokens))
        n_docs = max(len(self.corpus), 1)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=np.float32)
        for index, term_freq in enumerate(self.term_freqs):
            doc_len = self.lengths[index]
            for term in query_tokens:
                frequency = term_freq.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-12)
                )
                scores[index] += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
        return scores


class TextEmbedder:
    """MiniLM encoder with a fixed-width hashing fallback for offline use."""

    def __init__(self, preferred_backend: str | None = None):
        self.backend = "hashing"
        self.model_name = "sklearn-hashing-384"
        self._model = None
        self._hashing = None

        if preferred_backend != "hashing":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(EMBEDDING_MODEL)
                self.backend = "sentence-transformers"
                self.model_name = EMBEDDING_MODEL
            except Exception as error:
                print(f"MiniLM unavailable ({error}); using deterministic hashing embeddings.")

        if self.backend == "hashing":
            from sklearn.feature_extraction.text import HashingVectorizer

            self._hashing = HashingVectorizer(
                n_features=384,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
            )

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        values = [texts] if isinstance(texts, str) else list(texts)
        if self._model is not None:
            encoded = self._model.encode(
                values,
                batch_size=32,
                show_progress_bar=len(values) > 64,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(encoded, dtype=np.float32)
        assert self._hashing is not None
        return self._hashing.transform(values).toarray().astype(np.float32)


class HybridRetriever:
    """Retrieve documents with BM25 and dense cosine similarity, then RRF."""

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        *,
        force_rebuild: bool = False,
        write_integrity_manifest: bool = False,
    ) -> None:
        ensure_project_dirs()
        if corpus_path is None:
            corpus_path = DEMO_CORPUS_PATH if DEMO_CORPUS_PATH.exists() else BASE_CORPUS_PATH
        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {self.corpus_path}. Run scripts/build_index.py first."
            )

        self.documents = self._load_documents(self.corpus_path)
        if not self.documents:
            raise ValueError(f"Corpus is empty: {self.corpus_path}")
        tokenized = [tokenize(doc["text"]) for doc in self.documents]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(tokenized)
        except ImportError:
            self.bm25 = _SimpleBM25(tokenized)

        artifact_prefix = "demo_corpus" if self.corpus_path.resolve() == DEMO_CORPUS_PATH.resolve() else "corpus"
        self.embeddings_path = ARTIFACTS_DIR / f"{artifact_prefix}_embeddings.npy"
        self.embedding_meta_path = ARTIFACTS_DIR / f"{artifact_prefix}_embeddings.meta.json"
        self.integrity_path = ARTIFACTS_DIR / f"{artifact_prefix}_hashes.json"
        self.corpus_digest = hashlib.sha256(self.corpus_path.read_bytes()).hexdigest()

        cached_meta = self._read_json(self.embedding_meta_path)
        use_minilm = os.getenv("RAG_USE_MINILM", "0").strip().lower() in {"1", "true", "yes", "on"}
        preferred_backend = None if use_minilm else "hashing"
        self.embedder = TextEmbedder(preferred_backend=preferred_backend)
        cache_valid = (
            not force_rebuild
            and self.embeddings_path.exists()
            and cached_meta.get("corpus_sha256") == self.corpus_digest
            and cached_meta.get("backend") == self.embedder.backend
        )
        if cache_valid:
            self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        else:
            self.embeddings = _normalize_rows(
                self.embedder.encode([doc["text"] for doc in self.documents])
            )
            try:
                np.save(self.embeddings_path, self.embeddings)
                self.embedding_meta_path.write_text(
                    json.dumps(
                        {
                            "backend": self.embedder.backend,
                            "model": self.embedder.model_name,
                            "corpus_sha256": self.corpus_digest,
                            "shape": list(self.embeddings.shape),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                # Serverless deployments expose the application bundle as
                # read-only. Rebuilt embeddings remain valid in memory for the
                # current invocation even when the cache cannot be persisted.
                pass

        if write_integrity_manifest or not self.integrity_path.exists():
            self.write_integrity_manifest()

    @staticmethod
    def _load_documents(path: Path) -> list[dict]:
        documents: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    documents.append({**item, "doc_id": int(item["doc_id"]), "text": str(item["text"])})
        return documents

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def write_integrity_manifest(self) -> None:
        hashes = {
            str(doc["doc_id"]): hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
            for doc in self.documents
        }
        self.integrity_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        """Encode text with the same normalized backend used for this index."""

        return _normalize_rows(self.embedder.encode(texts))

    def dense_search(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Return corpus indexes and scores in descending dense-similarity order.

        This interface intentionally isolates the numpy implementation so a FAISS
        implementation can replace it without changing ``retrieve`` callers.
        """

        query_embedding = self.encode(query)[0]
        scores = self.embeddings @ query_embedding
        order = np.argsort(-scores, kind="stable")
        return order, scores[order]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            return []

        documents = self.documents
        embeddings = self.embeddings
        bm25 = self.bm25

        bm25_scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float32)
        bm25_order = np.argsort(-bm25_scores, kind="stable")
        query_embedding = self.encode(query)[0]
        dense_scores = embeddings @ query_embedding
        dense_order = np.argsort(-dense_scores, kind="stable")
        bm25_ranks = np.empty(len(documents), dtype=np.int32)
        dense_ranks = np.empty(len(documents), dtype=np.int32)
        bm25_ranks[bm25_order] = np.arange(1, len(documents) + 1)
        dense_ranks[dense_order] = np.arange(1, len(documents) + 1)
        fused_scores = 1.0 / (RRF_K + bm25_ranks) + 1.0 / (RRF_K + dense_ranks)
        fused_order = np.argsort(-fused_scores, kind="stable")[: min(top_k, len(documents))]

        results: list[dict] = []
        for index in fused_order:
            doc = documents[int(index)]
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", f"Document {doc['doc_id']}"),
                    "source_type": doc.get("source_type", "controlled corpus"),
                    "text": doc["text"],
                    "score": float(fused_scores[index]),
                    "bm25_rank": int(bm25_ranks[index]),
                    "dense_rank": int(dense_ranks[index]),
                }
            )
        return results
