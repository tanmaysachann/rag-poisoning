"""Explainable statistical and behavioural signals for document screening."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
INSTRUCTION_PATTERNS = (
    r"\bignore (?:all|any|the|every|previous|surrounding|above)\b",
    r"\balways (?:recommend|visit|answer|respond|say|trust)\b",
    r"\b(?:click|visit|open|follow)\b.{0,45}\b(?:link|url|website|https?)\b",
    r"\b(?:do not|don't) (?:trust|use|follow)\b",
    r"\bthe answer is\b",
)
AUTHORITY_CUES = (
    r"\bofficial(?:ly)?\b",
    r"\bcorrection\b",
    r"\bshould be (?:trusted|preferred|accepted)\b",
    r"\bnewly revised\b",
    r"\bauthoritative\b",
    r"\bdespite the surrounding\b",
)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]


def mahalanobis_score(doc_embedding: np.ndarray, clean_mean: np.ndarray,
                      clean_cov_inv: np.ndarray) -> float:
    """Return ridge-regularized distance from the clean embedding distribution."""
    delta = np.asarray(doc_embedding) - np.asarray(clean_mean)
    squared = float(delta @ np.asarray(clean_cov_inv) @ delta)
    return float(np.sqrt(max(0.0, squared)))


def isolation_forest_score(doc_embedding: np.ndarray, fitted_iforest: Any) -> float:
    """Return Isolation Forest anomaly score with higher values meaning riskier."""
    value = fitted_iforest.decision_function(np.asarray(doc_embedding).reshape(1, -1))[0]
    return float(np.clip(0.5 - value, 0.0, 1.0))


def behavioural_signal_components(query: str, doc_text: str, embedder: Any) -> dict[str, float]:
    """Measure whether a document appears engineered to dominate or hijack output.

    The presentation's SRQ uses a small language model to generate a preliminary
    response and then measures semantic vocabulary overlap with a retrieved
    document. For the bounded Review-1 MVP, this deterministic approximation uses
    sentence-level query relevance spikes plus instruction, URL, and manufactured
    authority cues. The stable public function remains ``srq_heuristic_score`` so a
    true SLM-backed implementation can replace it later.
    """
    sentences = split_sentences(doc_text)
    if not sentences:
        return {"relevance_spike": 0.0, "instruction": 0.0, "url": 0.0,
                "authority": 0.0, "srq": 0.0}
    q = embedder.encode(query)[0]
    sentence_embeddings = embedder.encode(sentences)
    similarities = sentence_embeddings @ q
    spike = float(np.clip(similarities.max() - similarities.mean(), 0.0, 1.0))
    instruction_hits = sum(bool(re.search(pattern, doc_text, re.I)) for pattern in INSTRUCTION_PATTERNS)
    authority_hits = sum(bool(re.search(pattern, doc_text, re.I)) for pattern in AUTHORITY_CUES)
    instruction = min(1.0, instruction_hits / 2.0)
    url = 1.0 if URL_RE.search(doc_text) else 0.0
    authority = min(1.0, authority_hits / 2.0)
    srq = float(np.clip(0.35 * spike + 0.35 * instruction + 0.15 * url + 0.15 * authority, 0.0, 1.0))
    return {"relevance_spike": spike, "instruction": instruction, "url": url,
            "authority": authority, "srq": srq}


def srq_heuristic_score(query: str, doc_text: str) -> float:
    """Compatibility wrapper for the deterministic SRQ approximation."""
    from retrieval.hybrid_retriever import TextEmbedder
    return behavioural_signal_components(query, doc_text, TextEmbedder(preferred_backend="hashing"))["srq"]
