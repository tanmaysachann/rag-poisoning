"""Symmetric query-conditioned features for two related evidence passages."""

from __future__ import annotations

import re

import numpy as np


FEATURE_NAMES = (
    "minilm_document_cosine", "content_jaccard", "content_overlap_coefficient",
    "content_bigram_jaccard", "longest_common_content_run_fraction",
    "content_length_ratio", "document_length_ratio",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "had", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _longest_common_run(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    best = 0
    for token in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, 1):
            if token == other:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def pair_features(query: str, left: str, right: str, embedder) -> np.ndarray:
    question = set(_tokens(query)) | STOPWORDS
    raw_left, raw_right = _tokens(left), _tokens(right)
    content_left = [token for token in raw_left if token not in question]
    content_right = [token for token in raw_right if token not in question]
    words_left, words_right = set(content_left), set(content_right)
    bigrams_left = set(zip(content_left, content_left[1:]))
    bigrams_right = set(zip(content_right, content_right[1:]))
    vectors = embedder.encode([left, right])
    return np.asarray([
        float(vectors[0] @ vectors[1]),
        _jaccard(words_left, words_right),
        len(words_left & words_right) / max(min(len(words_left), len(words_right)), 1),
        _jaccard(bigrams_left, bigrams_right),
        _longest_common_run(content_left, content_right) / max(min(len(content_left), len(content_right)), 1),
        min(len(content_left), len(content_right)) / max(max(len(content_left), len(content_right)), 1),
        min(len(raw_left), len(raw_right)) / max(max(len(raw_left), len(raw_right)), 1),
    ], dtype=np.float64)
