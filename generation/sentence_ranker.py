"""Trainable sentence selection for the research answerer (no ground truth at inference)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from detect.signals import split_sentences

FEATURE_NAMES = (
    "query_term_coverage", "query_sentence_cosine", "retrieval_rank_reciprocal",
    "rrf_score", "bm25_rank_reciprocal", "dense_rank_reciprocal",
    "sentence_position", "sentence_words", "has_digit", "starts_with_yes_no",
    "question_who", "question_when", "question_where", "question_how",
)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), re.UNICODE)


def _contains_alias(sentence: str, aliases: list[str]) -> bool:
    words = _words(sentence)
    return any(
        (tokens := _words(alias)) and any(words[index:index + len(tokens)] == tokens for index in range(len(words) - len(tokens) + 1))
        for alias in aliases
    )


def sentence_candidates(query: str, documents: list[dict], retriever) -> tuple[np.ndarray, list[dict]]:
    rows = []
    query_words = set(_words(query)) - {"what", "which", "who", "where", "when", "how", "the", "is", "are", "a", "an", "of", "to", "in"}
    question_start = _words(query)[:1]
    for rank, doc in enumerate(documents, 1):
        sentences = split_sentences(doc["text"])
        for index, sentence in enumerate(sentences):
            rows.append((doc, rank, sentence, index / max(len(sentences) - 1, 1)))
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []
    query_vector = retriever.encode(query)[0]
    vectors = retriever.encode([row[2] for row in rows])
    features = []
    metadata = []
    for (doc, rank, sentence, position), vector in zip(rows, vectors):
        words = set(_words(sentence))
        features.append([
            len(query_words & words) / max(len(query_words), 1),
            float(query_vector @ vector),
            1.0 / rank,
            float(doc.get("rrf_score", 0.0)),
            1.0 / max(int(doc.get("bm25_rank", rank)), 1),
            1.0 / max(int(doc.get("dense_rank", rank)), 1),
            position,
            min(len(_words(sentence)), 100) / 100.0,
            float(bool(re.search(r"\d", sentence))),
            float(bool(re.match(r"\s*(yes|no)\b", sentence, re.I))),
            float(question_start == ["who"]),
            float(question_start == ["when"]),
            float(question_start == ["where"]),
            float(question_start == ["how"]),
        ])
        metadata.append({"sentence": sentence, "doc_id": int(doc["doc_id"])})
    return np.asarray(features, dtype=np.float32), metadata


@dataclass
class SentenceRanker:
    classifier: object
    threshold: float
    embedding_model: str

    def select(self, query: str, documents: list[dict], retriever) -> tuple[str, int | None, str | None]:
        if retriever.embedder.model_name != self.embedding_model:
            raise ValueError("Sentence ranker and retriever use different embedding models")
        features, metadata = sentence_candidates(query, documents, retriever)
        if not len(metadata):
            return "Insufficient relevant evidence was retrieved to answer this question.", None, None
        probabilities = self.classifier.predict_proba(features)[:, 1]
        winner = int(np.argmax(probabilities))
        if probabilities[winner] < self.threshold:
            return "Insufficient relevant evidence was retrieved to answer this question.", None, None
        sentence = metadata[winner]["sentence"]
        return sentence, metadata[winner]["doc_id"], sentence
