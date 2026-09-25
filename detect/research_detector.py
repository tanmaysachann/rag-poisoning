"""Split-safe research detector trained on held-out benchmark examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from detect.signals import behavioural_signal_components, isolation_forest_score, mahalanobis_score, query_copy_ratio


FEATURE_NAMES = (
    "mahalanobis",
    "isolation_forest",
    "relevance_spike",
    "instruction_pattern",
    "url_pattern",
    "authority_cue",
    "query_doc_cosine",
    "query_copy_ratio",
)


@dataclass
class ResearchDetector:
    embedder: Any
    clean_mean: np.ndarray
    clean_cov_inv: np.ndarray
    isolation_forest: Any
    classifier: Any
    threshold: float
    embedding_model: str

    def features(self, query: str, doc_text: str) -> np.ndarray:
        embedding = self.embedder.encode(doc_text)[0]
        query_embedding = self.embedder.encode(query)[0]
        behaviour = behavioural_signal_components(query, doc_text, self.embedder)
        return np.asarray(
            [
                mahalanobis_score(embedding, self.clean_mean, self.clean_cov_inv),
                isolation_forest_score(embedding, self.isolation_forest),
                behaviour["relevance_spike"],
                behaviour["instruction"],
                behaviour["url"],
                behaviour["authority"],
                float(embedding @ query_embedding),
                query_copy_ratio(query, doc_text),
            ],
            dtype=np.float64,
        )

    def score(self, query: str, doc_text: str) -> dict:
        features = self.features(query, doc_text)
        risk = float(self.classifier.predict_proba(features.reshape(1, -1))[0, 1])
        return {
            "risk_score": risk,
            "threshold": self.threshold,
            "decision": "quarantine" if risk >= self.threshold else "accept",
            "features": dict(zip(FEATURE_NAMES, features.tolist())),
        }


def threshold_for_fpr(clean_scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold permitting at most floor(target_fpr * N) clean flags."""
    values = np.sort(np.asarray(clean_scores, dtype=np.float64))
    if values.ndim != 1 or not len(values):
        raise ValueError("clean_scores must be a nonempty one-dimensional array")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    allowed = int(np.floor(target_fpr * len(values)))
    boundary = values[-(allowed + 1)]
    return float(np.nextafter(boundary, np.inf))
