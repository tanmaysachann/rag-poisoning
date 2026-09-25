"""Actual MiniLM encoder hidden-state and attention measurements.

These are model-internal probes, unlike the demo's sentence-embedding proxy.
They use a cached encoder, not decoder activations from the answer generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from config import EMBEDDING_MODEL
from detect.signals import mahalanobis_score

PROBE_VERSION = "minilm-encoder-layer3-attn6-v1"


def attention_concentration(attention: np.ndarray) -> float:
    """Mean normalized Herfindahl concentration; uniform=0, one-token=1."""
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Expected attention with shape (heads, tokens, tokens)")
    length = values.shape[-1]
    if length < 2:
        return 0.0
    sums = values.sum(axis=-1, keepdims=True)
    probabilities = values / np.maximum(sums, 1e-12)
    squared = np.sum(probabilities ** 2, axis=-1)
    normalized = (length * squared - 1.0) / (length - 1.0)
    return float(np.clip(normalized.mean(), 0.0, 1.0))


@dataclass
class ProbeReference:
    mean: np.ndarray
    precision: np.ndarray
    attention_mean: float
    attention_std: float

    @classmethod
    def fit(cls, hidden_vectors: np.ndarray, attentions: np.ndarray) -> "ProbeReference":
        vectors = np.asarray(hidden_vectors, dtype=np.float64)
        concentrations = np.asarray(attentions, dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) < 5 or concentrations.shape != (len(vectors),):
            raise ValueError("Need at least five aligned clean hidden vectors and attention scores")
        covariance = LedoitWolf(store_precision=True).fit(vectors)
        return cls(covariance.location_, covariance.precision_,
                   float(concentrations.mean()), float(max(concentrations.std(), 1e-6)))

    def score(self, hidden_vector: np.ndarray, attention: float) -> dict[str, float]:
        return {
            "layer3_mahalanobis": mahalanobis_score(hidden_vector, self.mean, self.precision),
            "attention_deviation": abs(attention - self.attention_mean) / self.attention_std,
            "attention_concentration": attention,
        }


class MiniLMModelProbes:
    def __init__(self, *, local_files_only: bool = True, max_length: int = 192):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, local_files_only=local_files_only, attn_implementation="eager"
        )
        self.model.eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        self.max_length = max_length

    def extract(self, query: str, document: str) -> tuple[np.ndarray, float]:
        import torch

        encoded = self.tokenizer(
            query, document, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            output = self.model(**encoded, output_hidden_states=True, output_attentions=True)
        # Layer 3 is internal to the six-layer MiniLM encoder, not the final
        # pooled sentence embedding used by hybrid retrieval.
        hidden = output.hidden_states[3][0].mean(dim=0).cpu().numpy().astype(np.float32)
        attention = output.attentions[-1][0].cpu().numpy()
        return hidden, attention_concentration(attention)
