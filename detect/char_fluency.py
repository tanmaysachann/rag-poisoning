"""Small, auditable character language model for malformed-passage signals."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

ALPHABET = set("abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'-()/[]%$#@+=~")
VOCABULARY = len(ALPHABET)


def normalize(text: str) -> str:
    compact = " ".join(text.lower().split())
    return "^^" + "".join(char if char in ALPHABET else "~" for char in compact[:2000])


class CharFluency:
    """Laplace-smoothed trigram likelihood fitted only to clean source text."""

    def __init__(self, smoothing: float = 0.2) -> None:
        self.smoothing = smoothing
        self.trigrams: Counter[str] = Counter()
        self.contexts: Counter[str] = Counter()
        self.documents = 0

    def fit(self, texts) -> "CharFluency":
        for text in texts:
            sequence = normalize(text)
            for index in range(2, len(sequence)):
                self.trigrams[sequence[index - 2:index + 1]] += 1
                self.contexts[sequence[index - 2:index]] += 1
            self.documents += 1
        if not self.documents:
            raise ValueError("Clean training text is required")
        return self

    def features(self, text: str) -> np.ndarray:
        sequence = normalize(text)
        losses = []
        for index in range(2, len(sequence)):
            context = sequence[index - 2:index]
            gram = sequence[index - 2:index + 1]
            numerator = self.trigrams[gram] + self.smoothing
            denominator = self.contexts[context] + self.smoothing * VOCABULARY
            losses.append(-math.log(numerator / denominator))
        if not losses:
            return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        values = np.asarray(losses, dtype=np.float64)
        highest = max(1, int(math.ceil(len(values) * 0.1)))
        tail_mean = float(np.partition(values, -highest)[-highest:].mean())
        window = min(40, len(values))
        moving = np.convolve(values, np.ones(window) / window, mode="valid")
        punctuation_burst = sum(
            sequence[index] == sequence[index - 1] and sequence[index] in "!?.~"
            for index in range(3, len(sequence))
        ) / max(len(sequence) - 2, 1)
        return np.asarray([float(values.mean()), tail_mean, float(moving.max()),
                           float(punctuation_burst)], dtype=np.float64)
