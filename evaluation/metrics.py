"""Small-sample binary intervals and paired query bootstrap utilities."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total and total > 0")
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def paired_bootstrap_difference(
    before: list[bool], after: list[bool], *,
    draws: int = 5000, seed: int = 42,
) -> tuple[float, float]:
    if not before or len(before) != len(after) or draws < 100:
        raise ValueError("Require aligned nonempty observations and at least 100 draws")
    deltas = np.asarray(before, dtype=float) - np.asarray(after, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    estimates = deltas[samples].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)
