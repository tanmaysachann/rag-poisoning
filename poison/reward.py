"""Explicit proxy and terminal rewards for local attack-policy experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    relevance: float = 0.5
    detection: float = 0.5
    edit: float = -0.1
    invalid: float = -1.0
    retrieved: float = 0.2
    attack_success: float = 5.0


def step_reward(
    *, relevance_delta: float, detector_risk: float | None,
    valid: bool, weights: RewardWeights = RewardWeights(),
) -> dict[str, float]:
    """A small shaping reward; success is assessed only at episode end."""
    if not valid:
        return {"invalid": weights.invalid, "total": weights.invalid}
    parts = {
        "relevance": weights.relevance * max(-1.0, min(1.0, relevance_delta)),
        "detection": -weights.detection * detector_risk if detector_risk is not None else 0.0,
        "edit": weights.edit,
    }
    parts["total"] = sum(parts.values())
    return parts


def terminal_reward(
    *, retrieved: bool, attack_success: bool, valid_attack: bool = True,
    weights: RewardWeights = RewardWeights(),
) -> dict[str, float]:
    if attack_success and not retrieved:
        raise ValueError("Attack success requires retrieval")
    if attack_success and not valid_attack:
        raise ValueError("Attack success requires a real edit")
    parts = {
        "retrieved": weights.retrieved if retrieved and valid_attack else 0.0,
        "attack_success": weights.attack_success if attack_success else 0.0,
        "no_edit": 0.0 if valid_attack else weights.invalid,
    }
    parts["total"] = sum(parts.values())
    return parts
