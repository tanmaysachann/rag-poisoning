"""Inspectable nearest-neighbor memory of observed edit effects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poison.actions import EditAction


@dataclass
class EditEffectCache:
    state_dim: int
    rows: list[tuple[np.ndarray, EditAction, np.ndarray, float]] = field(default_factory=list)

    def add(self, before: np.ndarray, action: EditAction, after: np.ndarray, reward: float) -> None:
        before = np.asarray(before, dtype=np.float32)
        after = np.asarray(after, dtype=np.float32)
        if before.shape != (self.state_dim,) or after.shape != (self.state_dim,):
            raise ValueError("State dimensions do not match cache schema")
        self.rows.append((before.copy(), action, (after - before).copy(), float(reward)))

    def neighbors(self, state: np.ndarray, action: EditAction, k: int = 5) -> list[dict]:
        if k <= 0:
            raise ValueError("k must be positive")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError("State dimensions do not match cache schema")
        matches = []
        for row_index, (before, stored_action, delta, reward) in enumerate(self.rows):
            if stored_action != action:
                continue
            distance = float(np.linalg.norm(before - state))
            matches.append((distance, row_index, delta, reward))
        matches.sort(key=lambda row: (row[0], row[1]))
        return [
            {"distance": distance, "row_index": index, "delta": delta.copy(), "reward": reward}
            for distance, index, delta, reward in matches[:k]
        ]

    def predict(self, state: np.ndarray, action: EditAction, k: int = 5) -> dict | None:
        nearby = self.neighbors(state, action, k)
        if not nearby:
            return None
        weights = np.asarray([1.0 / max(item["distance"], 1e-6) for item in nearby])
        weights /= weights.sum()
        return {
            "next_state": np.asarray(state, dtype=np.float32)
            + sum(weight * item["delta"] for weight, item in zip(weights, nearby)),
            "expected_reward": float(sum(weight * item["reward"] for weight, item in zip(weights, nearby))),
            "neighbors": len(nearby),
        }
