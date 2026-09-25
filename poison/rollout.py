"""Baseline rollouts against the same edit MDP used for policy training."""

from __future__ import annotations

import random

import numpy as np

from poison.actions import EditAction, OPERATIONS


def valid_actions(mask: np.ndarray) -> list[EditAction]:
    return [
        EditAction(OPERATIONS[int(op)], int(position), int(payload))
        for op, position, payload in np.argwhere(mask)
    ]


def run_episode(env, *, seed: int, strategy: str = "random", cache=None) -> dict:
    if strategy not in {"random", "greedy_proxy", "greedy_repeat"}:
        raise ValueError("Unknown rollout strategy")
    rng = random.Random(seed)
    state, info = env.reset(seed=seed)
    trace = []
    while True:
        actions = valid_actions(info["action_mask"])
        if not actions:
            raise RuntimeError("No valid action is available")
        if strategy == "random":
            action = rng.choice(actions)
        else:
            non_stop = [action for action in actions if action.operation != "STOP"]
            if not non_stop or (strategy == "greedy_proxy" and env.steps > 0):
                action = EditAction("STOP")
            else:
                # Fixed payload ordering provides a budget-matched strong baseline.
                action = next(
                    (candidate for candidate in non_stop if candidate.operation == "INSERT"
                     and candidate.position == 0 and candidate.payload == (1 if env.steps == 0 else 2)),
                    non_stop[0],
                )
        next_state, reward, terminated, truncated, info = env.step(action)
        if cache is not None:
            cache.add(state, action, next_state, reward)
        trace.append({
            "action": action.__dict__, "reward": reward,
            "valid": info["valid"], "reward_components": info["reward_components"],
        })
        state = next_state
        if terminated or truncated:
            return {
                "trace": trace, "final_document": info["document"],
                "total_reward": sum(step["reward"] for step in trace),
                "terminal": info["terminal"], "state_version": info["state_version"],
            }
