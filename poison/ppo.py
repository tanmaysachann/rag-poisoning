"""Clipped PPO update with terminal-aware generalized advantage estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from poison.actions import OPERATIONS


@dataclass
class Transition:
    state: np.ndarray
    mask: np.ndarray
    action: tuple[int, int, int]
    old_log_probability: float
    value: float
    proxy_value: float
    reward: float
    proxy_reward: float
    done: bool


def collect_episode(env, policy, seed: int) -> tuple[list[Transition], dict]:
    state, info = env.reset(seed=seed)
    transitions = []
    while True:
        mask = info["action_mask"]
        action, log_probability, value, proxy_value = policy.act(state, mask)
        next_state, reward, terminated, truncated, info = env.step(action)
        transitions.append(Transition(
            state.copy(), mask.copy(),
            (OPERATIONS.index(action.operation), action.position, action.payload),
            log_probability, value, proxy_value, float(reward),
            float(info["reward_components"]["total"]), terminated or truncated,
        ))
        state = next_state
        if terminated or truncated:
            return transitions, info["terminal"]


def generalized_advantages(transitions: list[Transition], *, gamma: float = 0.99, gae_lambda: float = 0.95, proxy: bool = False):
    if not transitions:
        raise ValueError("Transitions cannot be empty")
    advantages = np.zeros(len(transitions), dtype=np.float32)
    estimate = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        row = transitions[index]
        next_value = 0.0 if row.done else (
            transitions[index + 1].proxy_value if proxy else transitions[index + 1].value
        )
        value = row.proxy_value if proxy else row.value
        reward = row.proxy_reward if proxy else row.reward
        estimate = reward + gamma * next_value * (not row.done) - value + gamma * gae_lambda * (not row.done) * estimate
        advantages[index] = estimate
    values = np.asarray([row.proxy_value if proxy else row.value for row in transitions], dtype=np.float32)
    return advantages, advantages + values


def ppo_update(
    policy, optimizer, transitions: list[Transition], *, epochs: int = 4,
    clip_ratio: float = 0.2, value_weight: float = 0.5,
    proxy_value_weight: float = 0.1, entropy_weight: float = 0.01,
) -> dict[str, float]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    advantages, returns = generalized_advantages(transitions)
    _, proxy_returns = generalized_advantages(transitions, proxy=True)
    advantage_tensor = torch.as_tensor(advantages)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)
    states = torch.as_tensor(np.stack([row.state for row in transitions]), dtype=torch.float32)
    masks = torch.as_tensor(np.stack([row.mask for row in transitions]), dtype=torch.bool)
    actions = torch.as_tensor([row.action for row in transitions], dtype=torch.long)
    old_log = torch.as_tensor([row.old_log_probability for row in transitions], dtype=torch.float32)
    returns = torch.as_tensor(returns)
    proxy_returns = torch.as_tensor(proxy_returns)
    metrics = {}
    for _ in range(epochs):
        _, log_probability, entropy, value, proxy_value = policy.distributions(states, masks, actions)
        ratio = torch.exp(log_probability - old_log)
        policy_loss = -torch.minimum(
            ratio * advantage_tensor,
            torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_tensor,
        ).mean()
        value_loss = torch.mean((value - returns) ** 2)
        proxy_loss = torch.mean((proxy_value - proxy_returns) ** 2)
        loss = policy_loss + value_weight * value_loss + proxy_value_weight * proxy_loss - entropy_weight * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        metrics = {
            "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()), "proxy_value_loss": float(proxy_loss.detach()),
            "entropy": float(entropy.mean().detach()),
        }
    return metrics
