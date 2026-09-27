"""Masked factored actor-critic for document edits."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from poison.actions import EditAction, OPERATIONS, PAYLOADS, POSITIONS


class FactoredActorCritic(nn.Module):
    def __init__(self, state_dim: int = 774, hidden_dim: int = 128,
                 conditional_heads: bool = True):
        super().__init__()
        self.conditional_heads = conditional_heads
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.operation_head = nn.Linear(hidden_dim, len(OPERATIONS))
        self.position_head = nn.Linear(hidden_dim + len(OPERATIONS), len(POSITIONS))
        self.payload_head = nn.Linear(hidden_dim + len(OPERATIONS) + len(POSITIONS), len(PAYLOADS))
        self.value_head = nn.Linear(hidden_dim, 1)
        self.proxy_value_head = nn.Linear(hidden_dim, 1)

    def distributions(self, states: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor | None = None, deterministic: bool = False):
        features = self.trunk(states)
        operation_mask = masks.any(dim=2).any(dim=2)
        operation_distribution = Categorical(logits=self.operation_head(features).masked_fill(~operation_mask, -1e9))
        operation = (operation_distribution.logits.argmax(dim=1) if deterministic else operation_distribution.sample()) if actions is None else actions[:, 0]
        op_hot = nn.functional.one_hot(operation, len(OPERATIONS)).float()
        if not self.conditional_heads:
            op_hot = torch.zeros_like(op_hot)
        batch = torch.arange(len(states), device=states.device)
        position_mask = masks[batch, operation].any(dim=2)
        position_features = torch.cat([features, op_hot], dim=1)
        position_distribution = Categorical(logits=self.position_head(position_features).masked_fill(~position_mask, -1e9))
        position = (position_distribution.logits.argmax(dim=1) if deterministic else position_distribution.sample()) if actions is None else actions[:, 1]
        pos_hot = nn.functional.one_hot(position, len(POSITIONS)).float()
        if not self.conditional_heads:
            pos_hot = torch.zeros_like(pos_hot)
        payload_mask = masks[batch, operation, position]
        payload_features = torch.cat([features, op_hot, pos_hot], dim=1)
        payload_distribution = Categorical(logits=self.payload_head(payload_features).masked_fill(~payload_mask, -1e9))
        payload = (payload_distribution.logits.argmax(dim=1) if deterministic else payload_distribution.sample()) if actions is None else actions[:, 2]
        chosen = torch.stack([operation, position, payload], dim=1)
        log_probability = (
            operation_distribution.log_prob(operation)
            + position_distribution.log_prob(position)
            + payload_distribution.log_prob(payload)
        )
        entropy = (
            operation_distribution.entropy()
            + position_distribution.entropy()
            + payload_distribution.entropy()
        )
        return chosen, log_probability, entropy, self.value_head(features).squeeze(-1), self.proxy_value_head(features).squeeze(-1)

    @torch.no_grad()
    def act(self, state, mask, *, deterministic: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        action, log_probability, _, value, proxy_value = self.distributions(state_tensor, mask_tensor, deterministic=deterministic)
        op, position, payload = action[0].tolist()
        return EditAction(OPERATIONS[op], position, payload), float(log_probability[0]), float(value[0]), float(proxy_value[0])
