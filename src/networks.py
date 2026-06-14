"""Network architectures: QNetwork (DQN), PPONetwork, SafePPONetwork (dual-critic)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class QNetwork(nn.Module):
    """MLP estimating Q-values for each discrete action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp(obs_dim: int, hidden_size: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, out_dim),
    )


class PPONetwork(nn.Module):
    """Actor-critic with categorical policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden_size, act_dim)
        self.critic = _mlp(obs_dim, hidden_size, 1)

    def get_action(self, obs: torch.Tensor):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        val = self.critic(obs).squeeze(-1)
        return action.item(), dist.log_prob(action), val

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        val = self.critic(obs).squeeze(-1)
        return log_prob, val, entropy


class SafePPONetwork(nn.Module):
    """Dual-critic actor: one critic for reward, one for safety cost."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden_size, act_dim)
        self.reward_critic = _mlp(obs_dim, hidden_size, 1)
        self.cost_critic = _mlp(obs_dim, hidden_size, 1)

    def get_action(self, obs: torch.Tensor):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        v_r = self.reward_critic(obs).squeeze(-1)
        v_c = self.cost_critic(obs).squeeze(-1)
        return action.item(), dist.log_prob(action), v_r, v_c

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        v_r = self.reward_critic(obs).squeeze(-1)
        v_c = self.cost_critic(obs).squeeze(-1)
        return log_prob, v_r, v_c, entropy
