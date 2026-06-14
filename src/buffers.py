"""Replay and rollout buffers for DQN, PPO, and Safe PPO."""
from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import scipy.signal
import torch


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Discounted cumulative sum, used for GAE and returns."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class DQNReplayBuffer:
    """Fixed-size FIFO buffer for off-policy DQN training."""

    def __init__(self, capacity: int = 50000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PPORolloutBuffer:
    """On-policy rollout buffer with GAE-Lambda for standard PPO."""

    def __init__(self, size: int, obs_dim: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp) -> None:
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self) -> dict:
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        adv_mean = float(np.mean(self.adv_buf))
        adv_std = float(np.std(self.adv_buf))
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(
            obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
            adv=self.adv_buf, logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class SafePPORolloutBuffer:
    """Dual-GAE rollout buffer for Safe PPO (CMDP)."""

    def __init__(self, size: int, obs_dim: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        # Reward channel
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_r_buf = np.zeros(size, dtype=np.float32)
        self.adv_r_buf = np.zeros(size, dtype=np.float32)
        self.ret_r_buf = np.zeros(size, dtype=np.float32)

        # Cost channel
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.val_c_buf = np.zeros(size, dtype=np.float32)
        self.adv_c_buf = np.zeros(size, dtype=np.float32)
        self.ret_c_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, cost, val_r, val_c, logp) -> None:
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_r_buf[self.ptr] = val_r
        self.val_c_buf[self.ptr] = val_c
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val_r: float = 0.0, last_val_c: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val_r)
        vals_r = np.append(self.val_r_buf[path_slice], last_val_r)
        deltas_r = rews[:-1] + self.gamma * vals_r[1:] - vals_r[:-1]
        self.adv_r_buf[path_slice] = discount_cumsum(deltas_r, self.gamma * self.lam)
        self.ret_r_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_val_c)
        vals_c = np.append(self.val_c_buf[path_slice], last_val_c)
        deltas_c = costs[:-1] + self.gamma * vals_c[1:] - vals_c[:-1]
        self.adv_c_buf[path_slice] = discount_cumsum(deltas_c, self.gamma * self.lam)
        self.ret_c_buf[path_slice] = discount_cumsum(costs, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self) -> dict:
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # Normalize reward advantage (mean 0, std 1)
        adv_r_mean = float(np.mean(self.adv_r_buf))
        adv_r_std = float(np.std(self.adv_r_buf))
        self.adv_r_buf = (self.adv_r_buf - adv_r_mean) / (adv_r_std + 1e-8)

        # Scale cost advantage by std only (no mean-centering, sign matters).
        # Guard: when the batch has no real cost, adv_c is pure cost-critic noise;
        # normalizing it would blow that noise up to unit variance and inject it
        # into the policy gradient via the lambda term. Zero it instead so Safe PPO
        # reduces to PPO until genuine constraint violations appear.
        if np.any(self.cost_buf != 0.0):
            self.adv_c_buf = self.adv_c_buf / (float(np.std(self.adv_c_buf)) + 1e-8)
        else:
            self.adv_c_buf[:] = 0.0

        data = dict(
            obs=self.obs_buf, act=self.act_buf, logp=self.logp_buf,
            ret_r=self.ret_r_buf, ret_c=self.ret_c_buf,
            adv_r=self.adv_r_buf, adv_c=self.adv_c_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
