"""DQN agent and training loop."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers import DQNReplayBuffer
from ..envs import ShapedMountainCar
from ..networks import QNetwork
from ..utils import ensure_dir, save_json, set_seed


@dataclass
class DQNConfig:
    episodes: int = 450
    max_steps: int = 200
    batch_size: int = 64
    buffer_capacity: int = 50000
    target_sync_freq: int = 5
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    hidden_size: int = 64
    log_every: int = 30


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DQNConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size

        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay

        self.q_net = QNetwork(obs_dim, act_dim, cfg.hidden_size)
        self.target_net = QNetwork(obs_dim, act_dim, cfg.hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.act_dim)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(torch.argmax(q_values).item())

    def update(self, buffer: DQNReplayBuffer) -> float:
        if len(buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        q_values = self.q_net(states).gather(1, actions).squeeze(-1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn(seed: int, cfg: DQNConfig, log_dir: str | Path) -> dict:
    """Train DQN with a single seed; returns a dict of metrics."""
    set_seed(seed)
    log_dir = ensure_dir(log_dir)

    env = ShapedMountainCar(gym.make("MountainCar-v0"))
    env.reset(seed=seed)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DQNAgent(obs_dim, act_dim, cfg)
    buffer = DQNReplayBuffer(capacity=cfg.buffer_capacity)

    episode_returns = []
    episode_lengths = []
    episode_solved = []
    start = time.time()

    for ep in range(1, cfg.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_len = 0
        terminated_flag = False

        for step in range(cfg.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, float(done))
            agent.update(buffer)

            obs = next_obs
            ep_reward += float(reward)
            ep_len += 1

            if done:
                terminated_flag = bool(terminated)
                break

        agent.decay_epsilon()
        if ep % cfg.target_sync_freq == 0:
            agent.sync_target_network()

        episode_returns.append(ep_reward)
        episode_lengths.append(ep_len)
        episode_solved.append(int(terminated_flag))

        if ep % cfg.log_every == 0:
            avg = float(np.mean(episode_returns[-cfg.log_every:]))
            print(f"[DQN seed={seed}] Ep {ep:3d} | avg return {avg:7.2f} | eps {agent.epsilon:.3f}")

    env.close()
    duration = time.time() - start

    metrics = dict(
        algorithm="dqn",
        seed=seed,
        episodes=cfg.episodes,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_solved=episode_solved,
        wall_time_sec=duration,
        config=cfg.__dict__,
    )
    save_json(metrics, Path(log_dir) / "metrics.json")
    torch.save(agent.q_net.state_dict(), Path(log_dir) / "qnet.pt")
    print(f"[DQN seed={seed}] done in {duration:.1f}s -> {log_dir}")
    return metrics
