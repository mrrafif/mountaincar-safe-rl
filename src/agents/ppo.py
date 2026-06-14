"""Standard PPO agent and training loop (single-critic baseline)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers import PPORolloutBuffer
from ..envs import ShapedMountainCar
from ..networks import PPONetwork
from ..utils import ensure_dir, save_json, set_seed


@dataclass
class PPOConfig:
    epochs: int = 150
    steps_per_epoch: int = 4000
    max_ep_len: int = 200
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 0.01
    hidden_size: int = 64
    max_speed: float = 0.04  # threshold for passive cost logging (no effect on training)
    log_every: int = 5
    record_every: int = 0  # if > 0, snapshot a rollout video every N epochs


def train_ppo(seed: int, cfg: PPOConfig, log_dir: str | Path) -> dict:
    set_seed(seed)
    log_dir = ensure_dir(log_dir)

    env = ShapedMountainCar(gym.make("MountainCar-v0"))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = PPONetwork(obs_dim, act_dim, cfg.hidden_size)
    pi_opt = optim.Adam(net.actor.parameters(), lr=cfg.pi_lr)
    v_opt = optim.Adam(net.critic.parameters(), lr=cfg.vf_lr)
    buf = PPORolloutBuffer(cfg.steps_per_epoch, obs_dim, gamma=cfg.gamma, lam=cfg.lam)

    def update():
        data = buf.get()
        obs_b, act_b, adv_b = data["obs"], data["act"], data["adv"]
        ret_b, logp_old = data["ret"], data["logp"]

        for _ in range(cfg.train_pi_iters):
            pi_opt.zero_grad()
            logp, _, entropy = net.evaluate(obs_b, act_b)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_b
            loss_pi = -(torch.min(ratio * adv_b, clip_adv)).mean() - cfg.ent_coef * entropy.mean()
            loss_pi.backward()
            pi_opt.step()

        for _ in range(cfg.train_v_iters):
            v_opt.zero_grad()
            _, val, _ = net.evaluate(obs_b, act_b)
            loss_v = F.mse_loss(val, ret_b)
            loss_v.backward()
            v_opt.step()

        return float(loss_pi.item()), float(loss_v.item())

    # Optional training-process video recorder
    recorder = None
    render_env = None
    if cfg.record_every and cfg.record_every > 0:
        from ..recording import TrainingRecorder
        render_env = ShapedMountainCar(gym.make("MountainCar-v0", render_mode="rgb_array"))
        recorder = TrainingRecorder(
            out_path=Path(log_dir) / "training.mp4",
            algorithm="ppo",
            seed=seed,
            hyperparams=dict(pi_lr=cfg.pi_lr, ent_coef=cfg.ent_coef,
                             gamma=cfg.gamma, clip=cfg.clip_ratio),
            total_epochs=cfg.epochs,
        )

        def _policy_fn(o):
            with torch.no_grad():
                logits = net.actor(torch.as_tensor(o, dtype=torch.float32))
                return int(torch.argmax(logits).item())

    epoch_returns: list[float] = []
    epoch_costs: list[float] = []  # passive: speeding cost, logged but not optimized
    epoch_solved_rates: list[float] = []
    obs, _ = env.reset()
    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
    start = time.time()

    for epoch in range(cfg.epochs):
        rets: list[float] = []
        costs: list[float] = []
        solved = 0
        completed_eps = 0

        for t in range(cfg.steps_per_epoch):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                act, logp, val = net.get_action(obs_tensor)

            next_obs, rew, term, trunc, _ = env.step(act)
            vel = abs(float(next_obs[1]))
            ep_cost += (vel - cfg.max_speed) * 100.0 if vel > cfg.max_speed else 0.0
            ep_ret += float(rew)
            ep_len += 1

            buf.store(obs, act, rew, val.item(), logp.item())
            obs = next_obs

            terminal = term or trunc
            epoch_ended = t == cfg.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                        _, _, val = net.get_action(obs_tensor)
                    last_val = float(val.item())
                else:
                    last_val = 0.0

                buf.finish_path(last_val=last_val)

                if terminal:
                    rets.append(ep_ret)
                    costs.append(ep_cost)
                    solved += int(bool(term))
                    completed_eps += 1

                obs, _ = env.reset()
                ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

        loss_pi, loss_v = update()
        avg_ret = float(np.mean(rets)) if rets else 0.0
        avg_cost = float(np.mean(costs)) if costs else 0.0
        solved_rate = (solved / completed_eps) if completed_eps else 0.0
        epoch_returns.append(avg_ret)
        epoch_costs.append(avg_cost)
        epoch_solved_rates.append(solved_rate)

        if (epoch + 1) % cfg.log_every == 0:
            print(f"[PPO seed={seed}] Ep {epoch+1:3d} | avg ret {avg_ret:7.2f} | "
                  f"cost {avg_cost:6.2f} | solved {solved_rate:.2f} | "
                  f"pi {loss_pi:6.3f} | v {loss_v:6.3f}")

        if recorder is not None and (
            (epoch + 1) % cfg.record_every == 0 or epoch == cfg.epochs - 1
        ):
            recorder.snapshot(
                epoch=epoch + 1,
                env=render_env,
                policy_fn=_policy_fn,
                returns_history=epoch_returns,
                costs_history=epoch_costs,
                max_steps=cfg.max_ep_len,
            )

    env.close()
    if recorder is not None:
        recorder.close()
        render_env.close()
    duration = time.time() - start

    metrics = dict(
        algorithm="ppo",
        seed=seed,
        epochs=cfg.epochs,
        epoch_returns=epoch_returns,
        epoch_costs=epoch_costs,
        epoch_solved_rates=epoch_solved_rates,
        wall_time_sec=duration,
        config=cfg.__dict__,
    )
    save_json(metrics, Path(log_dir) / "metrics.json")
    torch.save(net.state_dict(), Path(log_dir) / "ppo_net.pt")
    print(f"[PPO seed={seed}] done in {duration:.1f}s -> {log_dir}")
    return metrics
