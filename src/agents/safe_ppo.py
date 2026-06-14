"""Safe PPO with PPO-Lagrangian and dual-critic network (Phase 3 novelty)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers import SafePPORolloutBuffer
from ..envs import ConstrainedMountainCar
from ..networks import SafePPONetwork
from ..utils import ensure_dir, save_json, set_seed


@dataclass
class SafePPOConfig:
    epochs: int = 150
    steps_per_epoch: int = 4000
    max_ep_len: int = 200
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    lam_lr: float = 5e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 0.01
    hidden_size: int = 64
    max_speed: float = 0.04
    cost_limit: float = 15.0
    log_lambda_init: float = -3.0
    log_every: int = 5
    record_every: int = 0  # if > 0, snapshot a rollout video every N epochs


def train_safe_ppo(seed: int, cfg: SafePPOConfig, log_dir: str | Path) -> dict:
    set_seed(seed)
    log_dir = ensure_dir(log_dir)

    env = ConstrainedMountainCar(gym.make("MountainCar-v0"), max_speed=cfg.max_speed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = SafePPONetwork(obs_dim, act_dim, cfg.hidden_size)
    pi_opt = optim.Adam(net.actor.parameters(), lr=cfg.pi_lr)
    v_r_opt = optim.Adam(net.reward_critic.parameters(), lr=cfg.vf_lr)
    v_c_opt = optim.Adam(net.cost_critic.parameters(), lr=cfg.vf_lr)

    log_lambda = torch.tensor(cfg.log_lambda_init, requires_grad=True)
    lam_opt = optim.Adam([log_lambda], lr=cfg.lam_lr)

    buf = SafePPORolloutBuffer(cfg.steps_per_epoch, obs_dim, gamma=cfg.gamma, lam=cfg.lam)

    def update():
        data = buf.get()
        obs_b, act_b, logp_old = data["obs"], data["act"], data["logp"]
        adv_r, adv_c = data["adv_r"], data["adv_c"]
        ret_r, ret_c = data["ret_r"], data["ret_c"]

        current_lambda = float(torch.exp(log_lambda).item())

        for _ in range(cfg.train_pi_iters):
            pi_opt.zero_grad()
            logp, _, _, entropy = net.evaluate(obs_b, act_b)
            ratio = torch.exp(logp - logp_old)
            clip_adv_r = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_r
            obj_r = torch.min(ratio * adv_r, clip_adv_r).mean()
            obj_c = (ratio * adv_c).mean()
            loss_pi = -(obj_r - current_lambda * obj_c) - cfg.ent_coef * entropy.mean()
            loss_pi.backward()
            pi_opt.step()

        for _ in range(cfg.train_v_iters):
            v_r_opt.zero_grad()
            v_c_opt.zero_grad()
            _, v_r, v_c, _ = net.evaluate(obs_b, act_b)
            loss_v_r = F.mse_loss(v_r, ret_r)
            loss_v_c = F.mse_loss(v_c, ret_c)
            loss_v_r.backward()
            v_r_opt.step()
            loss_v_c.backward()
            v_c_opt.step()

        # Lambda dual ascent update
        lam_opt.zero_grad()
        actual_batch_cost = float(ret_c.mean().item())
        loss_lambda = -torch.exp(log_lambda) * (actual_batch_cost - cfg.cost_limit)
        loss_lambda.backward()
        lam_opt.step()

        new_lambda = float(torch.exp(log_lambda).item())
        return (
            float(loss_pi.item()),
            float(loss_v_r.item()),
            float(loss_v_c.item()),
            new_lambda,
            actual_batch_cost,
        )

    # Optional training-process video recorder
    recorder = None
    render_env = None
    if cfg.record_every and cfg.record_every > 0:
        from ..recording import TrainingRecorder
        render_env = ConstrainedMountainCar(
            gym.make("MountainCar-v0", render_mode="rgb_array"), max_speed=cfg.max_speed
        )
        recorder = TrainingRecorder(
            out_path=Path(log_dir) / "training.mp4",
            algorithm="safe_ppo",
            seed=seed,
            hyperparams=dict(lam_lr=cfg.lam_lr, log_lam0=cfg.log_lambda_init,
                             cost_limit=cfg.cost_limit, pi_lr=cfg.pi_lr),
            total_epochs=cfg.epochs,
            cost_limit=cfg.cost_limit,
        )

        def _policy_fn(o):
            with torch.no_grad():
                logits = net.actor(torch.as_tensor(o, dtype=torch.float32))
                return int(torch.argmax(logits).item())

    epoch_returns: list[float] = []
    epoch_costs: list[float] = []
    epoch_lambdas: list[float] = []
    epoch_solved_rates: list[float] = []
    epoch_violations: list[int] = []  # number of steps in epoch with cost > 0

    obs, _ = env.reset()
    ep_ret_r, ep_ret_c, ep_len = 0.0, 0.0, 0
    start = time.time()

    for epoch in range(cfg.epochs):
        rets: list[float] = []
        costs: list[float] = []
        violations = 0
        solved = 0
        completed_eps = 0

        for t in range(cfg.steps_per_epoch):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                act, logp, v_r, v_c = net.get_action(obs_tensor)

            next_obs, rew, term, trunc, info = env.step(act)
            cost = float(info.get("cost", 0.0))
            if cost > 0:
                violations += 1

            ep_ret_r += float(rew)
            ep_ret_c += cost
            ep_len += 1

            buf.store(obs, act, rew, cost, v_r.item(), v_c.item(), logp.item())
            obs = next_obs

            terminal = term or trunc
            epoch_ended = t == cfg.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                        _, _, v_r, v_c = net.get_action(obs_tensor)
                    last_r, last_c = float(v_r.item()), float(v_c.item())
                else:
                    last_r, last_c = 0.0, 0.0

                buf.finish_path(last_val_r=last_r, last_val_c=last_c)

                if terminal:
                    rets.append(ep_ret_r)
                    costs.append(ep_ret_c)
                    solved += int(bool(term))
                    completed_eps += 1

                obs, _ = env.reset()
                ep_ret_r, ep_ret_c, ep_len = 0.0, 0.0, 0

        loss_pi, loss_v_r, loss_v_c, current_lambda, _ = update()
        avg_ret = float(np.mean(rets)) if rets else 0.0
        avg_cost = float(np.mean(costs)) if costs else 0.0
        solved_rate = (solved / completed_eps) if completed_eps else 0.0
        epoch_returns.append(avg_ret)
        epoch_costs.append(avg_cost)
        epoch_lambdas.append(current_lambda)
        epoch_solved_rates.append(solved_rate)
        epoch_violations.append(violations)

        if (epoch + 1) % cfg.log_every == 0:
            print(
                f"[SafePPO seed={seed}] Ep {epoch+1:3d} | ret {avg_ret:7.2f} | "
                f"cost {avg_cost:5.2f} | lam {current_lambda:6.3f} | "
                f"solved {solved_rate:.2f} | viol {violations}"
            )

        if recorder is not None and (
            (epoch + 1) % cfg.record_every == 0 or epoch == cfg.epochs - 1
        ):
            recorder.snapshot(
                epoch=epoch + 1,
                env=render_env,
                policy_fn=_policy_fn,
                returns_history=epoch_returns,
                costs_history=epoch_costs,
                lam=current_lambda,
                max_steps=cfg.max_ep_len,
            )

    env.close()
    if recorder is not None:
        recorder.close()
        render_env.close()
    duration = time.time() - start

    metrics = dict(
        algorithm="safe_ppo",
        seed=seed,
        epochs=cfg.epochs,
        epoch_returns=epoch_returns,
        epoch_costs=epoch_costs,
        epoch_lambdas=epoch_lambdas,
        epoch_solved_rates=epoch_solved_rates,
        epoch_violations=epoch_violations,
        wall_time_sec=duration,
        config=cfg.__dict__,
    )
    save_json(metrics, Path(log_dir) / "metrics.json")
    torch.save(net.state_dict(), Path(log_dir) / "safe_ppo_net.pt")
    print(f"[SafePPO seed={seed}] done in {duration:.1f}s -> {log_dir}")
    return metrics
