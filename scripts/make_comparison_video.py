"""Render a side-by-side 3-panel MP4 of DQN, PPO, Safe PPO playing MountainCar.

Layout: [DQN | PPO | Safe PPO] horizontally, each panel showing the rendered
environment plus a header strip with the algorithm name and a footer strip
with episode/step/reward (and cost+lambda for Safe PPO).

Panels are step-synchronized: every video frame is one environment step. If an
agent finishes its episode early, its panel freezes on the final frame until
the longest-running agent finishes.

Usage:
    python scripts/make_comparison_video.py \
        --dqn-checkpoint results/dqn/seed_0/qnet.pt \
        --ppo-checkpoint results/ppo/seed_0/ppo_net.pt \
        --safe-checkpoint results/safe_ppo/seed_0/safe_ppo_net.pt \
        --episodes 3 \
        --out videos/comparison_3algo.mp4

If a checkpoint is missing the corresponding panel shows a "checkpoint not
available — run scripts/run_<algo>.py first" placeholder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs import ConstrainedMountainCar, ShapedMountainCar  # noqa: E402
from src.networks import PPONetwork, QNetwork, SafePPONetwork  # noqa: E402


PANEL_W, PANEL_H = 600, 400
HEADER_H, FOOTER_H = 30, 50


def _font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


HEADER_FONT = _font(18)
FOOTER_FONT = _font(14)


def _placeholder_panel(name: str, reason: str) -> Image.Image:
    img = Image.new("RGB", (PANEL_W, PANEL_H), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.text((20, PANEL_H // 2 - 30), name, fill=(255, 200, 0), font=HEADER_FONT)
    draw.text((20, PANEL_H // 2), reason, fill=(220, 220, 220), font=FOOTER_FONT)
    return img


def _wrap_panel(env_img: np.ndarray | None, name: str,
                footer_text: str, color_header: tuple[int, int, int]) -> Image.Image:
    panel = Image.new("RGB", (PANEL_W, PANEL_H + HEADER_H + FOOTER_H), color=(20, 20, 20))
    draw = ImageDraw.Draw(panel)

    # Header strip
    draw.rectangle([(0, 0), (PANEL_W, HEADER_H)], fill=color_header)
    draw.text((10, 5), name, fill=(255, 255, 255), font=HEADER_FONT)

    # Body
    if env_img is not None:
        body = Image.fromarray(env_img).resize((PANEL_W, PANEL_H))
        panel.paste(body, (0, HEADER_H))

    # Footer strip
    draw.rectangle([(0, HEADER_H + PANEL_H), (PANEL_W, HEADER_H + PANEL_H + FOOTER_H)],
                   fill=(0, 0, 0))
    # Wrap long footer text into max 2 lines
    lines = footer_text.split("\n")
    for i, line in enumerate(lines[:2]):
        draw.text((10, HEADER_H + PANEL_H + 5 + i * 18), line,
                  fill=(255, 255, 255), font=FOOTER_FONT)
    return panel


def _greedy_dqn(qnet: QNetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        return int(torch.argmax(qnet(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))).item())


def _greedy_ppo(net: PPONetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        logits = net.actor(torch.as_tensor(obs, dtype=torch.float32))
        return int(torch.argmax(logits).item())


def _greedy_safe(net: SafePPONetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        logits = net.actor(torch.as_tensor(obs, dtype=torch.float32))
        return int(torch.argmax(logits).item())


def _maybe_load_dqn(path: Path | None) -> QNetwork | None:
    if path is None or not Path(path).exists():
        return None
    net = QNetwork(obs_dim=2, act_dim=3, hidden_size=64)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def _maybe_load_ppo(path: Path | None) -> PPONetwork | None:
    if path is None or not Path(path).exists():
        return None
    net = PPONetwork(obs_dim=2, act_dim=3, hidden_size=64)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def _maybe_load_safe(path: Path | None) -> SafePPONetwork | None:
    if path is None or not Path(path).exists():
        return None
    net = SafePPONetwork(obs_dim=2, act_dim=3, hidden_size=64)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def run_episode_frames(env: gym.Env, agent_fn, max_steps: int,
                       safe: bool = False) -> list[dict]:
    """Run one episode; return per-step list of {frame, info_text}."""
    obs, _ = env.reset()
    frames: list[dict] = []
    ep_r = 0.0
    ep_c = 0.0
    last_speed = 0.0

    for step in range(1, max_steps + 1):
        action = agent_fn(obs)
        obs, rew, term, trunc, info = env.step(action)
        ep_r += float(rew)
        if safe:
            ep_c += float(info.get("cost", 0.0))
        last_speed = float(abs(obs[1]))
        frames.append({
            "frame": env.render(),
            "step": step,
            "ep_r": ep_r,
            "ep_c": ep_c,
            "speed": last_speed,
            "violating": safe and last_speed > 0.04,
        })
        if term or trunc:
            break
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-checkpoint", default=str(ROOT / "results/dqn/seed_0/qnet.pt"))
    parser.add_argument("--ppo-checkpoint", default=str(ROOT / "results/ppo/seed_0/ppo_net.pt"))
    parser.add_argument("--safe-checkpoint", default=str(ROOT / "results/safe_ppo/seed_0/safe_ppo_net.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=str(ROOT / "videos/comparison_3algo.mp4"))
    args = parser.parse_args()

    dqn_net = _maybe_load_dqn(Path(args.dqn_checkpoint))
    ppo_net = _maybe_load_ppo(Path(args.ppo_checkpoint))
    safe_net = _maybe_load_safe(Path(args.safe_checkpoint))

    print(f"DQN     : {'loaded' if dqn_net else 'MISSING ' + args.dqn_checkpoint}")
    print(f"PPO     : {'loaded' if ppo_net else 'MISSING ' + args.ppo_checkpoint}")
    print(f"SafePPO : {'loaded' if safe_net else 'MISSING ' + args.safe_checkpoint}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=8)

    for ep in range(1, args.episodes + 1):
        print(f"\n=== Episode {ep}/{args.episodes} ===")
        # Build envs (fresh per episode so they're synchronized at reset)
        envs = {}
        runs = {}

        if dqn_net is not None:
            envs["dqn"] = ShapedMountainCar(gym.make("MountainCar-v0", render_mode="rgb_array"))
            runs["dqn"] = run_episode_frames(envs["dqn"],
                                             lambda o: _greedy_dqn(dqn_net, o),
                                             args.max_steps)
        if ppo_net is not None:
            envs["ppo"] = ShapedMountainCar(gym.make("MountainCar-v0", render_mode="rgb_array"))
            runs["ppo"] = run_episode_frames(envs["ppo"],
                                             lambda o: _greedy_ppo(ppo_net, o),
                                             args.max_steps)
        if safe_net is not None:
            envs["safe"] = ConstrainedMountainCar(gym.make("MountainCar-v0", render_mode="rgb_array"),
                                                  max_speed=0.04)
            runs["safe"] = run_episode_frames(envs["safe"],
                                              lambda o: _greedy_safe(safe_net, o),
                                              args.max_steps, safe=True)

        for env in envs.values():
            env.close()

        # Step-synchronize: pad shorter runs by freezing on last frame
        max_len = max((len(r) for r in runs.values()), default=0)
        for k in list(runs.keys()):
            if runs[k]:
                last = runs[k][-1]
                while len(runs[k]) < max_len:
                    runs[k].append({**last, "step": last["step"]})

        # Render frames
        for t in range(max_len):
            # DQN panel
            if "dqn" in runs:
                f = runs["dqn"][t]
                footer = (f"Ep {ep}  Step {f['step']}\n"
                          f"Return {f['ep_r']:+7.1f}   speed {f['speed']:.3f}")
                panel_dqn = _wrap_panel(f["frame"], "DQN", footer, (30, 90, 180))
            else:
                panel_dqn = _wrap_panel(None, "DQN", "checkpoint missing", (60, 60, 60))
                panel_dqn = _placeholder_panel("DQN", "run scripts/run_dqn.py first")
                panel_dqn = panel_dqn.resize((PANEL_W, PANEL_H + HEADER_H + FOOTER_H))

            # PPO panel
            if "ppo" in runs:
                f = runs["ppo"][t]
                footer = (f"Ep {ep}  Step {f['step']}\n"
                          f"Return {f['ep_r']:+7.1f}   speed {f['speed']:.3f}")
                panel_ppo = _wrap_panel(f["frame"], "PPO", footer, (30, 140, 60))
            else:
                panel_ppo = _placeholder_panel("PPO", "run scripts/run_ppo.py first")
                panel_ppo = panel_ppo.resize((PANEL_W, PANEL_H + HEADER_H + FOOTER_H))

            # Safe PPO panel (extra cost line)
            if "safe" in runs:
                f = runs["safe"][t]
                vio_marker = "  VIOLATING!" if f["violating"] else ""
                footer = (f"Ep {ep}  Step {f['step']}   speed {f['speed']:.3f}{vio_marker}\n"
                          f"Return {f['ep_r']:+7.1f}   cost {f['ep_c']:5.1f}   limit 0.04")
                color = (180, 40, 40) if f["violating"] else (190, 40, 90)
                panel_safe = _wrap_panel(f["frame"], "Safe PPO", footer, color)
            else:
                panel_safe = _placeholder_panel("Safe PPO", "run scripts/run_safe_ppo.py first")
                panel_safe = panel_safe.resize((PANEL_W, PANEL_H + HEADER_H + FOOTER_H))

            # Concatenate panels horizontally
            total_w = 3 * PANEL_W
            total_h = PANEL_H + HEADER_H + FOOTER_H
            canvas = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))
            canvas.paste(panel_dqn, (0, 0))
            canvas.paste(panel_ppo, (PANEL_W, 0))
            canvas.paste(panel_safe, (2 * PANEL_W, 0))

            writer.append_data(np.asarray(canvas))

    writer.close()
    print(f"\nSaved comparison video to {args.out}")


if __name__ == "__main__":
    main()
