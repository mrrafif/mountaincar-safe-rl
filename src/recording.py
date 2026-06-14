"""Training-process video recorder (torch-free).

During training, an agent calls ``TrainingRecorder.snapshot(...)`` every K epochs.
Each snapshot runs one deterministic rollout of the *current* policy, renders the
environment, and composites every frame into a panel that shows:

    +------------------------------------------------------------------+
    |  HEADER: algorithm, seed, and the hyperparameters used           |
    +-----------------------------------+------------------------------+
    |                                   |                              |
    |   left: rendered environment      |   right: reward curve so far |
    |        (current rollout)          |        (grows each snapshot) |
    |                                   |                              |
    +-----------------------------------+------------------------------+
    |  FOOTER: epoch, step, return  (+ cost, lambda for Safe PPO)      |
    +------------------------------------------------------------------+

Frames are appended straight to a per-run MP4 (``training.mp4``), so memory stays
flat regardless of how many snapshots are taken. A separate script
(``scripts/make_grid_video.py``) tiles several of these per-run videos into one
synchronized comparison grid.

This module imports no torch: the agent passes a ``policy_fn(obs) -> action``
callable, so the recorder never needs to know about the network internals. That
also makes the whole video pipeline testable without PyTorch installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PANEL_W, PANEL_H = 600, 400
HEADER_H, FOOTER_H = 72, 40
CANVAS_W = 2 * PANEL_W
CANVAS_H = HEADER_H + PANEL_H + FOOTER_H

_BG = (24, 24, 27)
_FG = (235, 235, 235)
_ACCENT = (55, 138, 221)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


_F_TITLE = _font(22, bold=True)
_F_HYPER = _font(15)
_F_FOOT = _font(16)


def format_hyperparams(hyper: Mapping[str, object]) -> str:
    """Render a hyperparameter dict as a compact one-liner."""
    parts = []
    for k, v in hyper.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "   ".join(parts)


def _render_curve(
    returns: Sequence[float],
    total_epochs: int,
    current_epoch: int,
    costs: Sequence[float] | None = None,
    cost_limit: float | None = None,
) -> np.ndarray:
    """Render the reward (and optional cost) curve up to ``current_epoch``."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(PANEL_W / dpi, PANEL_H / dpi), dpi=dpi)
    fig.patch.set_facecolor(np.array(_BG) / 255)
    ax.set_facecolor(np.array(_BG) / 255)

    xs = np.arange(1, len(returns) + 1)
    ax.plot(xs, returns, color="#378ADD", lw=2, label="return")
    if len(returns):
        ax.scatter([xs[-1]], [returns[-1]], color="#FFFFFF", s=28, zorder=5)
    ax.set_xlim(1, max(total_epochs, 2))
    ax.set_xlabel("epoch", color=_FG_HEX())
    ax.set_ylabel("return", color="#378ADD")
    ax.tick_params(colors=_FG_HEX())
    for spine in ax.spines.values():
        spine.set_color("#555")

    if costs is not None:
        ax2 = ax.twinx()
        cxs = np.arange(1, len(costs) + 1)
        ax2.plot(cxs, costs, color="#D85A30", lw=1.6, label="cost")
        if cost_limit is not None:
            ax2.axhline(cost_limit, color="#9A9A9A", lw=1.2, ls=":")
        ax2.set_ylabel("cost", color="#D85A30")
        ax2.tick_params(axis="y", colors="#D85A30")
        ax2.set_ylim(0, None)
        for spine in ax2.spines.values():
            spine.set_color("#555")

    ax.set_title(f"learning curve  (epoch {current_epoch}/{total_epochs})",
                 color=_FG_HEX(), fontsize=11)
    ax.grid(alpha=0.18)
    fig.tight_layout(pad=0.6)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    if (h, w) != (PANEL_H, PANEL_W):
        img = np.asarray(Image.fromarray(img).resize((PANEL_W, PANEL_H)))
    return img


def _FG_HEX() -> str:
    return "#%02x%02x%02x" % _FG


def _fit_panel(frame: np.ndarray) -> np.ndarray:
    """Resize an env render frame to the panel size."""
    img = Image.fromarray(frame).convert("RGB").resize((PANEL_W, PANEL_H))
    return np.asarray(img)


class TrainingRecorder:
    """Accumulates training snapshots into a single per-run MP4."""

    def __init__(
        self,
        out_path: str | Path,
        algorithm: str,
        seed: int,
        hyperparams: Mapping[str, object],
        total_epochs: int,
        cost_limit: float | None = None,
        fps: int = 30,
        eval_seed: int = 12345,
    ):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm
        self.seed = seed
        self.hyperparams = dict(hyperparams)
        self.total_epochs = total_epochs
        self.cost_limit = cost_limit
        self.eval_seed = eval_seed
        self._title = f"{algorithm.upper()}  ·  seed {seed}"
        self._hyper_str = format_hyperparams(hyperparams)
        self.writer = imageio.get_writer(
            str(self.out_path), fps=fps, macro_block_size=8, codec="libx264"
        )
        self.n_snapshots = 0

    # -- header / footer painting -------------------------------------------
    def _base_canvas(self) -> Image.Image:
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), color=_BG)
        d = ImageDraw.Draw(canvas)
        d.text((16, 10), self._title, font=_F_TITLE, fill=_ACCENT)
        d.text((16, 42), self._hyper_str, font=_F_HYPER, fill=_FG)
        return canvas

    def _compose(
        self,
        env_frame: np.ndarray,
        curve_img: np.ndarray,
        epoch: int,
        step: int,
        ret: float,
        cost: float | None,
        lam: float | None,
    ) -> np.ndarray:
        canvas = self._base_canvas()
        canvas.paste(Image.fromarray(_fit_panel(env_frame)), (0, HEADER_H))
        canvas.paste(Image.fromarray(curve_img), (PANEL_W, HEADER_H))
        d = ImageDraw.Draw(canvas)
        foot_y = HEADER_H + PANEL_H + 10
        foot = f"epoch {epoch}/{self.total_epochs}    step {step:3d}    return {ret:7.2f}"
        if cost is not None:
            foot += f"    cost {cost:6.2f}"
        if lam is not None:
            foot += f"    lambda {lam:6.3f}"
        d.text((16, foot_y), foot, font=_F_FOOT, fill=_FG)
        return np.asarray(canvas)

    # -- the per-epoch snapshot ---------------------------------------------
    def snapshot(
        self,
        epoch: int,
        env,
        policy_fn: Callable[[np.ndarray], int],
        returns_history: Sequence[float],
        costs_history: Sequence[float] | None = None,
        lam: float | None = None,
        max_steps: int = 200,
    ) -> None:
        """Run one deterministic rollout and append its frames to the video.

        ``env`` must be created with ``render_mode='rgb_array'``. ``policy_fn``
        maps an observation to a (deterministic) action. The reward curve drawn
        on the right uses ``returns_history`` (the epoch returns so far).
        """
        curve = _render_curve(
            returns_history, self.total_epochs, epoch,
            costs=costs_history, cost_limit=self.cost_limit,
        )
        latest_ret = returns_history[-1] if len(returns_history) else 0.0
        latest_cost = (costs_history[-1] if costs_history and len(costs_history)
                       else (0.0 if costs_history is not None else None))

        obs, _ = env.reset(seed=self.eval_seed + self.n_snapshots)
        for step in range(max_steps):
            frame = env.render()
            self.writer.append_data(
                self._compose(frame, curve, epoch, step, latest_ret, latest_cost, lam)
            )
            action = policy_fn(np.asarray(obs, dtype=np.float32))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                # hold the final frame briefly so the outcome is visible
                final = env.render()
                for _ in range(15):
                    self.writer.append_data(
                        self._compose(final, curve, epoch, step + 1,
                                      latest_ret, latest_cost, lam)
                    )
                break
        self.n_snapshots += 1

    def close(self) -> None:
        self.writer.close()
