"""Analysis script: aggregates metrics across seeds and regenerates paper figures.

Run AFTER the multi-seed training completes:
    python -m scripts.run_all --seeds 0 1 2
    python notebooks/analysis.py

Reads results/{algo}/seed_*/metrics.json and writes plots to paper/figures/.

Falls back to seed_0-only mode if other seeds are missing (e.g. when only the
notebook outputs have been parsed). In single-seed mode no error bars are drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 120, "savefig.dpi": 300,
    "savefig.bbox": "tight", "font.family": "serif",
})


def load_seeds(algo: str) -> list[dict]:
    out = []
    for d in sorted((RESULTS / algo).glob("seed_*")):
        m = d / "metrics.json"
        if m.exists():
            out.append(json.loads(m.read_text()))
    return out


def stack(runs: list[dict], key: str) -> np.ndarray:
    arrs = [np.asarray(r[key], dtype=float) for r in runs if key in r]
    if not arrs:
        return np.empty((0, 0))
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs], axis=0)


def x_axis(runs: list[dict], key: str) -> np.ndarray:
    return np.asarray(runs[0][key], dtype=float)


def plot_mean_band(ax, x, y_runs, **kw):
    if y_runs.shape[0] >= 2:
        mean = y_runs.mean(0)
        std = y_runs.std(0)
        ax.plot(x, mean, **kw)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2,
                        color=kw.get("color", None))
    else:
        ax.plot(x, y_runs[0], **kw)


def main() -> None:
    dqn = load_seeds("dqn")
    ppo = load_seeds("ppo")
    safe = load_seeds("safe_ppo")
    n_seeds = max(len(dqn), len(ppo), len(safe))
    print(f"Found seeds — DQN: {len(dqn)}, PPO: {len(ppo)}, SafePPO: {len(safe)}")

    # --- Figure 3: 3-way return comparison ---
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    if dqn:
        key_y = "avg_reward_last30" if "avg_reward_last30" in dqn[0] else "episode_returns"
        y = stack(dqn, key_y)
        if "logged_episodes" in dqn[0]:
            x = x_axis(dqn, "logged_episodes") * 200 / 1000  # approx env steps in K
        else:
            x = np.arange(1, y.shape[1] + 1) * 200 / 1000
        plot_mean_band(ax, x, y, color="C0", lw=1.3, label="DQN")

    if ppo:
        key_ep = "logged_epochs" if "logged_epochs" in ppo[0] else None
        key_y = "avg_reward" if "avg_reward" in ppo[0] else "epoch_returns"
        if key_ep is None:
            x = np.arange(1, len(ppo[0][key_y]) + 1) * 4000 / 1000
        else:
            x = x_axis(ppo, key_ep) * 4000 / 1000
        y = stack(ppo, key_y)
        plot_mean_band(ax, x, y, color="C2", lw=1.3, label="PPO")

    if safe:
        key_ep = "logged_epochs" if "logged_epochs" in safe[0] else None
        key_y = "avg_reward" if "avg_reward" in safe[0] else "epoch_returns"
        if key_ep is None:
            x = np.arange(1, len(safe[0][key_y]) + 1) * 4000 / 1000
        else:
            x = x_axis(safe, key_ep) * 4000 / 1000
        y = stack(safe, key_y)
        plot_mean_band(ax, x, y, color="C3", lw=1.3, label="Safe PPO (Ours)")

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Environment Steps (thousands)")
    ax.set_ylabel("Shaped Return")
    title = f"Return Comparison ({n_seeds} seed{'s' if n_seeds != 1 else ''})"
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.savefig(FIG / "fig3_comparison_return.pdf")
    plt.savefig(FIG / "fig3_comparison_return.png")
    plt.close()

    # --- Figure 4: Safe PPO return + cost ---
    if safe:
        fig, ax1 = plt.subplots(figsize=(6.0, 3.5))
        x_key = "logged_epochs" if "logged_epochs" in safe[0] else None
        y_r = stack(safe, "avg_reward" if "avg_reward" in safe[0] else "epoch_returns")
        cost_key = "avg_cost" if "avg_cost" in safe[0] else "epoch_costs"
        y_c = stack(safe, cost_key)
        x = x_axis(safe, x_key) if x_key else np.arange(1, y_r.shape[1] + 1)

        plot_mean_band(ax1, x, y_r, color="C2", lw=1.5, label="Return")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Shaped Return", color="C2")
        ax1.tick_params(axis="y", labelcolor="C2")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        plot_mean_band(ax2, x, y_c, color="C3", lw=1.5, label="Cost")
        ax2.axhline(15.0, color="black", lw=1, ls=":", label="Budget $d$=15")
        ax2.set_ylabel("Safety Cost", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

        plt.title(f"Safe PPO: Return vs Cost ({len(safe)} seed{'s' if len(safe) != 1 else ''})")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)
        plt.savefig(FIG / "fig4_safe_ppo_dual.pdf")
        plt.savefig(FIG / "fig4_safe_ppo_dual.png")
        plt.close()

    # --- Figure 5: Lambda trajectory ---
    lam_key = next((k for k in ("lambda", "epoch_lambdas") if safe and k in safe[0]), None)
    if safe and lam_key:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        x_key = "logged_epochs" if "logged_epochs" in safe[0] else None
        y = stack(safe, lam_key)
        x = x_axis(safe, x_key) if x_key else np.arange(1, y.shape[1] + 1)
        plot_mean_band(ax, x, y, color="C4", lw=1.5, label=r"$\lambda$")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Lagrangian multiplier $\lambda$")
        ax.set_title("Adaptive $\\lambda$ Trajectory")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.savefig(FIG / "fig5_lambda_trajectory.pdf")
        plt.savefig(FIG / "fig5_lambda_trajectory.png")
        plt.close()

    # --- Figure 6: cost curve vs budget ---
    if safe:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        x_key = "logged_epochs" if "logged_epochs" in safe[0] else None
        cost_key = "avg_cost" if "avg_cost" in safe[0] else "epoch_costs"
        y_c = stack(safe, cost_key)
        x = x_axis(safe, x_key) if x_key else np.arange(1, y_c.shape[1] + 1)
        plot_mean_band(ax, x, y_c, color="C3", lw=1.5, label="Safe PPO cost")
        ax.axhline(15.0, color="black", lw=1, ls=":", label="Budget $d$=15")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Safety Cost")
        ax.set_title(f"Cost vs Budget ({len(safe)} seed{'s' if len(safe) != 1 else ''})")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.savefig(FIG / "fig6_cost_curve.pdf")
        plt.savefig(FIG / "fig6_cost_curve.png")
        plt.close()

    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
