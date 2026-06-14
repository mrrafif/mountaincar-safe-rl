"""Stage 1 driver: the P1-P6 base-PPO grid, each into its own result folder.

Implements the Stage 1 grid from PLAN.md Part 2: vary ``lr x ent_coef`` over six
combinations, each across multiple seeds, with training-process video recording.
After training it builds comparison grids and prints the top-3 selection table
(score = mean_return - std_return, the explicit rule from the plan).

Layout produced::

    results/ppo_P1/seed_0/{metrics.json, ppo_net.pt, training.mp4}
    results/ppo_P1/seed_1/...
    ...
    videos/stage1_P1_seeds.mp4     # 3 seeds of P1 side by side
    videos/stage1_all_configs.mp4  # one cell per combo (seed_0), 6-up grid
    results/stage1_summary.md      # the filled P1-P6 selection table

Examples
--------
Full run (local, needs PyTorch), record every 10 epochs::

    python -m scripts.run_stage1 --seeds 0 1 2 --record-every 10

Only rebuild grids + summary from existing results (no torch needed)::

    python -m scripts.run_stage1 --skip-train

Run just two combos::

    python -m scripts.run_stage1 --configs P3 P4
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_grid_video import build_grid  # noqa: E402

# The Stage 1 grid: lr x ent_coef  (see PLAN.md Part 2, Stage 1 table)
GRID: dict[str, dict[str, float]] = {
    "P1": dict(pi_lr=1e-4, ent_coef=0.0),
    "P2": dict(pi_lr=1e-4, ent_coef=0.01),
    "P3": dict(pi_lr=3e-4, ent_coef=0.0),
    "P4": dict(pi_lr=3e-4, ent_coef=0.01),
    "P5": dict(pi_lr=1e-3, ent_coef=0.0),
    "P6": dict(pi_lr=1e-3, ent_coef=0.01),
}

LAST_K = 10  # epochs averaged for the final return/solved figures


def _run_dir(out_root: Path, cid: str, seed: int) -> Path:
    return out_root / f"ppo_{cid}" / f"seed_{seed}"


def train_grid(configs: list[str], seeds: list[int], record_every: int,
               epochs: int | None, out_root: Path, skip_existing: bool) -> None:
    """Train every (config, seed). Imports torch lazily so the rest of this
    script (grids, summary) works in environments without PyTorch."""
    from src.agents.ppo import PPOConfig, train_ppo  # noqa: E402  (lazy/local)

    base = yaml.safe_load(open(ROOT / "configs" / "ppo.yaml"))
    for cid in configs:
        for seed in seeds:
            out = _run_dir(out_root, cid, seed)
            if skip_existing and (out / "metrics.json").exists():
                print(f"[skip] {cid} seed {seed} already has metrics.json")
                continue
            cfg_dict = dict(base)
            cfg_dict.update(GRID[cid])
            cfg_dict["record_every"] = record_every
            if epochs is not None:
                cfg_dict["epochs"] = epochs
            cfg = PPOConfig(**cfg_dict)
            print(f"\n=== Training {cid} (lr={cfg.pi_lr:g}, ent={cfg.ent_coef:g}) "
                  f"seed {seed} -> {out} ===")
            train_ppo(seed=seed, cfg=cfg, log_dir=out)


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration; '—' for missing values."""
    if seconds != seconds:  # NaN
        return "—"
    if seconds >= 3600:
        h = int(seconds // 3600)
        mm = int((seconds % 3600) // 60)
        return f"{h}h {mm:02d}m"
    return f"{seconds / 60:.1f} min"


def _seed_final(metrics_path: Path) -> tuple[float, float, float, float]:
    """Return (mean last-K return, solved rate, cost, wall_time_sec) for one seed."""
    d = json.load(open(metrics_path))
    er = d.get("epoch_returns", [])
    sr = d.get("epoch_solved_rates", [])
    ec = d.get("epoch_costs", [])  # present once PPO logs passive cost
    r = stats.mean(er[-LAST_K:]) if er else float("nan")
    s = stats.mean(sr[-LAST_K:]) if sr else float("nan")
    c = stats.mean(ec[-LAST_K:]) if ec else float("nan")
    t = float(d.get("wall_time_sec", float("nan")))
    return r, s, c, t


def summarize(configs: list[str], seeds: list[int], out_root: Path) -> str:
    """Build the P1-P6 selection table as markdown and return it."""
    rows = []
    for cid in configs:
        rets, sols, costs, times = [], [], [], []
        for seed in seeds:
            mp = _run_dir(out_root, cid, seed) / "metrics.json"
            if mp.exists():
                r, s, c, t = _seed_final(mp)
                rets.append(r)
                sols.append(s)
                costs.append(c)
                times.append(t)
        if not rets:
            rows.append((cid, None))
            continue
        mean_r = stats.mean(rets)
        std_r = stats.pstdev(rets) if len(rets) > 1 else 0.0
        mean_s = stats.mean(sols)
        mean_c = stats.mean(costs)
        valid_t = [t for t in times if t == t]  # drop NaN
        mean_t = stats.mean(valid_t) if valid_t else float("nan")
        rows.append((cid, dict(mean_r=mean_r, std_r=std_r, mean_s=mean_s,
                               mean_c=mean_c, score=mean_r - std_r,
                               mean_t=mean_t, sum_t=sum(valid_t), n=len(rets))))

    # Eligible = actually solves (mean solved-rate >= 0.5); rank by score.
    eligible = [(cid, m) for cid, m in rows if m and m["mean_s"] >= 0.5]
    eligible.sort(key=lambda x: x[1]["score"], reverse=True)
    top3 = {cid for cid, _ in eligible[:3]}

    lines = [
        "# Stage 1 — Base PPO selection (auto-generated)",
        "",
        f"Final figures = mean over the last {LAST_K} epochs, across "
        f"{len(seeds)} seeds. Score = mean_return - std_return. "
        "Top-3 = highest score among configs that solve (mean solved >= 0.5). "
        "Avg cost is logged passively (PPO does not optimize it) — use it as a "
        "tie-breaker: among similar returns, prefer the lower-cost base.",
        "",
        "| ID | lr | ent_coef | Return (mean±std) | Avg cost | Solved | Score | Time/run | Top-3 |",
        "|----|-----|----------|-------------------|----------|--------|-------|----------|-------|",
    ]
    for cid, m in rows:
        lr = GRID[cid]["pi_lr"]
        ent = GRID[cid]["ent_coef"]
        if m is None:
            lines.append(f"| {cid} | {lr:g} | {ent:g} | (no results) | — | — | — | — | |")
            continue
        mark = "✅" if cid in top3 else ""
        lines.append(
            f"| {cid} | {lr:g} | {ent:g} | {m['mean_r']:.1f} ± {m['std_r']:.1f} "
            f"| {m['mean_c']:.1f} | {m['mean_s']:.0%} | {m['score']:.1f} "
            f"| {_fmt_dur(m['mean_t'])} | {mark} |"
        )

    total_sec = sum(m["sum_t"] for _, m in rows if m and m["sum_t"] == m["sum_t"])
    n_runs = sum(m["n"] for _, m in rows if m)
    if n_runs:
        lines += ["", f"**Total training time:** {_fmt_dur(total_sec)} "
                  f"across {n_runs} runs."]
    if top3:
        ordered = [cid for cid, _ in eligible[:3]]
        lines += ["", f"**Selected base policies:** "
                  + ", ".join(f"B{i+1}={cid}" for i, cid in enumerate(ordered))
                  + "  (carry these into Stage 2)."]
    else:
        lines += ["", "**No config solves yet** — see Stage 1 unfreeze rules in PLAN.md."]
    return "\n".join(lines) + "\n"


def select_top3(configs: list[str], seeds: list[int], out_root: Path) -> list[str]:
    """Return up to 3 config ids that solve (mean solved >= 0.5), ranked by
    score = mean_return - std_return. Shared with the Stage 2 runner so both
    use one consistent base-selection rule."""
    scored = []
    for cid in configs:
        rets, sols = [], []
        for seed in seeds:
            mp = _run_dir(out_root, cid, seed) / "metrics.json"
            if mp.exists():
                r, s, _, _ = _seed_final(mp)
                rets.append(r)
                sols.append(s)
        if not rets:
            continue
        mean_r = stats.mean(rets)
        std_r = stats.pstdev(rets) if len(rets) > 1 else 0.0
        mean_s = stats.mean(sols)
        if mean_s >= 0.5:
            scored.append((cid, mean_r - std_r))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:3]]


def build_videos(configs: list[str], seeds: list[int], out_root: Path,
                 videos_dir: Path, scale: float, cols: int) -> None:
    videos_dir.mkdir(parents=True, exist_ok=True)
    # per-combo: seeds side by side
    for cid in configs:
        files = [str(_run_dir(out_root, cid, s) / "training.mp4") for s in seeds]
        files = [f for f in files if Path(f).exists()]
        if files:
            build_grid(files, str(videos_dir / f"stage1_{cid}_seeds.mp4"),
                       cols=len(files), scale=scale)
    # master: one cell per combo (use the first available seed)
    master = []
    for cid in configs:
        for s in seeds:
            f = _run_dir(out_root, cid, s) / "training.mp4"
            if f.exists():
                master.append(str(f))
                break
    if master:
        build_grid(master, str(videos_dir / "stage1_all_configs.mp4"),
                   cols=cols, scale=scale)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--configs", type=str, nargs="+", default=list(GRID),
                   help="subset of P1..P6 (default all)")
    p.add_argument("--record-every", type=int, default=10,
                   help="snapshot a training-process video every N epochs (0=off)")
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--skip-train", action="store_true",
                   help="only (re)build grids + summary from existing results")
    p.add_argument("--skip-existing", action="store_true",
                   help="during training, skip runs that already have metrics.json")
    p.add_argument("--no-grid", action="store_true", help="don't build videos")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--out-root", type=str, default=str(ROOT / "results"))
    p.add_argument("--videos-dir", type=str, default=str(ROOT / "videos"))
    args = p.parse_args()

    bad = [c for c in args.configs if c not in GRID]
    if bad:
        sys.exit(f"Unknown configs {bad}. Choose from {list(GRID)}.")
    out_root = Path(args.out_root)

    if not args.skip_train:
        train_grid(args.configs, args.seeds, args.record_every, args.epochs,
                   out_root, args.skip_existing)

    if not args.no_grid:
        build_videos(args.configs, args.seeds, out_root,
                     Path(args.videos_dir), args.scale, args.cols)

    table = summarize(args.configs, args.seeds, out_root)
    summary_path = out_root / "stage1_summary.md"
    summary_path.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
