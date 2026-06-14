"""Stage 2 driver: the S1-S18 Safe-PPO grid on top of the Stage 1 top-3 bases.

Implements Stage 2 from PLAN.md Part 2: for each of the three selected base
policies (B1, B2, B3 from Stage 1), sweep ``lam_lr x log_lambda_init`` with
``cost_limit`` fixed. Each Safe-PPO run inherits the base's actor hyperparameters
(``pi_lr``, ``ent_coef``) from its Stage 1 config and adds the safety settings.

A "base" here means the lr/ent_coef setting of the chosen Stage 1 PPO config;
Safe PPO trains its dual-critic network from scratch with those actor settings.

Grid (per base):
    lam_lr           in {0.02, 0.03, 0.05}
    log_lambda_init  in {-1.0, 0.0}
    -> 6 settings x 3 bases = 18 combos (S1..S18)

Layout produced::

    results/safe_S1/seed_0/{metrics.json, safe_ppo_net.pt, training.mp4}
    ...
    videos/stage2_S1_seeds.mp4      # 3 seeds of S1 side by side
    videos/stage2_B1_grid.mp4       # S1..S6 (base B1), one cell each (seed_0)
    videos/stage2_B2_grid.mp4 / _B3_grid.mp4
    results/stage2_summary.md       # the filled S1-S18 pass/winner table

Bases are auto-selected from Stage 1 results (score = mean_return - std_return),
or pass them explicitly with --bases.

Examples
--------
Full run (local, needs PyTorch), record every 10 epochs::

    python -m scripts.run_stage2 --seeds 0 1 2 --record-every 10

Force the bases and run a subset::

    python -m scripts.run_stage2 --bases P3 P4 P5 --combos S1 S2 S7

Rebuild grids + summary only (no torch)::

    python -m scripts.run_stage2 --skip-train
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
from run_stage1 import GRID as STAGE1_GRID, select_top3  # noqa: E402

# Safety-axis sweep applied on top of each base (PLAN.md Stage 2 table).
SAFE_SETTINGS: list[tuple[float, float]] = [
    (0.02, -1.0), (0.02, 0.0),
    (0.03, -1.0), (0.03, 0.0),
    (0.05, -1.0), (0.05, 0.0),
]
PER_BASE = len(SAFE_SETTINGS)  # 6

LAST_K = 10          # epochs averaged for final figures
COLLAPSE_DROP = 0.30  # >30% drop from peak (PLAN definition)
RECOVER_WINDOW = 20   # must recover to >=90% peak within last N epochs


def build_registry(bases: list[str]) -> dict[str, dict]:
    """Map S1..S18 -> {base, lam_lr, log_lambda_init}."""
    reg: dict[str, dict] = {}
    n = 1
    for b in bases:
        for lam_lr, init in SAFE_SETTINGS:
            reg[f"S{n}"] = dict(base=b, lam_lr=lam_lr, log_lambda_init=init)
            n += 1
    return reg


def _run_dir(out_root: Path, sid: str, seed: int) -> Path:
    return out_root / f"safe_{sid}" / f"seed_{seed}"


def train_grid(reg: dict[str, dict], combos: list[str], seeds: list[int],
               record_every: int, epochs: int | None, cost_limit: float | None,
               out_root: Path, skip_existing: bool) -> None:
    """Train every (combo, seed). torch imported lazily (local-only)."""
    from src.agents.safe_ppo import SafePPOConfig, train_safe_ppo  # noqa: E402

    base_yaml = yaml.safe_load(open(ROOT / "configs" / "safe_ppo.yaml"))
    for sid in combos:
        spec = reg[sid]
        base_cid = spec["base"]
        base_hp = STAGE1_GRID[base_cid]  # inherit pi_lr + ent_coef
        for seed in seeds:
            out = _run_dir(out_root, sid, seed)
            if skip_existing and (out / "metrics.json").exists():
                print(f"[skip] {sid} seed {seed} already has metrics.json")
                continue
            cfg_dict = dict(base_yaml)
            cfg_dict["pi_lr"] = base_hp["pi_lr"]
            cfg_dict["ent_coef"] = base_hp["ent_coef"]
            cfg_dict["lam_lr"] = spec["lam_lr"]
            cfg_dict["log_lambda_init"] = spec["log_lambda_init"]
            cfg_dict["record_every"] = record_every
            if cost_limit is not None:
                cfg_dict["cost_limit"] = cost_limit
            if epochs is not None:
                cfg_dict["epochs"] = epochs
            cfg = SafePPOConfig(**cfg_dict)
            print(f"\n=== Training {sid} (base {base_cid}: lr={cfg.pi_lr:g}, "
                  f"ent={cfg.ent_coef:g} | lam_lr={cfg.lam_lr:g}, "
                  f"log_lam0={cfg.log_lambda_init:g}) seed {seed} -> {out} ===")
            train_safe_ppo(seed=seed, cfg=cfg, log_dir=out)


def _collapsed(returns: list[float]) -> bool:
    """PLAN collapse rule: return drops >30% from its running peak and does not
    recover to >=90% of that peak within the last RECOVER_WINDOW epochs."""
    if not returns:
        return False
    peak = max(returns)
    if peak <= 0:
        return False
    drew_down = any(r < peak * (1 - COLLAPSE_DROP) for r in returns)
    recovered = any(r >= peak * 0.9 for r in returns[-RECOVER_WINDOW:])
    return drew_down and not recovered


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration; '—' for missing values."""
    if seconds != seconds:  # NaN
        return "—"
    if seconds >= 3600:
        h = int(seconds // 3600)
        mm = int((seconds % 3600) // 60)
        return f"{h}h {mm:02d}m"
    return f"{seconds / 60:.1f} min"


def _seed_final(metrics_path: Path) -> dict:
    d = json.load(open(metrics_path))
    er = d.get("epoch_returns", [])
    ec = d.get("epoch_costs", [])
    el = d.get("epoch_lambdas", [])
    return dict(
        ret=stats.mean(er[-LAST_K:]) if er else float("nan"),
        cost=stats.mean(ec[-LAST_K:]) if ec else float("nan"),
        lam=stats.mean(el[-LAST_K:]) if el else float("nan"),
        collapsed=_collapsed(er),
        wall=float(d.get("wall_time_sec", float("nan"))),
    )


def summarize(reg: dict[str, dict], combos: list[str], seeds: list[int],
              out_root: Path, cost_limit: float) -> str:
    rows = []
    for sid in combos:
        per = []
        for seed in seeds:
            mp = _run_dir(out_root, sid, seed) / "metrics.json"
            if mp.exists():
                per.append(_seed_final(mp))
        if not per:
            rows.append((sid, None))
            continue
        rets = [p["ret"] for p in per]
        costs = [p["cost"] for p in per]
        lams = [p["lam"] for p in per]
        wtimes = [p["wall"] for p in per if p["wall"] == p["wall"]]
        any_collapse = any(p["collapsed"] for p in per)
        mean_r = stats.mean(rets)
        std_r = stats.pstdev(rets) if len(rets) > 1 else 0.0
        mean_c = stats.mean(costs)
        passed = (mean_c <= cost_limit) and not any_collapse
        rows.append((sid, dict(mean_r=mean_r, std_r=std_r, mean_c=mean_c,
                               lam=stats.mean(lams), collapse=any_collapse,
                               passed=passed, n=len(per),
                               mean_t=stats.mean(wtimes) if wtimes else float("nan"),
                               sum_t=sum(wtimes))))

    passing = [(sid, m) for sid, m in rows if m and m["passed"]]
    # winner = highest mean return, then smallest std
    passing.sort(key=lambda x: (-x[1]["mean_r"], x[1]["std_r"]))
    winner = passing[0][0] if passing else None

    lines = [
        "# Stage 2 — Safe PPO selection (auto-generated)",
        "",
        f"Final figures = mean over the last {LAST_K} epochs across "
        f"{len(seeds)} seeds. Pass = avg cost <= {cost_limit:g} AND no collapse "
        f"(>{int(COLLAPSE_DROP*100)}% drop from peak, unrecovered in last "
        f"{RECOVER_WINDOW} epochs). Winner = highest mean return, then smallest std.",
        "",
        "| ID | base | lam_lr | log_lam0 | Avg cost | <=lim | Return (mean±std) "
        "| lambda final | Collapse | Pass | Time/run |",
        "|----|------|--------|----------|----------|-------|-------------------"
        "|--------------|----------|------|----------|",
    ]
    for sid, m in rows:
        spec = reg[sid]
        if m is None:
            lines.append(
                f"| {sid} | {spec['base']} | {spec['lam_lr']:g} | "
                f"{spec['log_lambda_init']:g} | (no results) | | | | | | — |")
            continue
        within = "yes" if m["mean_c"] <= cost_limit else "no"
        passmark = "✅" if m["passed"] else ""
        coll = "yes" if m["collapse"] else ""
        win = " 🏆" if sid == winner else ""
        lines.append(
            f"| {sid}{win} | {spec['base']} | {spec['lam_lr']:g} | "
            f"{spec['log_lambda_init']:g} | {m['mean_c']:.1f} | {within} | "
            f"{m['mean_r']:.1f} ± {m['std_r']:.1f} | {m['lam']:.2f} | {coll} | {passmark} "
            f"| {_fmt_dur(m['mean_t'])} |")

    total_sec = sum(m["sum_t"] for _, m in rows if m and m["sum_t"] == m["sum_t"])
    n_runs = sum(m["n"] for _, m in rows if m)
    if n_runs:
        lines += ["", f"**Total training time:** {_fmt_dur(total_sec)} "
                  f"across {n_runs} runs."]

    if winner:
        wm = dict(passing)[winner]
        lines += ["", f"**Winner: {winner}** (base {reg[winner]['base']}, "
                  f"lam_lr={reg[winner]['lam_lr']:g}, "
                  f"log_lam0={reg[winner]['log_lambda_init']:g}) — "
                  f"return {wm['mean_r']:.1f} ± {wm['std_r']:.1f}, "
                  f"cost {wm['mean_c']:.1f}. Re-run with 5 seeds before declaring final."]
    else:
        lines += ["", "**No combo passes yet** (cost > limit or collapse on some seed). "
                  "See the 'If infeasible' relaxation steps in PLAN.md."]
    return "\n".join(lines) + "\n"


def build_videos(reg: dict[str, dict], bases: list[str], combos: list[str],
                 seeds: list[int], out_root: Path, videos_dir: Path,
                 scale: float) -> None:
    videos_dir.mkdir(parents=True, exist_ok=True)
    # per-combo: seeds side by side
    for sid in combos:
        files = [str(_run_dir(out_root, sid, s) / "training.mp4") for s in seeds]
        files = [f for f in files if Path(f).exists()]
        if files:
            build_grid(files, str(videos_dir / f"stage2_{sid}_seeds.mp4"),
                       cols=len(files), scale=scale)
    # per-base master: the 6 settings of that base, one cell each (first seed)
    for b in bases:
        sids = [sid for sid in combos if reg[sid]["base"] == b]
        cells = []
        for sid in sids:
            for s in seeds:
                f = _run_dir(out_root, sid, s) / "training.mp4"
                if f.exists():
                    cells.append(str(f))
                    break
        if cells:
            build_grid(cells, str(videos_dir / f"stage2_{b}_grid.mp4"),
                       cols=3, scale=scale)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bases", type=str, nargs="+", default=None,
                   help="Stage 1 config ids to use as B1 B2 B3 (default: auto top-3)")
    p.add_argument("--combos", type=str, nargs="+", default=None,
                   help="subset of S1..S18 (default all)")
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--cost-limit", type=float, default=None,
                   help="override cost_limit (default from configs/safe_ppo.yaml)")
    p.add_argument("--stage1-seeds", type=int, nargs="+", default=None,
                   help="seeds used in Stage 1 for auto base selection (default = --seeds)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--out-root", type=str, default=str(ROOT / "results"))
    p.add_argument("--videos-dir", type=str, default=str(ROOT / "videos"))
    args = p.parse_args()

    out_root = Path(args.out_root)

    # Resolve the three base configs (B1, B2, B3)
    if args.bases:
        bases = args.bases
    else:
        s1_seeds = args.stage1_seeds or args.seeds
        bases = select_top3(list(STAGE1_GRID), s1_seeds, out_root)
        if len(bases) < 3:
            sys.exit(
                f"Auto base-selection found only {len(bases)} solving Stage 1 "
                f"config(s): {bases}. Run Stage 1 first, or pass --bases explicitly.")
    bad = [b for b in bases if b not in STAGE1_GRID]
    if bad:
        sys.exit(f"Unknown bases {bad}. Choose from {list(STAGE1_GRID)}.")
    print(f"Bases: B1={bases[0]}, B2={bases[1]}, B3={bases[2]}")

    reg = build_registry(bases)
    cost_limit = (args.cost_limit if args.cost_limit is not None
                  else float(yaml.safe_load(open(ROOT / "configs" / "safe_ppo.yaml"))["cost_limit"]))
    combos = args.combos or list(reg)
    bad = [c for c in combos if c not in reg]
    if bad:
        sys.exit(f"Unknown combos {bad}. Valid: {list(reg)}.")

    if not args.skip_train:
        train_grid(reg, combos, args.seeds, args.record_every, args.epochs,
                   args.cost_limit, out_root, args.skip_existing)

    if not args.no_grid:
        build_videos(reg, bases, combos, args.seeds, out_root,
                     Path(args.videos_dir), args.scale)

    table = summarize(reg, combos, args.seeds, out_root, cost_limit)
    summary_path = out_root / "stage2_summary.md"
    summary_path.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
