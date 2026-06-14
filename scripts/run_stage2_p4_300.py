"""Metrics-only 300-epoch rerun of the P4 Stage-2 Safe-PPO block.

This extends the original Stage-2 P4 configurations:

    S13 -> S19   lam_lr=0.02, log_lambda_init=-1
    S14 -> S20   lam_lr=0.02, log_lambda_init=0
    S15 -> S21   lam_lr=0.03, log_lambda_init=-1
    S16 -> S22   lam_lr=0.03, log_lambda_init=0
    S17 -> S23   lam_lr=0.05, log_lambda_init=-1
    S18 -> S24   lam_lr=0.05, log_lambda_init=0

No videos are generated; only metrics, checkpoints, and per-seed logs are written.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics as stats
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.safe_ppo import SafePPOConfig, train_safe_ppo  # noqa: E402


P4_BASE = dict(pi_lr=3e-4, ent_coef=0.01)
EXTENDED_GRID: dict[str, dict[str, float | str]] = {
    "S19": dict(source="S13", lam_lr=0.02, log_lambda_init=-1.0),
    "S20": dict(source="S14", lam_lr=0.02, log_lambda_init=0.0),
    "S21": dict(source="S15", lam_lr=0.03, log_lambda_init=-1.0),
    "S22": dict(source="S16", lam_lr=0.03, log_lambda_init=0.0),
    "S23": dict(source="S17", lam_lr=0.05, log_lambda_init=-1.0),
    "S24": dict(source="S18", lam_lr=0.05, log_lambda_init=0.0),
}
LAST_K = 10
COLLAPSE_DROP = 0.30
RECOVER_WINDOW = 20


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _run_dir(out_root: Path, sid: str, seed: int) -> Path:
    return out_root / f"safe_{sid}" / f"seed_{seed}"


def _collapsed(returns: list[float]) -> bool:
    if not returns:
        return False
    peak = max(returns)
    if peak <= 0:
        return False
    drew_down = any(r < peak * (1 - COLLAPSE_DROP) for r in returns)
    recovered = any(r >= peak * 0.9 for r in returns[-RECOVER_WINDOW:])
    return drew_down and not recovered


def _fmt_dur(seconds: float) -> str:
    if seconds != seconds:
        return "-"
    if seconds >= 3600:
        h = int(seconds // 3600)
        mm = int((seconds % 3600) // 60)
        return f"{h}h {mm:02d}m"
    return f"{seconds / 60:.1f} min"


def train_one(sid: str, seed: int, epochs: int, out_root: Path, skip_existing: bool) -> None:
    out = _run_dir(out_root, sid, seed)
    metrics_path = out / "metrics.json"
    if skip_existing and metrics_path.exists():
        print(f"[skip] {sid} seed {seed} already has metrics.json")
        return

    spec = EXTENDED_GRID[sid]
    cfg_dict = dict(yaml.safe_load(open(ROOT / "configs" / "safe_ppo.yaml")))
    cfg_dict.update(P4_BASE)
    cfg_dict["lam_lr"] = float(spec["lam_lr"])
    cfg_dict["log_lambda_init"] = float(spec["log_lambda_init"])
    cfg_dict["epochs"] = epochs
    cfg_dict["record_every"] = 0
    cfg = SafePPOConfig(**cfg_dict)

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    print(
        f"\n=== Training {sid} ({spec['source']} extended): "
        f"P4 pi_lr={cfg.pi_lr:g}, ent={cfg.ent_coef:g}, "
        f"lam_lr={cfg.lam_lr:g}, log_lam0={cfg.log_lambda_init:g}, "
        f"epochs={cfg.epochs}, seed={seed} -> {out} ==="
    )
    with open(log_path, "w") as f, contextlib.redirect_stdout(_Tee(sys.stdout, f)):
        train_safe_ppo(seed=seed, cfg=cfg, log_dir=out)


def _seed_final(metrics_path: Path) -> dict:
    d = json.load(open(metrics_path))
    er = d.get("epoch_returns", [])
    ec = d.get("epoch_costs", [])
    el = d.get("epoch_lambdas", [])
    sr = d.get("epoch_solved_rates", [])
    return dict(
        ret=stats.mean(er[-LAST_K:]) if er else float("nan"),
        cost=stats.mean(ec[-LAST_K:]) if ec else float("nan"),
        lam=stats.mean(el[-LAST_K:]) if el else float("nan"),
        solved=stats.mean(sr[-LAST_K:]) if sr else float("nan"),
        collapsed=_collapsed(er),
        wall=float(d.get("wall_time_sec", float("nan"))),
    )


def summarize(combos: list[str], seeds: list[int], out_root: Path, cost_limit: float) -> str:
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
        solved = [p["solved"] for p in per]
        walls = [p["wall"] for p in per if p["wall"] == p["wall"]]
        mean_c = stats.mean(costs)
        any_collapse = any(p["collapsed"] for p in per)
        rows.append((sid, dict(
            mean_r=stats.mean(rets),
            std_r=stats.pstdev(rets) if len(rets) > 1 else 0.0,
            mean_c=mean_c,
            std_c=stats.pstdev(costs) if len(costs) > 1 else 0.0,
            lam=stats.mean(lams),
            solved=stats.mean(solved),
            collapse=any_collapse,
            passed=(mean_c <= cost_limit) and not any_collapse,
            n=len(per),
            mean_t=stats.mean(walls) if walls else float("nan"),
            sum_t=sum(walls),
        )))

    lines = [
        "# Stage 2 P4 extension — Safe PPO 300 epochs (auto-generated)",
        "",
        f"Final figures = mean over the last {LAST_K} epochs across {len(seeds)} seeds. "
        f"Pass = avg cost <= {cost_limit:g} AND no collapse.",
        "",
        "| ID | Source | base | lam_lr | log_lam0 | Avg cost | <=lim | Return (mean±std) | lambda final | solved | Collapse | Pass | Time/run |",
        "|----|--------|------|--------|----------|----------|-------|-------------------|--------------|--------|----------|------|----------|",
    ]
    for sid, m in rows:
        spec = EXTENDED_GRID[sid]
        if m is None:
            lines.append(
                f"| {sid} | {spec['source']} | P4 | {spec['lam_lr']:g} | "
                f"{spec['log_lambda_init']:g} | (no results) | | | | | | | - |"
            )
            continue
        within = "yes" if m["mean_c"] <= cost_limit else "no"
        coll = "yes" if m["collapse"] else ""
        passmark = "yes" if m["passed"] else ""
        lines.append(
            f"| {sid} | {spec['source']} | P4 | {spec['lam_lr']:g} | "
            f"{spec['log_lambda_init']:g} | {m['mean_c']:.1f} ± {m['std_c']:.1f} | "
            f"{within} | {m['mean_r']:.1f} ± {m['std_r']:.1f} | {m['lam']:.3f} | "
            f"{m['solved']:.2f} | {coll} | {passmark} | {_fmt_dur(m['mean_t'])} |"
        )

    total_sec = sum(m["sum_t"] for _, m in rows if m and m["sum_t"] == m["sum_t"])
    n_runs = sum(m["n"] for _, m in rows if m)
    if n_runs:
        lines += ["", f"**Total training time:** {_fmt_dur(total_sec)} across {n_runs} runs."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--combos", type=str, nargs="+", default=list(EXTENDED_GRID))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out-root", type=str, default=str(ROOT / "results"))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    bad = [c for c in args.combos if c not in EXTENDED_GRID]
    if bad:
        raise SystemExit(f"Unknown combos {bad}. Valid: {list(EXTENDED_GRID)}")

    out_root = Path(args.out_root)
    if not args.skip_train:
        for sid in args.combos:
            for seed in args.seeds:
                train_one(sid, seed, args.epochs, out_root, args.skip_existing)

    cost_limit = float(yaml.safe_load(open(ROOT / "configs" / "safe_ppo.yaml"))["cost_limit"])
    table = summarize(args.combos, args.seeds, out_root, cost_limit)
    summary_path = out_root / "stage2_p4_300_summary.md"
    summary_path.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
