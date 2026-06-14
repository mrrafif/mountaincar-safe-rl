"""Run all algorithms across multiple seeds sequentially.

Usage:
    python -m scripts.run_all --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dqn import DQNConfig, train_dqn
from src.agents.ppo import PPOConfig, train_ppo
from src.agents.safe_ppo import SafePPOConfig, train_safe_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algos", type=str, nargs="+",
                        default=["dqn", "ppo", "safe_ppo"])
    parser.add_argument("--record-every", type=int, default=None,
                        help="snapshot training-process videos every N epochs (0=off)")
    args = parser.parse_args()

    dqn_cfg = DQNConfig(**yaml.safe_load(open(ROOT / "configs" / "dqn.yaml")))
    ppo_cfg = PPOConfig(**yaml.safe_load(open(ROOT / "configs" / "ppo.yaml")))
    safe_cfg = SafePPOConfig(**yaml.safe_load(open(ROOT / "configs" / "safe_ppo.yaml")))
    if args.record_every is not None:
        ppo_cfg.record_every = args.record_every
        safe_cfg.record_every = args.record_every

    for seed in args.seeds:
        if "dqn" in args.algos:
            train_dqn(seed, dqn_cfg, ROOT / "results" / "dqn" / f"seed_{seed}")
        if "ppo" in args.algos:
            train_ppo(seed, ppo_cfg, ROOT / "results" / "ppo" / f"seed_{seed}")
        if "safe_ppo" in args.algos:
            train_safe_ppo(seed, safe_cfg, ROOT / "results" / "safe_ppo" / f"seed_{seed}")


if __name__ == "__main__":
    main()
