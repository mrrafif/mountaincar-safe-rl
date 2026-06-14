"""CLI entry: train PPO baseline with a given seed and config."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.ppo import PPOConfig, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "ppo.yaml"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--record-every", type=int, default=None,
                        help="snapshot a training-process video every N epochs (0=off)")
    args = parser.parse_args()

    cfg_dict = yaml.safe_load(open(args.config))
    cfg = PPOConfig(**cfg_dict)
    if args.record_every is not None:
        cfg.record_every = args.record_every

    out = args.out or str(ROOT / "results" / "ppo" / f"seed_{args.seed}")
    train_ppo(seed=args.seed, cfg=cfg, log_dir=out)


if __name__ == "__main__":
    main()
