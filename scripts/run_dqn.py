"""CLI entry: train DQN with a given seed and config."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dqn import DQNConfig, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dqn.yaml"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg_dict = yaml.safe_load(open(args.config))
    cfg = DQNConfig(**cfg_dict)

    out = args.out or str(ROOT / "results" / "dqn" / f"seed_{args.seed}")
    train_dqn(seed=args.seed, cfg=cfg, log_dir=out)


if __name__ == "__main__":
    main()
