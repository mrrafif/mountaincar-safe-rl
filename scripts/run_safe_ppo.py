"""CLI entry: train Safe PPO (CMDP / PPO-Lagrangian) with seed + config."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.safe_ppo import SafePPOConfig, train_safe_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "safe_ppo.yaml"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--record-every", type=int, default=None,
                        help="snapshot a training-process video every N epochs (0=off)")
    args = parser.parse_args()

    cfg_dict = yaml.safe_load(open(args.config))
    cfg = SafePPOConfig(**cfg_dict)
    if args.record_every is not None:
        cfg.record_every = args.record_every

    out = args.out or str(ROOT / "results" / "safe_ppo" / f"seed_{args.seed}")
    train_safe_ppo(seed=args.seed, cfg=cfg, log_dir=out)


if __name__ == "__main__":
    main()
