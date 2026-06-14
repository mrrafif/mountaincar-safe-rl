# MountainCar Safe RL — DQN, PPO, Safe PPO

Reproducible study of three RL agents on `MountainCar-v0`, culminating in **Safe PPO**
(a dual-critic PPO-Lagrangian agent that respects a soft proportional speed constraint
expressed as a Constrained MDP).

## Project structure

```
mountaincar-safe-rl/
├── src/                          # core library (all training logic lives here)
│   ├── envs.py                   # ShapedMountainCar, ConstrainedMountainCar
│   ├── networks.py               # QNetwork, PPONetwork, SafePPONetwork (dual-critic)
│   ├── buffers.py                # DQNReplayBuffer, PPORolloutBuffer, SafePPORolloutBuffer
│   ├── recording.py              # training / rollout video recording helpers
│   ├── utils.py                  # seeding, JSON logging
│   └── agents/
│       ├── dqn.py                # DQNConfig, DQNAgent, train_dqn
│       ├── ppo.py                # PPOConfig, train_ppo
│       └── safe_ppo.py           # SafePPOConfig, train_safe_ppo — main contribution
├── scripts/                      # thin CLI entry points + analysis/video tooling
│   ├── run_dqn.py                # python -m scripts.run_dqn --seed 0
│   ├── run_ppo.py
│   ├── run_safe_ppo.py
│   ├── run_all.py                # python -m scripts.run_all --seeds 0 1 2
│   ├── run_stage1.py             # PPO hyperparameter grid P1–P6
│   ├── run_stage2.py             # Safe-PPO grid S1–S18 on Stage 1's top-3
│   ├── run_stage2_p4_300.py      # P4 300-epoch extension S19–S24
│   ├── analysis.py               # aggregates results/, regenerates figures 3–6
│   ├── make_grid_video.py        # tile training videos into a synced grid
│   └── make_comparison_video.py  # 3-panel DQN | PPO | Safe PPO playback
├── configs/
│   ├── dqn.yaml
│   ├── ppo.yaml
│   └── safe_ppo.yaml
├── notebooks/
│   ├── feasibility_probe.ipynb   # constraint feasibility probe
│   └── cost_vs_speed_cap.png     # feasibility probe figure
├── results/                      # {algo}/seed_{s}/metrics.json + summaries (checkpoints .pt are gitignored)
├── requirements.txt
└── README.md

# generated/untracked: .venv/, videos/*.mp4, results/**/*.pt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Linux/macOS
# or:    .venv\Scripts\activate                        # Windows

pip install -r requirements.txt
```

Tested on Python 3.14. CPU is fine — full sweep takes well under 2 hours.

## Reproduce the trained weights

Full training pipeline — regenerates every `metrics.json` and model checkpoint under
`results/`. CPU is fine; the full sweep takes well under 2 hours.

```bash
# 1. Core agents: DQN, PPO, Safe PPO  -> results/{dqn,ppo,safe_ppo}/seed_*/
python -m scripts.run_all --seeds 0 1 2

# 2. Stage 1 — PPO hyperparameter grid P1–P6  -> results/ppo_P1..P6/
python -m scripts.run_stage1 --seeds 0 1 2 --record-every 10

# 3. Stage 2 — Safe-PPO grid S1–S18 on Stage 1's top-3 bases  -> results/safe_S1..S18/
python -m scripts.run_stage2 --seeds 0 1 2 --record-every 10

# 4. P4 300-epoch extension S19–S24  -> results/safe_S19..S24/
python -m scripts.run_stage2_p4_300
```

Drop `--record-every 10` from steps 2–3 to skip the training videos.

Or run a single core algorithm:

```bash
python -m scripts.run_dqn      --seed 0
python -m scripts.run_ppo      --seed 0
python -m scripts.run_safe_ppo --seed 0
```

Each run writes `metrics.json` + checkpoint to `results/<algo>/seed_<s>/`.

## Regenerate figures from results

```bash
python scripts/analysis.py
```

Reads every `results/<algo>/seed_*/metrics.json`, computes mean±std bands across seeds,
and writes figures to `paper/figures/`.