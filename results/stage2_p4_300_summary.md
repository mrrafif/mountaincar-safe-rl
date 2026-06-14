# Stage 2 P4 extension — Safe PPO 300 epochs (auto-generated)

Final figures = mean over the last 10 epochs across 3 seeds. Pass = avg cost <= 15 AND no collapse.

| ID | Source | base | lam_lr | log_lam0 | Avg cost | <=lim | Return (mean±std) | lambda final | solved | Collapse | Pass | Time/run |
|----|--------|------|--------|----------|----------|-------|-------------------|--------------|--------|----------|------|----------|
| S19 | S13 | P4 | 0.02 | -1 | 13.0 ± 4.1 | yes | 48.9 ± 0.3 | 0.071 | 1.00 |  | yes | 5.7 min |
| S20 | S14 | P4 | 0.02 | 0 | 10.8 ± 3.7 | yes | 48.3 ± 1.7 | 0.298 | 1.00 |  | yes | 5.7 min |
| S21 | S15 | P4 | 0.03 | -1 | 11.2 ± 3.1 | yes | 49.2 ± 0.7 | 0.041 | 1.00 |  | yes | 5.6 min |
| S22 | S16 | P4 | 0.03 | 0 | 15.6 ± 7.5 | no | 48.4 ± 0.8 | 0.155 | 1.00 |  |  | 5.5 min |
| S23 | S17 | P4 | 0.05 | -1 | 12.4 ± 4.5 | yes | 48.5 ± 0.7 | 0.025 | 1.00 |  | yes | 5.4 min |
| S24 | S18 | P4 | 0.05 | 0 | 13.1 ± 0.3 | yes | 48.1 ± 1.1 | 0.049 | 1.00 |  | yes | 5.4 min |

**Total training time:** 1h 40m across 18 runs.
