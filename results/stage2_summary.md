# Stage 2 — Safe PPO selection (auto-generated)

Final figures = mean over the last 10 epochs across 3 seeds. Pass = avg cost <= 15 AND no collapse (>30% drop from peak, unrecovered in last 20 epochs). Winner = highest mean return, then smallest std.

| ID | base | lam_lr | log_lam0 | Avg cost | <=lim | Return (mean±std) | lambda final | Collapse | Pass | Time/run |
|----|------|--------|----------|----------|-------|-------------------|--------------|----------|------|----------|
| S1 | P6 | 0.02 | -1 | 19.2 | no | 46.0 ± 1.3 | 0.14 |  |  | 4.8 min |
| S2 | P6 | 0.02 | 0 | 18.9 | no | 29.6 ± 8.8 | 0.25 |  |  | 4.9 min |
| S3 | P6 | 0.03 | -1 | 16.0 | no | 47.9 ± 0.8 | 0.09 |  |  | 5.2 min |
| S4 | P6 | 0.03 | 0 | 14.7 | yes | 46.8 ± 0.1 | 0.19 |  | ✅ | 5.5 min |
| S5 | P6 | 0.05 | -1 | 22.1 | no | 45.6 ± 2.8 | 0.07 |  |  | 5.4 min |
| S6 🏆 | P6 | 0.05 | 0 | 15.0 | yes | 47.3 ± 0.3 | 0.09 |  | ✅ | 6.6 min |
| S7 | P5 | 0.02 | -1 | 17.1 | no | 46.9 ± 3.5 | 0.09 |  |  | 5.0 min |
| S8 | P5 | 0.02 | 0 | 2.7 | yes | -27.2 ± 52.5 | 0.18 | yes |  | 5.0 min |
| S9 | P5 | 0.03 | -1 | 17.7 | no | 47.8 ± 1.0 | 0.06 |  |  | 5.0 min |
| S10 | P5 | 0.03 | 0 | 11.6 | yes | 40.2 ± 8.3 | 0.14 |  | ✅ | 5.0 min |
| S11 | P5 | 0.05 | -1 | 17.4 | no | 47.5 ± 1.3 | 0.03 |  |  | 5.0 min |
| S12 | P5 | 0.05 | 0 | 16.3 | no | 48.1 ± 0.2 | 0.09 |  |  | 5.2 min |
| S13 | P4 | 0.02 | -1 | 32.1 | no | 39.4 ± 7.5 | 0.10 |  |  | 5.3 min |
| S14 | P4 | 0.02 | 0 | 33.9 | no | 32.8 ± 12.2 | 0.22 |  |  | 5.4 min |
| S15 | P4 | 0.03 | -1 | 31.6 | no | 36.5 ± 11.5 | 0.07 |  |  | 6.4 min |
| S16 | P4 | 0.03 | 0 | 29.2 | no | 35.4 ± 15.8 | 0.15 |  |  | 6.1 min |
| S17 | P4 | 0.05 | -1 | 32.6 | no | 38.4 ± 9.8 | 0.03 |  |  | 5.0 min |
| S18 | P4 | 0.05 | 0 | 33.5 | no | 37.3 ± 10.3 | 0.09 |  |  | 5.0 min |

**Total training time:** 4h 46m across 54 runs.

**Winner: S6** (base P6, lam_lr=0.05, log_lam0=0) — return 47.3 ± 0.3, cost 15.0. Re-run with 5 seeds before declaring final.
