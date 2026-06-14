# Stage 1 — Base PPO selection (auto-generated)

Final figures = mean over the last 10 epochs, across 3 seeds. Score = mean_return - std_return. Top-3 = highest score among configs that solve (mean solved >= 0.5). Avg cost is logged passively (PPO does not optimize it) — use it as a tie-breaker: among similar returns, prefer the lower-cost base.

| ID | lr | ent_coef | Return (mean±std) | Avg cost | Solved | Score | Time/run | Top-3 |
|----|-----|----------|-------------------|----------|--------|-------|----------|-------|
| P1 | 0.0001 | 0 | -149.7 ± 7.3 | 0.0 | 0% | -157.1 | 4.6 min |  |
| P2 | 0.0001 | 0.01 | -145.8 ± 0.2 | 0.0 | 0% | -146.0 | 5.7 min |  |
| P3 | 0.0003 | 0 | -29.2 ± 92.8 | 24.2 | 65% | -122.0 | 4.4 min |  |
| P4 | 0.0003 | 0.01 | 47.5 ± 1.1 | 28.9 | 100% | 46.4 | 4.1 min | ✅ |
| P5 | 0.001 | 0 | 47.8 ± 1.4 | 15.2 | 100% | 46.4 | 4.0 min | ✅ |
| P6 | 0.001 | 0.01 | 47.8 ± 0.2 | 15.0 | 100% | 47.6 | 4.0 min | ✅ |

**Total training time:** 1h 20m across 18 runs.

**Selected base policies:** B1=P6, B2=P5, B3=P4  (carry these into Stage 2).
