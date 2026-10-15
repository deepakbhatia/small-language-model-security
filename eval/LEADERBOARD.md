# SEI Public Benchmark Leaderboard

Primary metrics: disposition macro-F1 (GUIDE), technique exact-ID F1, evidence span F1, severity ordinal MAE.

| Run | Disposition macro-F1 | Technique F1 | Evidence F1 | Severity MAE |
|-----|---------------------:|-------------:|------------:|-------------:|
| identity_golden | 1.000 | 1.000 | 1.000 | 0.000 |
| held-out-orpo (guided, n=49) | 0.490 | 0.412 | 0.280 | — |
| held-out-consolidate (guided, n=49) | 0.425 | 0.270 | 0.261 | — |

Submit by opening a PR with `eval/reports/<run>.json` from `scripts/eval_harness.py`.
