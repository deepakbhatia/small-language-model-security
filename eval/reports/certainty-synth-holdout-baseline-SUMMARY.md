# Baseline before certainty ORPO (ship adapter)

Adapter: `checkpoints/sei-orpo-disposition/adapter`  
Gold: `data/processed/certainty_synth/merged_sft_holdout.jsonl` (n=155)  
Report: `eval/reports/certainty-synth-holdout-baseline.json`

| Metric | Value |
|--------|------:|
| Disposition macro-F1 | 0.187 |
| Needs-more-data F1 (p/r) | 0.534 (0.37 / **0.98**) |
| True-positive F1 | 0.108 |
| Benign-true-positive F1 | 0.000 |
| False-positive F1 | 0.105 |
| Technique exact-ID F1 | 0.247 |
| JSON valid rate | 0.187 |
| Verifier-clean rate | 0.000 |

Sealed ship reference (`held-out-guided`, n=49): disposition macro **0.490**, NMD F1 **0.133**, verifier-clean **0.84**.

**Read:** On synth contrast holdout the ship model **over-abstains** (almost always needs_more_data) and often fails schema/verify. Training target: improve calibrated NMD vs TP/BTP/FP **without** collapsing sealed held-out.
