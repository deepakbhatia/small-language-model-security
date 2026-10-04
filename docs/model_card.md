# SEI Model Card (template — fill metrics after training runs)

## Model Details
- **Name:** sei-8b-instruct (planned)
- **Base:** fdtn-ai/Foundation-Sec-8B (Llama-3.1-8B + Cisco cyber CPT)
- **Task:** Security Event Intelligence — structured SOC triage JSON
- **License:** Apache-2.0 on training code; base inherits Llama 3.1 Community + Cisco Apache deltas (see NOTICE)

## Intended Use
- On-prem SOC assist for disposition, ATT&CK technique suggestion, severity, grounded evidence, and recommended actions
- Not for autonomous containment without human approval

## Training Data
See [data_card.md](data_card.md). Commercial-safe allowlist enforced via `configs/license_allowlist.yaml`.

## Training Procedure
1. Curriculum QLoRA SFT stages 1–4 (`configs/sft_stage*.yaml`)
2. ORPO preference alignment (`configs/orpo.yaml`)
3. Merge adapters; optional GGUF export

Details: [training_mechanics.md](training_mechanics.md)

## Evaluation
Run `scripts/eval_harness.py`. Track public numbers in `eval/LEADERBOARD.md`.

| Split | Disposition macro-F1 | Technique exact F1 | Evidence F1 | Severity MAE |
|-------|---------------------:|-------------------:|------------:|-------------:|
| seed-val (identity) | TBD | TBD | TBD | TBD |
| GUIDE holdout | TBD | — | — | TBD |

## Ethical Considerations
- May amplify biases in synthetic or vendor-labeled triage data
- Never log raw prompts/outputs containing PII by default
- Technique IDs constrained; still verify evidence grounding in production

## Edge SKU
- `Qwen/Qwen2.5-1.5B-Instruct` via `configs/distill_1p5b.yaml` / `scripts/distill_edge.py`
