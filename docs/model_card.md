# SEI Model Card (template — fill metrics after training runs)

## Model Details
- **Name:** sei-8b-instruct (planned merge of ship adapter)
- **Ship adapter:** `checkpoints/sei-orpo-disposition/adapter` (`s3://…/sei-adapter-orpo/`)
- **Base:** fdtn-ai/Foundation-Sec-8B (Llama-3.1-8B + Cisco cyber CPT)
- **Task:** Security Event Intelligence — structured SOC triage JSON
- **License:** Apache-2.0 on training code; base inherits Llama 3.1 Community + Cisco Apache deltas (see NOTICE)

## Intended Use
- On-prem SOC assist for disposition, ATT&CK technique suggestion, severity, grounded evidence, and recommended actions
- Not for autonomous containment without human approval

## Training Data
See [data_card.md](data_card.md). Commercial-safe allowlist enforced via `configs/license_allowlist.yaml`.

## Training Procedure
1. Curriculum QLoRA SFT (stage1 → attack → ood → disposition)
2. ORPO preference alignment (`configs/orpo_disposition.yaml`) — **ship checkpoint**
3. Merge adapters; optional GGUF export
4. ~~TP-replay consolidate SFT~~ — held-out regression; do not ship

Details: [training_mechanics.md](training_mechanics.md), [aws_train.md](aws_train.md)

## Evaluation
Run `scripts/eval_harness.py` / `scripts/eval_model.py`. Track public numbers in `eval/LEADERBOARD.md`.

| Split | Disposition macro-F1 | Technique exact F1 | Evidence F1 | Severity MAE | Notes |
|-------|---------------------:|-------------------:|------------:|-------------:|-------|
| held-out ORPO+guided (n=49) | 0.490 | 0.412 | 0.280 | — | **ship** |
| held-out consolidate+guided | 0.425 | 0.270 | 0.261 | — | regress; skip |
| identity_golden | 1.000 | 1.000 | 1.000 | 0.000 | stub/identity |

## Ethical Considerations
- May amplify biases in synthetic or vendor-labeled triage data
- Never log raw prompts/outputs containing PII by default
- Technique IDs constrained; still verify evidence grounding in production

## Edge SKU
- `Qwen/Qwen2.5-1.5B-Instruct` via `configs/distill_1p5b.yaml` / `scripts/distill_edge.py`
