# Security Event Intelligence SLM

Open-source **Security Event Intelligence** small language model for structured SOC triage:

```
Security telemetry → Incident classification → ATT&CK technique → Severity → Evidence → Recommended action
```

Not a chatbot. A schema-constrained information-extraction model that emits validated SEI JSON.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,demo]"

# Generate golden + synthetic seed data
python scripts/compose_synthetic.py --out data/processed/seed

# Verify examples
python scripts/verify_jsonl.py --input data/processed/seed/train.jsonl

# Eval harness on golden predictions (baseline / stub)
python scripts/eval_harness.py --gold data/golden --pred data/golden

# Local SIEM webhook demo (no GPU required for echo/stub mode)
uvicorn demos.webhook.app:app --reload --port 8080

# Run stage-1 adapter (after aws s3 sync → checkpoints/sei-sft-stage1/adapter)
python scripts/infer_sei.py
# Webhook with real weights:
#   SEI_STUB=0 uvicorn demos.webhook.app:app --port 8080
```

Training tutorial (read first): [`docs/training_mechanics.md`](docs/training_mechanics.md)

**Kaggle (free GPU):** see [`docs/kaggle.md`](docs/kaggle.md) and [`notebooks/kaggle_train_stage1.ipynb`](notebooks/kaggle_train_stage1.ipynb).

**AWS (programmatic smoke → full):** see [`docs/aws_train.md`](docs/aws_train.md) (`scripts/aws/launch.py`).

## Stack

| Layer | Choice |
|-------|--------|
| Base | [`fdtn-ai/Foundation-Sec-8B`](https://huggingface.co/fdtn-ai/Foundation-Sec-8B) |
| Train | QLoRA curriculum SFT → ORPO (`trl` + `peft`) |
| Output | [`schemas/sei_v1.json`](schemas/sei_v1.json) + [`grammars/sei_v1.gbnf`](grammars/sei_v1.gbnf) |
| Knowledge | MITRE ATT&CK v16.1 + D3FEND RAG |
| Edge | Distill / SFT `Qwen2.5-1.5B-Instruct` |

## Schema (v1)

See [`schemas/sei_v1.json`](schemas/sei_v1.json). Hard rules:

- Technique IDs from pinned ATT&CK version
- Evidence `value` must be a substring of input telemetry
- Max 5 recommended actions
- Explicit abstain via `needs_more_data` / empty lists

## Data

Composed from open corpora (GUIDE, Atomic/Sigma/OTRF, ESCU, D3FEND). License allowlist: [`configs/license_allowlist.yaml`](configs/license_allowlist.yaml).

```bash
python scripts/download_attack.py
python scripts/download_guide.py   # requires Kaggle credentials
python scripts/compose_all.py --out data/processed/v1
```

## Training

```bash
python scripts/train_sft.py --stage 1 --config configs/sft_stage1.yaml
# ... stages 2–4 ...
python scripts/train_orpo.py --config configs/orpo.yaml
python scripts/merge_adapters.py --adapter checkpoints/sei-orpo/adapter --out checkpoints/sei-8b-instruct
python scripts/export_gguf.py --model checkpoints/sei-8b-instruct --quant Q4_K_M
```

## License

Code: Apache-2.0. Model weights inherit Foundation-Sec / Llama-3.1 notices — see [`NOTICE`](NOTICE). Training data licenses are tracked per-row in `meta.license`.
