# Air-gap GGUF packaging (ORPO ship path)

Ship from the **ORPO disposition** adapter only — not consolidate.

## 1. Sync champion adapter

```bash
aws s3 sync s3://$BUCKET/sei-adapter-orpo/ checkpoints/sei-orpo-disposition/adapter/
```

## 2. Merge LoRA → full HF weights

```bash
python scripts/merge_adapters.py \
  --adapter checkpoints/sei-orpo-disposition/adapter \
  --out checkpoints/sei-8b-instruct
```

Defaults are the same paths if you omit flags.

## 3. Export GGUF + bundle artifacts

Manifest + schema/grammar/ATT&CK IDs/NOTICE (no llama.cpp required yet):

```bash
python scripts/export_gguf.py --model checkpoints/sei-8b-instruct --quant Q4_K_M
```

Convert + quantize when llama.cpp is available:

```bash
python scripts/export_gguf.py --model checkpoints/sei-8b-instruct --quant Q4_K_M \
  --llama-cpp /path/to/llama.cpp
```

Contents under `checkpoints/gguf/`:

- `sei-8b-instruct-q4_k_m.gguf` (after quantize)
- `sei_v1.json`
- `sei_v1.gbnf` (or rebuild via `scripts/build_grammar.py`)
- `technique_ids.txt`
- `NOTICE`
- `airgap_manifest.txt`

## 4. Serve offline

```bash
llama-cli -m checkpoints/gguf/sei-8b-instruct-q4_k_m.gguf \
  --grammar-file checkpoints/gguf/sei_v1.gbnf -n 1024 -p "..."
```

Still run `verify_example` / evidence grounding on outputs — grammar enforces shape, not correctness.
