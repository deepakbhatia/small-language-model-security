#!/usr/bin/env bash
# Sync ORPO champion → merge → GGUF manifest bundle.
# Usage:
#   export BUCKET=sei-slm-...
#   ./scripts/ship_orpo_bundle.sh
# Optional: LLAMA_CPP=/path/to/llama.cpp ./scripts/ship_orpo_bundle.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ADAPTER_LOCAL="${ADAPTER_LOCAL:-checkpoints/sei-orpo-disposition/adapter}"
MERGED="${MERGED:-checkpoints/sei-8b-instruct}"
QUANT="${QUANT:-Q4_K_M}"

if [[ -n "${BUCKET:-}" ]]; then
  echo "sync s3://$BUCKET/sei-adapter-orpo/ → $ADAPTER_LOCAL"
  aws s3 sync "s3://${BUCKET}/sei-adapter-orpo/" "$ADAPTER_LOCAL/"
elif [[ ! -d "$ADAPTER_LOCAL" ]]; then
  echo "Set BUCKET=… or place adapter at $ADAPTER_LOCAL" >&2
  exit 1
fi

python scripts/merge_adapters.py --adapter "$ADAPTER_LOCAL" --out "$MERGED"
if [[ -n "${LLAMA_CPP:-}" ]]; then
  python scripts/export_gguf.py --model "$MERGED" --quant "$QUANT" --llama-cpp "$LLAMA_CPP"
else
  python scripts/export_gguf.py --model "$MERGED" --quant "$QUANT"
fi
echo "done → checkpoints/gguf/ (see airgap_manifest.txt)"
