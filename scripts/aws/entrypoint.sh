#!/usr/bin/env bash
# Runs on the EC2 GPU instance (invoked from user-data).
# Env:
#   SEI_MODE=smoke|full
#   SEI_REPO_URL=...
#   SEI_S3_BUCKET=...          # required for artifact sync
#   SEI_GUIDE_S3=s3://bucket/GUIDE_Train.csv   # full mode
#   SEI_ADAPTER_S3=s3://bucket/sei-adapter/
#   SEI_MODEL_S3=s3://bucket/models/Foundation-Sec-8B/  # optional cache
#   SEI_GUIDE_LIMIT=20000
#   SEI_SHARD_SIZE=2000
#   SEI_KEEP_ALIVE=0|1         # 1 = do not shutdown (debug)
#   HF_TOKEN=...               # optional
set -euo pipefail

exec > >(tee -a /var/log/sei-train.log) 2>&1
echo "[sei] start $(date -Is) mode=${SEI_MODE:-smoke}"

WORK=/opt/sei
REPO_URL="${SEI_REPO_URL:-https://github.com/deepakbhatia/small-language-model-security.git}"
MODE="${SEI_MODE:-smoke}"
KEEP_ALIVE="${SEI_KEEP_ALIVE:-0}"
BUCKET="${SEI_S3_BUCKET:-}"
ADAPTER_S3="${SEI_ADAPTER_S3:-}"
GUIDE_S3="${SEI_GUIDE_S3:-}"
MODEL_S3="${SEI_MODEL_S3:-}"
GUIDE_LIMIT="${SEI_GUIDE_LIMIT:-20000}"
SHARD_SIZE="${SEI_SHARD_SIZE:-2000}"

cleanup() {
  local code=$?
  echo "[sei] finished exit=$code $(date -Is)"
  if [[ -n "$BUCKET" ]]; then
    aws s3 cp /var/log/sei-train.log "s3://${BUCKET}/sei-logs/sei-train-$(date +%Y%m%d-%H%M%S).log" || true
  fi
  if [[ "$KEEP_ALIVE" != "1" ]]; then
    echo "[sei] shutting down"
    shutdown -h now
  else
    echo "[sei] KEEP_ALIVE=1 — leaving instance running"
  fi
  exit "$code"
}
trap cleanup EXIT

mkdir -p "$WORK"
cd "$WORK"

if [[ ! -d repo/.git ]]; then
  git clone "$REPO_URL" repo
fi
cd repo
git fetch --all || true
git reset --hard origin/main || git reset --hard HEAD

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[train]"
pip install awscli
export PYTHONPATH="${PWD}/src:${PWD}"
export HF_HUB_DISABLE_XET=1
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# GPU check
python - <<'PY'
import torch
print("[sei] cuda=", torch.cuda.is_available(), "device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "CUDA not available"
PY

# Optional: warm model cache from S3 (speeds smoke/full a lot)
if [[ -n "$MODEL_S3" ]]; then
  mkdir -p /opt/hf-cache
  export HF_HOME=/opt/hf-cache
  echo "[sei] syncing model cache from $MODEL_S3"
  aws s3 sync "$MODEL_S3" /opt/hf-cache/ || true
fi

if [[ "$MODE" == "smoke" ]]; then
  echo "[sei] SMOKE: seed data + 2 train steps + sync + shutdown"
  python scripts/compose_synthetic.py --out data/processed/seed
  mkdir -p data/processed/v1/shards
  # one tiny shard from seed train
  head -n 8 data/processed/seed/train.jsonl > data/processed/v1/shards/train_shard00.jsonl
  cp data/processed/seed/val.jsonl data/processed/v1/val.jsonl
  ADAPTER_ARG=()
  if [[ -n "$ADAPTER_S3" ]]; then
    ADAPTER_ARG=(--adapter-s3 "$ADAPTER_S3")
  fi
  python scripts/aws/train_shards.py \
    --shards-dir data/processed/v1/shards \
    --val-file data/processed/v1/val.jsonl \
    --max-shards 1 \
    --max-steps 2 \
    --max-length 256 \
    --batch-size 1 \
    --grad-accum 1 \
    --epochs 1 \
    --eval-strategy no \
    "${ADAPTER_ARG[@]}"
  echo "SMOKE_OK" | tee /tmp/sei-smoke-ok.txt
  if [[ -n "$BUCKET" ]]; then
    aws s3 cp /tmp/sei-smoke-ok.txt "s3://${BUCKET}/sei-logs/SMOKE_OK"
  fi

elif [[ "$MODE" == "full" ]]; then
  echo "[sei] FULL training"
  if [[ -z "$GUIDE_S3" ]]; then
    echo "SEI_GUIDE_S3 is required for full mode" >&2
    exit 2
  fi
  aws s3 cp "$GUIDE_S3" /tmp/GUIDE_Train.csv
  python scripts/compose_all.py \
    --out data/processed/v1 \
    --guide /tmp/GUIDE_Train.csv \
    --guide-limit "$GUIDE_LIMIT" \
    --guide-stage 1 \
    --val-ratio 0.2 \
    --val-monitor-size 200
  python scripts/shard_jsonl.py \
    --input data/processed/v1/train.jsonl \
    --out-dir data/processed/v1/shards \
    --shard-size "$SHARD_SIZE"
  ADAPTER_ARG=()
  if [[ -n "$ADAPTER_S3" ]]; then
    ADAPTER_ARG=(--adapter-s3 "$ADAPTER_S3")
  fi
  python scripts/aws/train_shards.py \
    --shards-dir data/processed/v1/shards \
    --val-file data/processed/v1/val.jsonl \
    --max-length 1024 \
    --batch-size 2 \
    --grad-accum 8 \
    --epochs 1 \
    --eval-steps 500 \
    "${ADAPTER_ARG[@]}"
else
  echo "Unknown SEI_MODE=$MODE (use smoke|full)" >&2
  exit 2
fi

echo "[sei] job complete successfully"
