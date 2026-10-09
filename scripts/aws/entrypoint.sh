#!/usr/bin/env bash
# Runs on the EC2 GPU instance (invoked from user-data).
# Env:
#   SEI_MODE=smoke|full|stage1|stage2|attack
#   SEI_REPO_URL=...
#   SEI_S3_BUCKET=...          # required for artifact sync
#   SEI_GUIDE_S3=s3://bucket/GUIDE_Train.csv
#   SEI_ADAPTER_S3=s3://bucket/sei-adapter/          # write target
#   SEI_RESUME_ADAPTER_S3=s3://bucket/sei-adapter/   # warm-start (optional)
#   SEI_MODEL_S3=s3://bucket/models/Foundation-Sec-8B/  # optional cache
#   SEI_CORPORA_S3=s3://bucket/corpora/              # optional pre-downloaded atomic/sigma
#   SEI_GUIDE_LIMIT=50000
#   SEI_GUIDE_OFFSET=20000     # stage1 continue: skip already-trained rows
#   SEI_GUIDE_STAGE=1|2
#   SEI_ATOMIC_LIMIT=8000
#   SEI_SIGMA_LIMIT=8000
#   SEI_SHARD_SIZE=2000
#   SEI_KEEP_ALIVE=0|1
#   HF_TOKEN=...
set -euo pipefail

exec > >(tee -a /var/log/sei-train.log) 2>&1
echo "[sei] start $(date -Is) mode=${SEI_MODE:-smoke}"

WORK=/opt/sei
REPO_URL="${SEI_REPO_URL:-https://github.com/deepakbhatia/small-language-model-security.git}"
MODE="${SEI_MODE:-smoke}"
KEEP_ALIVE="${SEI_KEEP_ALIVE:-0}"
BUCKET="${SEI_S3_BUCKET:-}"
ADAPTER_S3="${SEI_ADAPTER_S3:-}"
RESUME_ADAPTER_S3="${SEI_RESUME_ADAPTER_S3:-}"
GUIDE_S3="${SEI_GUIDE_S3:-}"
MODEL_S3="${SEI_MODEL_S3:-}"
CORPORA_S3="${SEI_CORPORA_S3:-}"
GUIDE_LIMIT="${SEI_GUIDE_LIMIT:-20000}"
GUIDE_OFFSET="${SEI_GUIDE_OFFSET:-0}"
GUIDE_STAGE="${SEI_GUIDE_STAGE:-1}"
ATOMIC_LIMIT="${SEI_ATOMIC_LIMIT:-8000}"
SIGMA_LIMIT="${SEI_SIGMA_LIMIT:-8000}"
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

# DLAMI (Ubuntu) often ships conda+torch; system python3 may lack ensurepip/venv.
activate_python() {
  if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    for env in pytorch pytorch_p312 pytorch_p311 base; do
      if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$env"; then
        echo "[sei] conda activate $env"
        conda activate "$env"
        break
      fi
    done
    if python -c "import torch" 2>/dev/null; then
      return 0
    fi
  fi

  echo "[sei] conda/torch not ready — installing python3-venv and creating .venv"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  apt-get install -y "python${PY_VER}-venv" python3-pip || apt-get install -y python3-venv python3-pip
  python3 -m venv --system-site-packages .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

activate_python
pip install -U pip wheel
pip install -e ".[train]"
pip install awscli
export PYTHONPATH="${PWD}/src:${PWD}"
export HF_HUB_DISABLE_XET=1
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

python - <<'PY'
import torch
print("[sei] cuda=", torch.cuda.is_available(), "device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "CUDA not available"
PY

if [[ -n "$MODEL_S3" ]]; then
  mkdir -p /opt/hf-cache
  export HF_HOME=/opt/hf-cache
  echo "[sei] syncing model cache from $MODEL_S3"
  aws s3 sync "$MODEL_S3" /opt/hf-cache/ || true
fi

pull_resume_adapter() {
  local dest="$1"
  if [[ -z "$RESUME_ADAPTER_S3" ]]; then
    return 0
  fi
  echo "[sei] pulling resume adapter from $RESUME_ADAPTER_S3 → $dest"
  mkdir -p "$dest"
  aws s3 sync "$RESUME_ADAPTER_S3" "$dest/"
  if [[ ! -f "$dest/adapter_model.safetensors" && ! -f "$dest/adapter_model.bin" ]]; then
    echo "[sei] ERROR: resume adapter missing weights under $dest" >&2
    ls -la "$dest" || true
    exit 3
  fi
}

run_compose_and_shard() {
  local out_dir="$1"
  local guide_stage="$2"
  local seed_stages="$3"
  mkdir -p "$out_dir"
  python scripts/compose_all.py \
    --out "$out_dir" \
    --guide /tmp/GUIDE_Train.csv \
    --guide-limit "$GUIDE_LIMIT" \
    --guide-offset "$GUIDE_OFFSET" \
    --guide-stage "$guide_stage" \
    --seed-stages "$seed_stages" \
    --val-ratio 0.2 \
    --val-monitor-size 200
  python scripts/shard_jsonl.py \
    --input "$out_dir/train.jsonl" \
    --out-dir "$out_dir/shards" \
    --shard-size "$SHARD_SIZE"
}

require_guide() {
  if [[ -z "$GUIDE_S3" ]]; then
    echo "SEI_GUIDE_S3 is required for mode=$MODE" >&2
    exit 2
  fi
  aws s3 cp "$GUIDE_S3" /tmp/GUIDE_Train.csv
}

if [[ "$MODE" == "smoke" ]]; then
  echo "[sei] SMOKE: seed data + 2 train steps + sync + shutdown"
  python scripts/compose_synthetic.py --out data/processed/seed
  mkdir -p data/processed/v1/shards
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

elif [[ "$MODE" == "full" || "$MODE" == "stage1" ]]; then
  # stage1: continue GUIDE (default offset/limit set by launch.py for +50k)
  # full: backward-compatible alias (from-scratch if no resume adapter)
  echo "[sei] STAGE1 training (mode=$MODE offset=$GUIDE_OFFSET limit=$GUIDE_LIMIT)"
  require_guide
  STAGE1_ADAPTER="checkpoints/sei-sft-stage1/adapter"
  pull_resume_adapter "$STAGE1_ADAPTER"
  run_compose_and_shard "data/processed/v1" "${GUIDE_STAGE:-1}" "1"
  ADAPTER_ARG=()
  RESUME_ARG=()
  if [[ -n "$ADAPTER_S3" ]]; then
    ADAPTER_ARG=(--adapter-s3 "$ADAPTER_S3")
  fi
  if [[ -d "$STAGE1_ADAPTER" ]]; then
    RESUME_ARG=(--resume-adapter "$STAGE1_ADAPTER")
  fi
  python scripts/aws/train_shards.py \
    --base-config configs/sft_stage1.yaml \
    --run-config configs/sft_stage1_aws.yaml \
    --output-dir checkpoints/sei-sft-stage1 \
    --shards-dir data/processed/v1/shards \
    --val-file data/processed/v1/val.jsonl \
    --max-length 1024 \
    --batch-size 2 \
    --grad-accum 8 \
    --epochs 1 \
    --eval-steps 500 \
    "${RESUME_ARG[@]}" \
    "${ADAPTER_ARG[@]}"
  if [[ -n "$BUCKET" ]]; then
    echo "STAGE1_OK" | aws s3 cp - "s3://${BUCKET}/sei-logs/STAGE1_OK"
  fi

elif [[ "$MODE" == "stage2" ]]; then
  echo "[sei] STAGE2 training from stage-1 adapter (GUIDE stage=2 + synthetic 1,2)"
  require_guide
  STAGE1_ADAPTER="checkpoints/sei-sft-stage1/adapter"
  if [[ -z "$RESUME_ADAPTER_S3" && -n "$ADAPTER_S3" ]]; then
    echo "[sei] SEI_RESUME_ADAPTER_S3 unset — set it to stage-1 adapter prefix"
  fi
  pull_resume_adapter "$STAGE1_ADAPTER"
  if [[ ! -d "$STAGE1_ADAPTER" ]]; then
    echo "[sei] ERROR: stage2 requires stage-1 adapter at $STAGE1_ADAPTER (set SEI_RESUME_ADAPTER_S3)" >&2
    exit 3
  fi
  run_compose_and_shard "data/processed/stage2" "2" "1,2"
  ADAPTER_ARG=()
  if [[ -n "$ADAPTER_S3" ]]; then
    ADAPTER_ARG=(--adapter-s3 "$ADAPTER_S3")
  fi
  python scripts/aws/train_shards.py \
    --base-config configs/sft_stage2.yaml \
    --run-config configs/sft_stage2_aws.yaml \
    --output-dir checkpoints/sei-sft-stage2 \
    --shards-dir data/processed/stage2/shards \
    --val-file data/processed/stage2/val.jsonl \
    --resume-adapter "$STAGE1_ADAPTER" \
    --max-length 1024 \
    --batch-size 2 \
    --grad-accum 8 \
    --epochs 1 \
    --eval-steps 500 \
    --lr 5.0e-5 \
    "${ADAPTER_ARG[@]}"
  if [[ -n "$BUCKET" ]]; then
    echo "STAGE2_OK" | aws s3 cp - "s3://${BUCKET}/sei-logs/STAGE2_OK"
  fi

elif [[ "$MODE" == "attack" ]]; then
  echo "[sei] ATTACK mix training (Atomic + Sigma + OTRF), resume from prior adapter"
  if [[ -z "$RESUME_ADAPTER_S3" ]]; then
    echo "[sei] ERROR: attack mode requires SEI_RESUME_ADAPTER_S3 (stage2 adapter recommended)" >&2
    exit 3
  fi
  RESUME_DEST="checkpoints/sei-resume/adapter"
  pull_resume_adapter "$RESUME_DEST"

  if [[ -n "$CORPORA_S3" ]]; then
    echo "[sei] syncing corpora from $CORPORA_S3"
    mkdir -p data/raw
    aws s3 sync "$CORPORA_S3" data/raw/
  fi
  if [[ ! -d data/raw/atomic/atomics ]] || [[ ! -d data/raw/sigma/rules ]]; then
    echo "[sei] downloading Atomic + Sigma corpora"
    python scripts/download_attack_corpora.py --out-root data/raw
  fi

  GUIDE_ARGS=()
  if [[ -n "$GUIDE_S3" ]]; then
    aws s3 cp "$GUIDE_S3" /tmp/GUIDE_Train.csv
    # small GUIDE stage-2 mix so disposition doesn't collapse; techniques come from attack corpora
    GUIDE_ARGS=(--guide /tmp/GUIDE_Train.csv --guide-limit "${GUIDE_LIMIT:-5000}" --guide-offset 0 --guide-stage 2)
  fi

  mkdir -p data/processed/attack
  python scripts/compose_all.py \
    --out data/processed/attack \
    --attack-mix \
    --otrf data/otrf/scenarios.yaml \
    --atomic-limit "$ATOMIC_LIMIT" \
    --sigma-limit "$SIGMA_LIMIT" \
    --seed-stages 2,3,4 \
    --attack-stages 2,3,4 \
    --val-ratio 0.15 \
    --val-monitor-size 200 \
    "${GUIDE_ARGS[@]}"
  python scripts/shard_jsonl.py \
    --input data/processed/attack/train.jsonl \
    --out-dir data/processed/attack/shards \
    --shard-size "$SHARD_SIZE"

  ADAPTER_ARG=()
  if [[ -n "$ADAPTER_S3" ]]; then
    ADAPTER_ARG=(--adapter-s3 "$ADAPTER_S3")
  fi
  python scripts/aws/train_shards.py \
    --base-config configs/sft_attack.yaml \
    --run-config configs/sft_attack_aws.yaml \
    --output-dir checkpoints/sei-sft-attack \
    --shards-dir data/processed/attack/shards \
    --val-file data/processed/attack/val.jsonl \
    --resume-adapter "$RESUME_DEST" \
    --max-length 1024 \
    --batch-size 2 \
    --grad-accum 8 \
    --epochs 1 \
    --eval-steps 500 \
    --lr 5.0e-5 \
    "${ADAPTER_ARG[@]}"
  if [[ -n "$BUCKET" ]]; then
    echo "ATTACK_OK" | aws s3 cp - "s3://${BUCKET}/sei-logs/ATTACK_OK"
  fi

else
  echo "Unknown SEI_MODE=$MODE (use smoke|full|stage1|stage2|attack)" >&2
  exit 2
fi

echo "[sei] job complete successfully"
