#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/yuzhang_fei/code/event-traffic-prediction}"
CONDA_BIN="${CONDA_BIN:-/home/yuzhang_fei/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
GPU_ID="${GPU_ID:-1}"
MEMORY_LIMIT_MB="${MEMORY_LIMIT_MB:-5000}"
UTIL_LIMIT_PCT="${UTIL_LIMIT_PCT:-20}"
POLL_SECONDS="${POLL_SECONDS:-300}"
DATASET="${DATASET:-tky}"

RUN_DIR="$PROJECT_DIR/reproduction/ConFormer/model"
DATA_DIR="$PROJECT_DIR/reproduction/ConFormer/data/${DATASET^^}"
LOG_DIR="$PROJECT_DIR/reproduction/ConFormer/logs/server_logs"
RUN_LOG="$LOG_DIR/conformer_${DATASET}_official_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing dataset directory: $DATA_DIR" | tee -a "$RUN_LOG"
  exit 1
fi

echo "Waiting for GPU $GPU_ID: memory <= ${MEMORY_LIMIT_MB} MiB and util <= ${UTIL_LIMIT_PCT}%." | tee -a "$RUN_LOG"
while true; do
  read -r used util < <(
    nvidia-smi --id="$GPU_ID" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
  )

  echo "$(date '+%F %T') gpu=$GPU_ID used=${used}MiB util=${util}%" | tee -a "$RUN_LOG"
  if (( used <= MEMORY_LIMIT_MB && util <= UTIL_LIMIT_PCT )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

cd "$RUN_DIR"
echo "Starting ConFormer ${DATASET} official-style run." | tee -a "$RUN_LOG"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" \
  python train.py -d "$DATASET" -g 0 -m train 2>&1 | tee -a "$RUN_LOG"
