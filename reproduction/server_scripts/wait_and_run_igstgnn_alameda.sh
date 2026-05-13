#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/yuzhang_fei/code/event-traffic-prediction-git}"
CONDA_BIN="${CONDA_BIN:-/home/yuzhang_fei/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
GPU_ID="${GPU_ID:-0}"
FREE_MEMORY_MIN_PCT="${FREE_MEMORY_MIN_PCT:-50}"
POLL_SECONDS="${POLL_SECONDS:-300}"

RUN_DIR="$PROJECT_DIR/reproduction/IGSTGNN"
LOG_DIR="$RUN_DIR/experiments/IGSTGNN/server_logs"
RUN_LOG="$LOG_DIR/igstgnn_alameda_official_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "Waiting for GPU $GPU_ID: free memory > ${FREE_MEMORY_MIN_PCT}%." | tee -a "$RUN_LOG"
while true; do
  read -r free total < <(
    nvidia-smi --id="$GPU_ID" \
      --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
  )
  free_pct=$(( free * 100 / total ))

  echo "$(date '+%F %T') gpu=$GPU_ID free=${free}MiB total=${total}MiB free_pct=${free_pct}%" | tee -a "$RUN_LOG"
  if (( free * 100 > total * FREE_MEMORY_MIN_PCT )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

cd "$RUN_DIR"
echo "Starting IGSTGNN Alameda official-style run." | tee -a "$RUN_LOG"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" \
  python experiments/IGSTGNN/main.py \
    --device cuda:0 \
    --dataset Alameda \
    --model_name igstgnn \
    --seed 2025 \
    --bs 48 \
    --incident \
    --use_sensor_info 2>&1 | tee -a "$RUN_LOG"
