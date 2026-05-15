#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/yuzhang_fei/code/event-traffic-prediction-git}"
CONDA_BIN="${CONDA_BIN:-/home/yuzhang_fei/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
GPU_ID="${GPU_ID:-0}"
FREE_MEMORY_MIN_PCT="${FREE_MEMORY_MIN_PCT:-50}"
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
echo "Starting ConFormer ${DATASET} official-style run." | tee -a "$RUN_LOG"
case "${DATASET,,}" in
  ba|sd)
    echo "Protocol note: paper-aligned BA/SD horizon=12/12, interval=15min, split=6:2:2, US Accidents enabled, regulation disabled." | tee -a "$RUN_LOG"
    echo "AI-supplemented note: official repo does not ship BA/SD YAML; config fills missing settings and logs protocol_notes from ConFormer.yaml." | tee -a "$RUN_LOG"
    ;;
esac
CUDA_VISIBLE_DEVICES="$GPU_ID" "$CONDA_BIN" run -n "$CONDA_ENV" \
  python train.py -d "$DATASET" -g 0 -m train 2>&1 | tee -a "$RUN_LOG"
