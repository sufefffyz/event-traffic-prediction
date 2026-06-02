#!/bin/bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/yuzhang_fei/code/event-traffic-prediction-git}
CONDA_ROOT=${CONDA_ROOT:-/home/yuzhang_fei/miniconda3}
CONDA_ENV=${CONDA_ENV:-STGraph}
GPU_ID=${GPU_ID:-1}
EPOCHS=${TRAFFIDENT_NUM_EPOCHS:-100}
DATASET_NAME=${TRAFFIDENT_DATA_NAME:-TraffiDent_D5_2023Q1_OfficialAll}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-/data/yuzhang_fei/TraffiDent/official}
OFFICIAL_REPO="$OFFICIAL_ROOT/XTraffic"
LOG_DIR="$REPO_ROOT/reproduction/logs"
mkdir -p "$LOG_DIR" "$OFFICIAL_ROOT"

cd "$REPO_ROOT"
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
conda activate "$CONDA_ENV"

if [ ! -f "$OFFICIAL_REPO/process/traffic_incident_match.py" ]; then
  git clone --depth 1 https://github.com/XAITraffic/XTraffic.git "$OFFICIAL_REPO"
else
  git -C "$OFFICIAL_REPO" pull --ff-only
fi

gpu_free_ratio() {
  nvidia-smi --id="$GPU_ID" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); printf "%.0f", ($1 * 100 / $2)}'
}

python reproduction/TraffiDent/prepare_paper_area_basicts.py \
  --area D5 \
  --sensor-type all \
  --event-types all \
  --matching-mode official-script \
  --official-match-script "$OFFICIAL_REPO/process/traffic_incident_match.py" \
  --match-scope subset \
  --months 1,2,3 \
  --split-ratio 0.6,0.2,0.2 \
  --dataset-name "$DATASET_NAME" \
  --overwrite

while true; do
  FREE_RATIO="$(gpu_free_ratio)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${GPU_ID} free memory ratio: ${FREE_RATIO}%"
  if [ "$FREE_RATIO" -ge "${FREE_RATIO_THRESHOLD:-50}" ]; then
    break
  fi
  sleep 300
done

export TRAFFIDENT_NUM_EPOCHS="$EPOCHS"
export TRAFFIDENT_SEED="${TRAFFIDENT_SEED:-2023}"
export TRAFFIDENT_BASICTS_ROOT=/data/yuzhang_fei/TraffiDent/basicts
export TRAFFIDENT_DATA_NAME="$DATASET_NAME"
export TRAFFIDENT_NUM_NODES=565

python BasicTS/experiments/train.py \
  -c baselines/AGCRN/TraffiDent_D5.py \
  -g "$GPU_ID" \
  2>&1 | tee "$LOG_DIR/traffident_post_incident_agcrn_d5_official_all_${EPOCHS}ep_g${GPU_ID}.log"

python reproduction/analysis/traffident_post_incident_forecasting_table.py \
  --repo-root "$REPO_ROOT" \
  --dataset "$DATASET_NAME" \
  --models AGCRN \
  --output-dir reproduction/analysis/traffident_post_incident_table \
  2>&1 | tee "$LOG_DIR/traffident_post_incident_agcrn_d5_official_all_table.log"
