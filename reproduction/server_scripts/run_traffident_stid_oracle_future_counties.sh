#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/home/yuzhang_fei/code/event-traffic-prediction-git}"
CONDA_SH="${CONDA_SH:-/home/yuzhang_fei/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
GPU_ID="${GPU_ID:-0}"
FREE_RATIO_THRESHOLD="${FREE_RATIO_THRESHOLD:-50}"
TRAFFIDENT_BASICTS_ROOT="${TRAFFIDENT_BASICTS_ROOT:-/data/yuzhang_fei/TraffiDent/basicts}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/reproduction/logs}"
COUNTIES="${COUNTIES:-LosAngeles,Orange,Alameda,ContraCosta}"
TRAFFIDENT_NUM_EPOCHS="${TRAFFIDENT_NUM_EPOCHS:-100}"

mkdir -p "${LOG_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export TRAFFIDENT_BASICTS_ROOT
export TRAFFIDENT_NUM_EPOCHS
export WANDB_MODE="${WANDB_MODE:-online}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

gpu_free_ratio() {
  nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); printf "%.0f", ($1 * 100 / $2)}'
}

wait_for_gpu() {
  while true; do
    ratio="$(gpu_free_ratio)"
    echo "[$(timestamp)] GPU ${GPU_ID} free ratio: ${ratio}%"
    if [ "${ratio}" -ge "${FREE_RATIO_THRESHOLD}" ]; then
      return 0
    fi
    echo "[$(timestamp)] Waiting for GPU ${GPU_ID} free ratio >= ${FREE_RATIO_THRESHOLD}%"
    sleep 300
  done
}

IFS=',' read -r -a county_array <<< "${COUNTIES}"

cd "${CODE_DIR}/BasicTS"

for county in "${county_array[@]}"; do
  cfg="baselines/STIDOracleFutureAccident/TraffiDent_${county}.py"
  run_name="traffident_STIDOracleFutureAccident_${county}_${TRAFFIDENT_NUM_EPOCHS}ep_g${GPU_ID}"
  log_file="${LOG_DIR}/${run_name}_$(date +%Y%m%d_%H%M%S).log"
  wait_for_gpu
  echo "[$(timestamp)] Starting ${run_name}; cfg=${cfg}; log=${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python experiments/train.py -c "${cfg}" -g 0 2>&1 | tee "${log_file}"
  echo "[$(timestamp)] Finished ${run_name}"
done

echo "[$(timestamp)] All TraffiDent STIDOracleFutureAccident county jobs finished."
