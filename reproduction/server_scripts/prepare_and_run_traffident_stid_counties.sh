#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/home/yuzhang_fei/code/event-traffic-prediction-git}"
CONDA_SH="${CONDA_SH:-/home/yuzhang_fei/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-STGraph}"
GPU_ID="${GPU_ID:-0}"
FREE_RATIO_THRESHOLD="${FREE_RATIO_THRESHOLD:-50}"
TRAFFIDENT_ZIP="${TRAFFIDENT_ZIP:-/data/yuzhang_fei/TraffiDent/xtraffic.zip}"
TRAFFIDENT_BASICTS_ROOT="${TRAFFIDENT_BASICTS_ROOT:-/data/yuzhang_fei/TraffiDent/basicts}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs/server_logs}"
COUNTIES="${COUNTIES:-LosAngeles,Orange,Alameda,ContraCosta}"
MONTHS="${MONTHS:-1,2,3}"
YEAR="${YEAR:-2023}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
RUN_MODELS="${RUN_MODELS:-STID,STIDAccident}"

mkdir -p "${LOG_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export TRAFFIDENT_BASICTS_ROOT
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

if [ "${SKIP_PREPARE}" != "1" ]; then
  echo "[$(timestamp)] Preparing TraffiDent BasicTS datasets..."
  python "${CODE_DIR}/reproduction/TraffiDent/prepare_county_basicts.py" \
    --zip-path "${TRAFFIDENT_ZIP}" \
    --output-root "${TRAFFIDENT_BASICTS_ROOT}" \
    --year "${YEAR}" \
    --months "${MONTHS}" \
    --counties "${COUNTIES}" \
    --event-types accident \
    --match-scope subset \
    --overwrite
fi

IFS=',' read -r -a county_array <<< "${COUNTIES}"
IFS=',' read -r -a model_array <<< "${RUN_MODELS}"

cd "${CODE_DIR}/BasicTS"

for county in "${county_array[@]}"; do
  for model in "${model_array[@]}"; do
    cfg="baselines/${model}/TraffiDent_${county}.py"
    run_name="traffident_${model}_${county}_g${GPU_ID}"
    log_file="${LOG_DIR}/${run_name}_$(date +%Y%m%d_%H%M%S).log"
    wait_for_gpu
    echo "[$(timestamp)] Starting ${run_name}; cfg=${cfg}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python experiments/train.py -c "${cfg}" -g 0 2>&1 | tee "${log_file}"
    echo "[$(timestamp)] Finished ${run_name}"
  done
done

echo "[$(timestamp)] All TraffiDent STID county jobs finished."
