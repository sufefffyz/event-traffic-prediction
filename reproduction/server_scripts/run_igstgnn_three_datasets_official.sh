#!/bin/bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/yuzhang_fei/code/event-traffic-prediction-git}
CONDA_ROOT=${CONDA_ROOT:-/home/yuzhang_fei/miniconda3}
CONDA_ENV=${CONDA_ENV:-igstgnn}
DATA_ROOT=${IGSTGNN_DATA_ROOT:-/home/yuzhang_fei/data/event-traffic-prediction/IGSTGNN/data}
DATA_ZIP=${IGSTGNN_DATA_ZIP:-}
SEED=${IGSTGNN_SEED:-2025}
BATCH_SIZE=${IGSTGNN_BATCH_SIZE:-48}
MAX_EPOCHS=${IGSTGNN_MAX_EPOCHS:-100}
PATIENCE=${IGSTGNN_PATIENCE:-20}
RUN_SMOKE=${IGSTGNN_RUN_SMOKE:-1}
SMOKE_EPOCHS=${IGSTGNN_SMOKE_EPOCHS:-1}
SMOKE_PATIENCE=${IGSTGNN_SMOKE_PATIENCE:-1}
FREE_RATIO_THRESHOLD=${FREE_RATIO_THRESHOLD:-50}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_IDS=(${IGSTGNN_GPU_IDS:-1 3})
DATASETS=(Alameda Contra_Costa Orange)

RUN_ROOT="$REPO_ROOT/reproduction/IGSTGNN"
REPO_XTRAFFIC_ROOT="$RUN_ROOT/data/xtraffic"
XTRAFFIC_ROOT="$DATA_ROOT/xtraffic"
LOG_DIR="$REPO_ROOT/reproduction/logs"
RUN_TS=$(date +%Y%m%d_%H%M%S)
FULL_RESULTS_LIST="$LOG_DIR/igstgnn_three_dataset_full_results_${RUN_TS}.txt"
SUMMARY_CSV="$LOG_DIR/igstgnn_three_dataset_summary_s${SEED}_${RUN_TS}.csv"
DATASET_SUMMARY_JSON="$LOG_DIR/igstgnn_three_dataset_data_summary_${RUN_TS}.json"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ensure_env() {
  log "Ensuring conda env: $CONDA_ENV"
  eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -n "$CONDA_ENV" python=3.11 pip -y
  fi
  conda activate "$CONDA_ENV"
  python -m pip install -r "$RUN_ROOT/requirements.txt"
}

download_or_unpack_data() {
  mkdir -p "$DATA_ROOT" "$XTRAFFIC_ROOT"
  if all_dataset_sources_present; then
    return
  fi

  if [ -n "$DATA_ZIP" ] && [ -f "$DATA_ZIP" ]; then
    log "Unpacking provided IGSTGNN data zip: $DATA_ZIP"
    unzip -oq "$DATA_ZIP" -d "$DATA_ROOT"
    return
  fi

  if command -v kaggle >/dev/null 2>&1 && [ -f "$HOME/.kaggle/kaggle.json" ]; then
    log "Downloading IGSTGNN data with Kaggle CLI"
    kaggle datasets download -d lixiangfan/data4igstgnn -p "$DATA_ROOT" --unzip
    return
  fi

  local zip_path="$DATA_ROOT/data4igstgnn.zip"
  log "Trying public Kaggle download URL: $zip_path"
  curl -L --fail -o "$zip_path" \
    https://www.kaggle.com/api/v1/datasets/download/lixiangfan/data4igstgnn
  unzip -oq "$zip_path" -d "$DATA_ROOT"
}

all_dataset_sources_present() {
  local dataset
  for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$XTRAFFIC_ROOT/$dataset/incident_all.npy" ]; then
      return 1
    fi
  done
  return 0
}

normalize_data_layout() {
  mkdir -p "$XTRAFFIC_ROOT" "$REPO_XTRAFFIC_ROOT"
  local dataset candidate target repo_target
  for dataset in "${DATASETS[@]}"; do
    target="$XTRAFFIC_ROOT/$dataset"
    if [ ! -d "$target" ]; then
      candidate=$(find "$DATA_ROOT" -type d -name "$dataset" | head -n 1 || true)
      if [ -z "$candidate" ]; then
        echo "Could not find dataset directory after unpacking: $dataset" >&2
        exit 1
      fi
      ln -s "$candidate" "$target"
    fi

    repo_target="$REPO_XTRAFFIC_ROOT/$dataset"
    if [ -L "$repo_target" ] || [ -d "$repo_target" ]; then
      continue
    fi
    ln -s "$target" "$repo_target"
  done
}

prepare_splits() {
  cd "$RUN_ROOT"
  local dataset
  for dataset in "${DATASETS[@]}"; do
    log "Preparing official 70/15/15 split for $dataset"
    python data/xtraffic/prepare_splits.py --dataset "$dataset" --overwrite
  done

  python "$REPO_ROOT/reproduction/server_scripts/igstgnn_check_dataset.py" \
    --xtraffic-root "$REPO_XTRAFFIC_ROOT" \
    --summary-json "$DATASET_SUMMARY_JSON"
}

gpu_free_ratio() {
  local gpu_id="$1"
  nvidia-smi --id="$gpu_id" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); printf "%.0f", ($1 * 100 / $2)}'
}

wait_for_gpu() {
  local gpu_id="$1"
  while true; do
    local free_ratio
    free_ratio=$(gpu_free_ratio "$gpu_id")
    log "GPU ${gpu_id} free memory ratio: ${free_ratio}%"
    if [ "$free_ratio" -ge "$FREE_RATIO_THRESHOLD" ]; then
      break
    fi
    sleep "$POLL_SECONDS"
  done
}

latest_run_dir() {
  local dataset="$1"
  find "$RUN_ROOT/experiments/igstgnn" -maxdepth 1 -type d \
    -name "${dataset}_igstgnn_s${SEED}_*" 2>/dev/null | sort | tail -n 1
}

run_dataset() {
  local dataset="$1"
  local gpu_id="$2"
  local epochs="$3"
  local patience="$4"
  local phase="$5"
  local log_file="$LOG_DIR/igstgnn_${dataset}_${phase}_s${SEED}_${RUN_TS}_g${gpu_id}.log"

  wait_for_gpu "$gpu_id"
  cd "$RUN_ROOT"
  log "Starting IGSTGNN $phase run: dataset=$dataset gpu=$gpu_id epochs=$epochs"
  CUDA_VISIBLE_DEVICES="$gpu_id" python experiments/IGSTGNN/main.py \
    --device cuda:0 \
    --dataset "$dataset" \
    --model_name igstgnn \
    --seed "$SEED" \
    --bs "$BATCH_SIZE" \
    --incident \
    --use_sensor_info \
    --max_epochs "$epochs" \
    --patience "$patience" \
    2>&1 | tee "$log_file"

  local run_dir
  run_dir=$(latest_run_dir "$dataset")
  if [ -z "$run_dir" ]; then
    echo "Could not locate IGSTGNN run directory for $dataset" >&2
    exit 1
  fi

  local output_file="$run_dir/test_result_s${SEED}.npz"
  if [ "$phase" = "smoke" ]; then
    output_file="$run_dir/test_result_s${SEED}_smoke.npz"
  fi
  CUDA_VISIBLE_DEVICES="$gpu_id" python "$REPO_ROOT/reproduction/server_scripts/igstgnn_export_test_results.py" \
    --igstgnn-root "$RUN_ROOT" \
    --device cuda:0 \
    --dataset "$dataset" \
    --seed "$SEED" \
    --bs "$BATCH_SIZE" \
    --incident \
    --use_sensor_info \
    --run-dir "$run_dir" \
    --output "$output_file" \
    2>&1 | tee -a "$log_file"

  if [ "$phase" = "full" ]; then
    echo "$output_file" >> "$FULL_RESULTS_LIST"
  fi
}

run_full_queue() {
  local active=0
  local idx=0
  local dataset gpu_id
  for dataset in "${DATASETS[@]}"; do
    gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
    run_dataset "$dataset" "$gpu_id" "$MAX_EPOCHS" "$PATIENCE" "full" &
    idx=$((idx + 1))
    active=$((active + 1))
    if [ "$active" -ge "${#GPU_IDS[@]}" ]; then
      wait -n
      active=$((active - 1))
    fi
  done
  wait
}

summarize_full_results() {
  if [ ! -s "$FULL_RESULTS_LIST" ]; then
    echo "No full-run test results were recorded." >&2
    exit 1
  fi
  mapfile -t result_files < "$FULL_RESULTS_LIST"
  python "$REPO_ROOT/reproduction/server_scripts/igstgnn_summarize_results.py" \
    --results "${result_files[@]}" \
    --output-csv "$SUMMARY_CSV"
}

main() {
  log "IGSTGNN three-dataset official reproduction launcher"
  log "Repo: $REPO_ROOT"
  log "Data root: $DATA_ROOT"
  log "GPUs: ${GPU_IDS[*]}"
  ensure_env
  download_or_unpack_data
  normalize_data_layout
  prepare_splits

  if [ "$RUN_SMOKE" = "1" ]; then
    local dataset
    for dataset in "${DATASETS[@]}"; do
      run_dataset "$dataset" "${GPU_IDS[0]}" "$SMOKE_EPOCHS" "$SMOKE_PATIENCE" "smoke"
    done
  fi

  run_full_queue
  summarize_full_results
  log "Summary CSV: $SUMMARY_CSV"
}

main "$@"
