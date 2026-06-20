# Reproduction Runs

This log records server-side reproduction setup and launch commands for the
external event-aware traffic forecasting repositories.

## Server Layout

- Server: `183.174.228.180`
- Code: `/home/yuzhang_fei/code/event-traffic-prediction-git`
- Data: `/data/yuzhang_fei/event-traffic-prediction`
- Conda env: `STGraph`

Large datasets are kept under `/data/yuzhang_fei/event-traffic-prediction` and
linked into each reproduction repository:

- `reproduction/IGSTGNN/data -> /data/yuzhang_fei/event-traffic-prediction/IGSTGNN/data`
- `reproduction/ConFormer/data -> /data/yuzhang_fei/event-traffic-prediction/ConFormer/data`

## IGSTGNN

Official data source:

- Kaggle dataset: `lixiangfan/incidentwithtraffic4alameda`
- Server path: `/data/yuzhang_fei/event-traffic-prediction/IGSTGNN/data/xtraffic/Alameda`

The Kaggle release stores Alameda as one large object array. The helper script
below converts it into the split files expected by the repository dataloader:

```bash
cd /home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/IGSTGNN
conda run -n STGraph python src/utils/prepare_alameda_splits.py \
  --data-dir data/xtraffic/Alameda
```

Generated split summary:

- Train: 8812 samples
- Val: 2937 samples
- Test: 2938 samples
- Mean: 147.8659210205078
- Std: 173.55581665039062

Smoke run completed on 2026-05-13 with one epoch:

```bash
cd /home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/IGSTGNN
CUDA_VISIBLE_DEVICES=1 conda run -n STGraph python experiments/IGSTGNN/main.py \
  --device cuda:0 \
  --dataset Alameda \
  --model_name igstgnn \
  --seed 2025 \
  --bs 48 \
  --incident \
  --use_sensor_info \
  --max_epochs 1 \
  --patience 1
```

Smoke result:

- Average Test MAE: 5975.7181
- Test RMSE: 9723.6697
- Test MAPE: 0.3602

IGSTGNN now saves the validation-best checkpoint and test arrays:

- Best checkpoint: `experiments/igstgnn/Alameda_2025/best_model_s2025.pt`
- Backward-compatible checkpoint copy: `experiments/igstgnn/Alameda_2025/final_model_s2025.pt`
- Test result: `experiments/igstgnn/Alameda_2025/test_result_s2025.npz`
- Test result keys: `prediction`, `target`, `metrics_by_horizon`,
  `metrics_average`, `incident_features`, `incident_position`,
  `incident_distances`, `durations`, `checkpoint`

Full official-style run is launched through a GPU-waiting screen:

```bash
screen -dmS igstgnn_alameda_official \
  bash /home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/server_scripts/wait_and_run_igstgnn_alameda.sh
```

Screen/log:

- Screen: `igstgnn_alameda_official`
- Log directory: `reproduction/IGSTGNN/experiments/IGSTGNN/server_logs`

The screen waits for GPU 0 to have more than 50% free memory, then runs the
README/run.sh-style Alameda command with default training epochs.

### Three-Dataset Official Reproduction

Current official IGSTGNN code is an independent PyTorch framework, not BasicTS
or LargeST. The official entrypoint is `experiments/IGSTGNN/main.py`; its
dataloader consumes incident-centered sample dictionaries with structured
incident fields and optional sensor metadata.

The current three-dataset reproduction target is server `183.174.228.172`:

- Code: `/home/yuzhang_fei/code/event-traffic-prediction-git`
- Data root: `/home/yuzhang_fei/data/event-traffic-prediction/IGSTGNN/data`
- Conda env: `igstgnn`
- GPUs: `1 3`
- Datasets: `Alameda`, `Contra_Costa`, `Orange`
- Official command settings: `seed=2025`, `seq_len=12`, `horizon=12`,
  `--incident`, `--use_sensor_info`, `max_epochs=100`, `patience=20`
- Paper batch sizes: `Alameda=48`, `Contra_Costa=48`, `Orange=24`

The 172 server does not have `/data/yuzhang_fei`, and `/data` is not writable
by the user, so the reproduction data is kept under `/home/yuzhang_fei/data`.
The server also did not have a Kaggle token at planning time; provide
`~/.kaggle/kaggle.json`, set `IGSTGNN_DATA_ZIP=/path/to/data4igstgnn.zip`, or
allow the script to try the public Kaggle download URL.

Project-side helper scripts:

```bash
reproduction/server_scripts/run_igstgnn_three_datasets_official.sh
reproduction/server_scripts/igstgnn_check_dataset.py
reproduction/server_scripts/igstgnn_export_test_results.py
reproduction/server_scripts/igstgnn_summarize_results.py
```

Launch on 172:

```bash
cd /home/yuzhang_fei/code/event-traffic-prediction-git
screen -dmS igstgnn_three_dataset_official \
  bash reproduction/server_scripts/run_igstgnn_three_datasets_official.sh
```

For a recovery run after one dataset fails, the launcher can be restricted
without changing any model or training hyperparameters:

```bash
IGSTGNN_DATASETS="Alameda Contra_Costa" \
  bash reproduction/server_scripts/run_igstgnn_three_datasets_official.sh
```

The launcher follows the paper batch sizes by default. The earlier Orange
smoke attempt with `bs=48` was an invalid non-paper setting and OOMed before
the first batch completed; it should not be counted as an official reproduction
failure.

Orange was then retried with the paper setting `bs=24` on 172 GPU 2 using
screen `igstgnn_orange_official_bs24`. The run did use `bs=24` but still OOMed
on the first training batch before the one-epoch smoke check completed:

```text
reproduction/logs/igstgnn_Orange_smoke_s2025_20260620_222257_g2.log
Namespace(... dataset='Orange', seed=2025, bs=24, ... layer=5, ...)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
```

No Orange full run or test-result file was produced from this attempt. This is
recorded as a current environment/implementation memory issue under the paper
batch setting, not as a reason to silently change model size, number of layers,
or batch size.

The launcher first prepares the official 70/15/15 splits without reshuffling or
renormalizing, runs one-epoch smoke checks, then runs the three full
reproductions. Training remains official; prediction arrays are exported after
training by loading `final_model_s2025.pt` and saving `test_result_s2025.npz`.
The summary CSV is written under `reproduction/logs/`.

Compatibility note: the Kaggle `incident_all.npy` object arrays may be pickled
with NumPy 2.x module paths while the official environment pins
`numpy==1.24.4`. The split/check helpers install a `numpy._core` import alias
only for loading these arrays; they do not reshuffle, renormalize, or change
training data values.

## ConFormer

Official data source:

- Google Drive folder from the repository README
- Server path: `/data/yuzhang_fei/event-traffic-prediction/ConFormer/data`

The server cannot reach Google Drive directly, so data is downloaded locally
with `gdown` and then synced to the server data path.

Synced data summary:

- BA, SD, TKY directories downloaded from the official Google Drive folder.
- Total server data size: 4.9G
- TKY originally ships `data.h5`, `adj.npy`, `external.npz`, and `index.npz`.
- `data.npz` is generated from `data.h5 + external.npz` before training.

TKY dataloader check completed on 2026-05-13:

```bash
conda run -n STGraph python \
  /home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/server_scripts/check_conformer_tky_data.py
```

Check result:

- Train: `x=(6557, 6, 1843, 5)`, `y=(6557, 6, 1843, 1)`
- Val: `x=(2186, 6, 1843, 5)`, `y=(2186, 6, 1843, 1)`
- Test: `x=(4493, 6, 1843, 5)`, `y=(4493, 6, 1843, 1)`
- First batch: `x=(16, 6, 1843, 5)`, `y=(16, 6, 1843, 1)`
- Scaler mean: 58.43083080428048
- Scaler std: 23.61220449621385

ConFormer already saves the validation-best model under `saved_models/`. It
now also writes a compressed test result file under `test_results/`:

- Test result pattern: `test_results/ConFormer-TKY-*-test_result.npz`
- Test result keys: `prediction`, `target`, `input_context`,
  `context_channel_names`, `metrics_overall`, `metrics_by_horizon`,
  `checkpoint`, `scaler_mean`, `scaler_std`
- `input_context` stores the accident/region input channels as `float16`, so
  accident-conditioned forecast behavior can be analyzed after training.

Official training entrypoint:

```bash
cd /home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/ConFormer/model
CUDA_VISIBLE_DEVICES=1 conda run -n STGraph python train.py -d tky -g 0 -m train
```

Prepared GPU-waiting launcher:

```bash
screen -dmS conformer_tky_official \
  bash /home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/server_scripts/wait_and_run_conformer_tky.sh
```

Screen/log:

- Screen: `conformer_tky_official`
- Log directory: `reproduction/ConFormer/logs/server_logs`

The screen waits for GPU 0 to have more than 50% free memory, then runs the
README-style TKY command.
