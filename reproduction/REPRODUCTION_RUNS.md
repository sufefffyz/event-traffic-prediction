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
