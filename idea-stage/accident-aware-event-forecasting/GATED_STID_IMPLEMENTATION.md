# Implementation Note: STIDGatedAccident

**Date**: 2026-05-27  
**Code**: `BasicTS/baselines/STIDGatedAccident`  
**Goal**: turn the sparse-router pilot into the first trainable BasicTS model.

## Model

The model keeps STID as the normal-traffic branch and adds a sparse accident residual branch:

```text
y_hat = y_base + gate(history, node, time, accident) * residual(history, node, time, accident)
```

The current implementation uses:

- `base_prediction`: pure STID-style prediction from traffic, node identity, time-of-day, and day-of-week;
- `accident_active`: binary mask from the accident feature in the input history window;
- `gate`: learned sigmoid gate initialized with negative bias;
- `residual`: learned additive accident residual;
- `active_mask`: forces residual correction to zero when the input history has no accident.

This directly follows the pilot result: dense residual correction caused negative transfer, while impact-gated residual correction improved all four county subsets.

## Configs

Added county configs:

- `BasicTS/baselines/STIDGatedAccident/TraffiDent_LosAngeles.py`
- `BasicTS/baselines/STIDGatedAccident/TraffiDent_Orange.py`
- `BasicTS/baselines/STIDGatedAccident/TraffiDent_Alameda.py`
- `BasicTS/baselines/STIDGatedAccident/TraffiDent_ContraCosta.py`

These match the existing TraffiDent STID/STIDAccident settings:

- input/output length: 12/12;
- split: existing `index.npz`;
- scaler: `IndexedNPZStandardScaler`, global channel normalization;
- metrics: BasicTS MAE/MAPE/RMSE;
- seed: 42;
- WandB enabled through `WandBTimeSeriesForecastingRunner`;
- saved checkpoints and test-results through BasicTS evaluation.

## First Training Plan

Run a short smoke training first:

```bash
cd /home/yuzhang_fei/code/event-traffic-prediction-git/BasicTS
WANDB_MODE=online TRAFFIDENT_NUM_EPOCHS=2 \
python experiments/train.py -c baselines/STIDGatedAccident/TraffiDent_Alameda.py -g 0
```

If the smoke run saves a checkpoint and test-results normally, launch the four-county 100-epoch run with the same seed and settings as STID/STIDAccident.

Server helper:

```bash
screen -dmS aris_stid_gated_counties_g0 bash -lc \
  'cd /home/yuzhang_fei/code/event-traffic-prediction-git && \
   GPU_ID=0 TRAFFIDENT_NUM_EPOCHS=100 \
   bash reproduction/server_scripts/run_traffident_stid_gated_counties.sh'
```

## Smoke Status

- Initial random forward passed on the server.
- First Alameda smoke run exposed a config issue: the prepared Alameda tensor has 521 nodes, not 325. ContraCosta has 496 nodes, not 280. The new `STIDGatedAccident` county configs were corrected to the actual `data.npz` shapes.
- Corrected random forward passed for all four county configs:
  - LosAngeles: 1771 nodes;
  - Orange: 990 nodes;
  - Alameda: 521 nodes;
  - ContraCosta: 496 nodes.
- A 2-epoch Alameda smoke run completed on GPU 0 with online WandB.

Smoke artifacts on the server:

```text
log:
/home/yuzhang_fei/code/event-traffic-prediction-git/reproduction/logs/aris_stid_gated_alameda_smoke2_20260527_2241.log

checkpoint/test-results:
/home/yuzhang_fei/code/event-traffic-prediction-git/BasicTS/checkpoints/STIDGatedAccident/TraffiDent_Alameda_2023Q1_2_12_12_gated_accident/4f0386f5bf8ad31e01f7d35755be1f0f
```

Smoke metrics after 2 epochs:

| Split | MAE | MAPE | RMSE |
| --- | ---: | ---: | ---: |
| val | 12.6836 | 0.2496 | 22.8253 |
| test | 12.9333 | 0.2224 | 23.7523 |

| Horizon | MAE | MAPE | RMSE |
| --- | ---: | ---: | ---: |
| h3 | 11.4607 | 0.1972 | 21.1850 |
| h6 | 12.8993 | 0.2193 | 23.7167 |
| h12 | 15.1027 | 0.2540 | 27.1950 |

WandB run:

```text
https://wandb.ai/825521004-renmin-university-of-china/event-traffic-prediction/runs/hf5a7wh8
```

## Known Limit

This first version only uses accidents visible in the input history window. It does not use future-known incident schedules, weather, or lane closure features. That choice keeps it aligned with the current TraffiDent county BasicTS tensors and avoids changing BasicTS core data loading.
