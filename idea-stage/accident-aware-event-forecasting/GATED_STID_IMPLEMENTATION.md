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

## Known Limit

This first version only uses accidents visible in the input history window. It does not use future-known incident schedules, weather, or lane closure features. That choice keeps it aligned with the current TraffiDent county BasicTS tensors and avoids changing BasicTS core data loading.
