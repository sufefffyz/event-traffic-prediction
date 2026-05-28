# Event-Sensitive Metric Analysis

**Last update**: 2026-05-28 10:55 Asia/Shanghai  
**Script**: `reproduction/analysis/traffident_model_event_metrics.py`  
**Output**: `reproduction/analysis/traffident_model_event_metrics/`

## Setup

This analysis reuses the saved BasicTS `test_results` from the 100-epoch,
seed-42 TraffiDent 2023Q1 county runs:

- `STID`
- `STIDAccident`
- `STIDGatedAccident`

Metrics are recomputed from saved predictions/targets with the same BasicTS
`null_val=0.0` masking rule. Event groups are defined per test sample and node:

| Group | Definition |
| --- | --- |
| `future_onset` | history has no accident, future window has accident |
| `post_last_slot` | last history slot has accident |
| `ongoing` | history and future both have accident |
| `history_only` | history has accident, future has no accident |
| `future_any` | future window has accident |
| `history_any` | history window has accident |
| `no_event` | no accident in history or future |

## Gated vs Baselines

Average over the four counties:

| Group | Mean event ratio | Gated - STID MAE | Wins vs STID | Gated - STIDAcc MAE | Wins vs STIDAcc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `future_onset` | 0.218% | -0.0128 | 3/4 | -0.0683 | 3/4 |
| `future_any` | 0.243% | -0.0080 | 3/4 | -0.1008 | 4/4 |
| `ongoing` | 0.025% | +0.0298 | 3/4 | -0.3635 | 4/4 |
| `post_last_slot` | 0.039% | +0.0281 | 2/4 | -0.3399 | 4/4 |
| `history_only` | 0.218% | +0.0710 | 1/4 | -0.2180 | 4/4 |
| `no_event` | 99.539% | +0.0047 | 1/4 | -0.0421 | 4/4 |
| `all` | 100.000% | +0.0048 | 1/4 | -0.0426 | 4/4 |

Interpretation:

- The gated residual router consistently fixes the naive `STIDAccident`
  degradation. This holds for almost every event slice and for all-window MAE.
- Compared with pure STID, the only promising slice is accident entering the
  forecast horizon: `future_onset` / `future_any`.
- `post_last_slot` and `history_only` do not yet support the claim that the
  current gated module improves post-incident forecasting over STID.
- Important caveat: the current model only sees accident activity in the input
  history and hard-masks the residual branch when history has no accident.
  Therefore `future_onset` gains cannot yet be claimed as true future-event
  awareness; they are indirect effects of retraining the base branch.
- Event windows are extremely sparse. `post_last_slot` is only about 0.03-0.06%
  of test node-windows, so per-county variance is large.

## Per-County Key Slices

Rows report `STIDGatedAccident`; `Delta` is MAE minus pure STID.

| Group | County | Node-windows | Ratio | MAE | Delta | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `future_onset` | LosAngeles | 31,254 | 0.341% | 11.9280 | -0.0164 | -0.5357 |
| `future_onset` | Orange | 8,503 | 0.166% | 11.3285 | -0.0284 | +0.0481 |
| `future_onset` | Alameda | 6,005 | 0.223% | 12.3599 | -0.0826 | -1.8617 |
| `future_onset` | ContraCosta | 3,671 | 0.143% | 14.3663 | +0.0763 | +0.5149 |
| `post_last_slot` | LosAngeles | 5,628 | 0.061% | 12.1969 | +0.0349 | -0.9613 |
| `post_last_slot` | Orange | 1,527 | 0.030% | 11.3090 | -0.0180 | -0.4238 |
| `post_last_slot` | Alameda | 1,087 | 0.040% | 12.6856 | -0.1633 | -1.6318 |
| `post_last_slot` | ContraCosta | 664 | 0.026% | 14.4256 | +0.2590 | -1.1547 |
| `ongoing` | LosAngeles | 3,572 | 0.039% | 12.5219 | -0.0086 | -1.2198 |
| `ongoing` | Orange | 954 | 0.019% | 11.3894 | -0.0019 | -0.5143 |
| `ongoing` | Alameda | 708 | 0.026% | 13.1664 | -0.1065 | -1.3830 |
| `ongoing` | ContraCosta | 445 | 0.017% | 13.8379 | +0.2359 | +0.0467 |
| `history_only` | LosAngeles | 31,217 | 0.340% | 12.2461 | +0.0450 | -0.6526 |
| `history_only` | Orange | 8,483 | 0.165% | 11.2998 | +0.0902 | -0.4183 |
| `history_only` | Alameda | 6,005 | 0.223% | 12.1624 | -0.0970 | -1.8749 |
| `history_only` | ContraCosta | 3,674 | 0.143% | 14.3286 | +0.2458 | -1.4830 |

Bias is `prediction - target`. Negative values mean underprediction. Most
event slices are underpredicted, especially Alameda and post-last cases.

## Current Research Read

The event-window analysis refines the claim:

> Sparse accident routing is better than direct accident embedding, but this
> first neural implementation is not yet a robust improvement over pure STID.
> The current future-onset signal is not sufficient because v0 does not receive
> future incident information. The next version must beat pure STID on
> observed-event or matched-impact windows.

Next checks:

1. Split event windows by incident type and local impact magnitude; the current
   binary accident channel may mix weak and strong events.
2. Add a stricter local propagation mask using `abs_pm` and upstream/downstream
   direction, especially for ContraCosta where the gated module hurts STID.
3. Report matched-control traffic change next to model error, so the method is
   judged on windows where the incident actually changes traffic.
