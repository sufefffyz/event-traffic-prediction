# Training Status: STIDGatedAccident

**Last update**: 2026-05-28 05:15 Asia/Shanghai  
**Server screen**: `aris_stid_gated_counties_g0`  
**Script**: `reproduction/server_scripts/run_traffident_stid_gated_counties.sh`  
**Commit**: `a48db2a`

## Current Queue State

| County | Status | Epoch | Best Val MAE | Latest Test MAE | Latest Test RMSE | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LosAngeles | finished | 100/100 | 10.4122 | 11.0595 | 21.5549 | `ilz7hepg` |
| Orange | finished | 100/100 | 9.9055 | 10.4499 | 20.6822 | `vfi3y9o0` |
| Alameda | running | 36/100 | 10.9765 | 11.2728 | 21.6940 | `6n1ubk16` |
| ContraCosta | pending | - | - | - | - | - |

## LosAngeles Final

W&B:

```text
https://wandb.ai/825521004-renmin-university-of-china/event-traffic-prediction/runs/ilz7hepg
```

Artifacts:

```text
log:
reproduction/logs/traffident_STIDGatedAccident_LosAngeles_100ep_g0_20260527_230011.log

checkpoint/test-results:
BasicTS/checkpoints/STIDGatedAccident/TraffiDent_LosAngeles_2023Q1_100_12_12_gated_accident/2742d597068ff19e251df0b9964af382
```

Final test metrics:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 11.0595 | 10.2413 | 11.0767 | 12.3050 |
| MAPE | 0.2132 | 0.1935 | 0.2139 | 0.2401 |
| RMSE | 21.5549 | 19.5964 | 21.5870 | 24.1702 |

## Orange Final

W&B:

```text
https://wandb.ai/825521004-renmin-university-of-china/event-traffic-prediction/runs/vfi3y9o0
```

Artifacts:

```text
log:
reproduction/logs/traffident_STIDGatedAccident_Orange_100ep_g0_20260528_022742.log

checkpoint/test-results:
BasicTS/checkpoints/STIDGatedAccident/TraffiDent_Orange_2023Q1_100_12_12_gated_accident/e5986ea714679cd5c7e0d08413b634e4
```

Final test metrics:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 10.4499 | 9.6312 | 10.4761 | 11.6870 |
| MAPE | 0.2193 | 0.2000 | 0.2217 | 0.2468 |
| RMSE | 20.6822 | 18.5897 | 20.7013 | 23.4249 |

## Alameda Running Snapshot

W&B:

```text
https://wandb.ai/825521004-renmin-university-of-china/event-traffic-prediction/runs/6n1ubk16
```

Artifacts:

```text
log:
reproduction/logs/traffident_STIDGatedAccident_Alameda_100ep_g0_20260528_044544.log

checkpoint/test-results:
BasicTS/checkpoints/STIDGatedAccident/TraffiDent_Alameda_2023Q1_100_12_12_gated_accident/c2e7a0b7230869ca07041f6c44f746a9
```

Latest test metrics at epoch 35/36 snapshot:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 11.2728 | 10.4574 | 11.2731 | 12.5574 |
| MAPE | 0.1943 | 0.1844 | 0.1915 | 0.2169 |
| RMSE | 21.6940 | 19.7854 | 21.7230 | 24.2120 |

## Notes

- No training errors were found in the LA, Orange, or Alameda logs.
- `test_results` directories contain saved predictions/targets/inputs and are being overwritten with the latest evaluated checkpoint during training.
- Final comparison against STID/STIDAccident should wait until all four counties finish.
