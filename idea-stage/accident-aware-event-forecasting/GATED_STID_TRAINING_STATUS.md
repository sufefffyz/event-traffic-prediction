# Training Status: STIDGatedAccident

**Last update**: 2026-05-28 03:15 Asia/Shanghai  
**Server screen**: `aris_stid_gated_counties_g0`  
**Script**: `reproduction/server_scripts/run_traffident_stid_gated_counties.sh`  
**Commit**: `496ca80`

## Current Queue State

| County | Status | Epoch | Best Val MAE | Latest Test MAE | Latest Test RMSE | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LosAngeles | finished | 100/100 | 10.4122 | 11.0595 | 21.5549 | `ilz7hepg` |
| Orange | running | 37/100 | 10.1368 | 10.5883 | 20.6274 | `vfi3y9o0` |
| Alameda | pending | - | - | - | - | - |
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

## Orange Running Snapshot

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

Latest test metrics at epoch 36/37 snapshot:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 10.5883 | 9.7574 | 10.6189 | 11.9008 |
| MAPE | 0.2544 | 0.2303 | 0.2623 | 0.3098 |
| RMSE | 20.6274 | 18.6470 | 20.6501 | 23.2836 |

## Notes

- No training errors were found in the LA or Orange logs.
- `test_results` directories contain saved predictions/targets/inputs and are being overwritten with the latest evaluated checkpoint during training.
- Final comparison against STID/STIDAccident should wait until all four counties finish.
