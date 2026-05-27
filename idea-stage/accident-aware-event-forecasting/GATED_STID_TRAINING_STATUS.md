# Training Status: STIDGatedAccident

**Last update**: 2026-05-28 07:15 Asia/Shanghai  
**Server screen**: `aris_stid_gated_counties_g0`  
**Script**: `reproduction/server_scripts/run_traffident_stid_gated_counties.sh`  
**Commit**: `b725800`

## Current Queue State

| County | Status | Epoch | Best Val MAE | Latest Test MAE | Latest Test RMSE | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LosAngeles | finished | 100/100 | 10.4122 | 11.0595 | 21.5549 | `ilz7hepg` |
| Orange | finished | 100/100 | 9.9055 | 10.4499 | 20.6822 | `vfi3y9o0` |
| Alameda | finished | 100/100 | 10.7280 | 11.1020 | 21.4389 | `6n1ubk16` |
| ContraCosta | running | 80/100 | 10.2682 | 10.8247 | 20.9411 | `utzrira3` |

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

## Alameda Final

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

Final test metrics:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 11.1020 | 10.3127 | 11.1226 | 12.3141 |
| MAPE | 0.1896 | 0.1721 | 0.1890 | 0.2183 |
| RMSE | 21.4389 | 19.6250 | 21.4937 | 23.8266 |

## ContraCosta Running Snapshot

W&B:

```text
https://wandb.ai/825521004-renmin-university-of-china/event-traffic-prediction/runs/utzrira3
```

Artifacts:

```text
log:
reproduction/logs/traffident_STIDGatedAccident_ContraCosta_100ep_g0_20260528_060858.log

checkpoint/test-results:
BasicTS/checkpoints/STIDGatedAccident/TraffiDent_ContraCosta_2023Q1_100_12_12_gated_accident/895b18270447cef32d32ee42e7bb34f8
```

Latest test metrics at epoch 79/80 snapshot:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 10.8247 | 10.0181 | 10.8176 | 12.0868 |
| MAPE | 0.2259 | 0.2192 | 0.2179 | 0.2499 |
| RMSE | 20.9411 | 18.9193 | 20.9171 | 23.6761 |

## Notes

- No training errors were found in the LA, Orange, Alameda, or ContraCosta logs.
- `test_results` directories contain saved predictions/targets/inputs and are being overwritten with the latest evaluated checkpoint during training.
- Final comparison against STID/STIDAccident should wait until all four counties finish.
