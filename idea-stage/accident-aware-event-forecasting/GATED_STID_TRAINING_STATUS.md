# Training Status: STIDGatedAccident

**Last update**: 2026-05-28 09:15 Asia/Shanghai  
**Server screen**: `aris_stid_gated_counties_g0` finished  
**Script**: `reproduction/server_scripts/run_traffident_stid_gated_counties.sh`  
**Training code commit**: `496ca80`

## Current Queue State

| County | Status | Epoch | Best Val MAE | Latest Test MAE | Latest Test RMSE | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LosAngeles | finished | 100/100 | 10.4122 | 11.0595 | 21.5549 | `ilz7hepg` |
| Orange | finished | 100/100 | 9.9055 | 10.4499 | 20.6822 | `vfi3y9o0` |
| Alameda | finished | 100/100 | 10.7280 | 11.1020 | 21.4389 | `6n1ubk16` |
| ContraCosta | finished | 100/100 | 10.2149 | 10.7854 | 20.9561 | `utzrira3` |

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

## ContraCosta Final

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

Final test metrics:

| Metric | All | h3 | h6 | h12 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 10.7854 | 9.9584 | 10.7904 | 12.0584 |
| MAPE | 0.2074 | 0.1928 | 0.2050 | 0.2335 |
| RMSE | 20.9561 | 18.9308 | 20.9275 | 23.7066 |

## Baseline Comparison

All rows use the same TraffiDent 2023Q1 county splits, BasicTS metrics, seed 42, and 100 epochs.

| County | STID MAE | STIDAcc MAE | Gated MAE | Gated - STID | Gated - STIDAcc | Gated RMSE | H3/H6/H12 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LosAngeles | 11.0493 | 11.1014 | 11.0595 | +0.0102 | -0.0419 | 21.5549 | 10.2413/11.0767/12.3050 |
| Orange | 10.4077 | 10.4653 | 10.4499 | +0.0422 | -0.0154 | 20.6822 | 9.6312/10.4761/11.6870 |
| Alameda | 11.1491 | 11.1231 | 11.1020 | -0.0471 | -0.0211 | 21.4389 | 10.3127/11.1226/12.3141 |
| ContraCosta | 10.7717 | 10.8774 | 10.7854 | +0.0137 | -0.0920 | 20.9561 | 9.9584/10.7904/12.0584 |

Interpretation:

- Gated residual improves over the direct accident-embedding baseline on all four counties.
- Gated residual only improves over pure STID on Alameda under overall MAE; LA/Orange/ContraCosta are near parity but slightly worse.
- This means the current module is a useful accident-aware alternative to naive accident embedding, but the overall metric does not yet support a broad claim that it beats pure STID.

## Notes

- No training errors were found in the LA, Orange, Alameda, or ContraCosta logs.
- `test_results` directories contain saved predictions/targets/inputs for the best evaluated checkpoints.
- Next analysis should compare event-window/post-incident metrics, because overall MAE partly washes out sparse accident effects.
