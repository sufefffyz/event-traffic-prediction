# Oracle Future Accident Router

**Status**: four-county 100-epoch run finished; signal mostly negative
**Code**: `BasicTS/baselines/STIDOracleFutureAccident`
**Purpose**: aggressive upper-bound diagnostic

## Setting

This module intentionally reads the future accident sequence from
`future_data[..., accident_feature_index]`. It does not read future target
flow. The result should be interpreted as an oracle / known-future-event
upper bound, not as a deployable forecasting protocol.

The experiment answers:

> If the model is told which sensor-time slots will have accident records in
> the forecast horizon, can it finally beat pure STID on event windows?

## Formula

Let pure STID produce:

```text
Z = STIDEncoder(X_hist)
Y_base = W_base(Z)
```

The oracle branch builds future event features for each horizon `h` and node
`i`:

```text
e_{h,i}         = 1[future accident at horizon h, node i]
any_i           = 1[sum_h e_{h,i} > 0]
tau_h           = h / H
```

The prediction is:

```text
F = EventConv([e, any, tau * any, (1 - tau) * any])
H_h = concat(Z, F_h)
Y_hat_h = Y_base_h + any_i * sigmoid(G(H_h)) * R(H_h)
```

`any_i` keeps no-future-event windows identical to STID, while allowing all
horizons in a future-event window to be corrected.

## Why This Is More Aggressive

Previous modules used only history accident states:

```text
history accident -> residual
```

This one uses the future accident sequence directly:

```text
future accident sequence -> horizon-wise residual
```

So it can affect `future_onset` windows where history has no accident but the
forecast horizon does. This is exactly the slice where earlier gated results
showed the cleanest weak positive signal.

## Success Criterion

The first check is not paper-valid realism. It is signal existence:

```text
MAE(STIDOracleFutureAccident) - MAE(STID) < 0
```

Priority slices:

- `future_onset`
- `future_any`
- `ongoing`
- `high_impact_*`

If this model cannot beat STID on those slices, future accident labels alone
are probably not enough under the current node-event mapping. If it can, the
next research step is to replace oracle future events with predicted /
retrieved event likelihoods.

## 2026-05-29 Result

四县 100 epoch 已跑完，`checkpoint`、`test_results`、`test_metrics.json` 均已落盘。
server 端主要输出：

- `BasicTS/checkpoints/STIDOracleFutureAccident/*/test_results`
- `reproduction/analysis/traffident_model_event_metrics_oracle/`
- `reproduction/analysis/traffident_event_factor_metrics_oracle/`

### Overall MAE

| County | STID | OracleFuture | Delta |
| --- | ---: | ---: | ---: |
| LosAngeles | 11.0493 | 11.0691 | +0.0199 |
| Orange | 10.4077 | 10.4757 | +0.0680 |
| Alameda | 11.1491 | 11.1224 | -0.0267 |
| ContraCosta | 10.7717 | 10.8118 | +0.0401 |

结论：整体指标没有打赢纯 `STID`，只有 Alameda 胜出。

### Event-Window MAE Delta vs STID

| Slice | Mean Delta | Wins |
| --- | ---: | ---: |
| `future_onset` | +0.0987 | 2/4 |
| `ongoing` | +0.1169 | 2/4 |
| `post_last_slot` | +0.0131 | 3/4 |
| `history_only` | +0.0124 | 2/4 |

关键结论：直接喂未来事故序列并没有在 `future_onset` 上产生上界信号，
反而比纯 `STID` 更差。这说明问题不只是“模型看不到未来事件”，还包括
事件-节点匹配、事故强度、传播方向、以及 residual 目标是否可学。

### Fine-Grained Signal

仍有少量局部正信号：

| Factor | Slice | Mean Delta | Wins |
| --- | --- | ---: | ---: |
| `incident_type=1141` | `post_last_slot` | -0.1356 | 4/4 |
| `incident_type=1141` | `ongoing` | -0.1222 | 4/4 |
| `incident_type=1141` | `future_any` | -0.0214 | 2/4 |
| `pm_relation=downstream` | `post_last_slot` | -0.0281 | 3/4 |
| `pm_relation=downstream` | `ongoing` | -0.0061 | 3/4 |
| `impact_abs=high` | `post_last_slot` | -0.0691 | 3/4 |

但是 `future_any` 的 high-impact 平均 delta 为 `+0.1531`，`ongoing`
high-impact 平均 delta 为 `+0.1681`，说明局部胜出不稳定。

### Decision

`STIDOracleFutureAccident` 不支持“只要知道未来事故就能稳定提升”的假设。
下一步不应继续简单加大 future accident embedding，而应转向：

1. type-aware residual，尤其优先处理 `1141`；
2. downstream/upstream directional routing；
3. high-impact post-event residual，但必须避免一个 county 的大幅恶化；
4. 重新检查事故到节点的映射和影响半径，而不是只在 matched node 上做局部标记。
