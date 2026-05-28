# Event Factor Analysis

**Last update**: 2026-05-28 12:45 Asia/Shanghai
**Script**: `reproduction/analysis/traffident_event_factor_metrics.py`
**Output**: `reproduction/analysis/traffident_event_factor_metrics/`

## Purpose

上一版事件窗口指标说明：`STIDGatedAccident` 能稳定打赢
`STIDAccident`，但没有稳定打赢纯 `STID`。本轮继续拆分事件窗口，回答：

1. 哪些事故类型下 gated residual 有用？
2. 上游/下游/事故源附近是否不同？
3. matched-control 交通变化强度是否决定了模型能否打赢 STID？

所有指标仍然来自保存好的 `test_results`，不重新训练。

## Main Findings

### 1. 事故类型：`1141` 最像可学习信号，`NoInj` 最不稳

四县平均 `Gated - STID` MAE：

| Type | Scope | Delta | Wins |
| --- | --- | ---: | ---: |
| `1141` | `future_any` | -0.0400 | 2/4 |
| `1141` | `post_last_slot` | -0.0506 | 3/4 |
| `UnknInj` | `ongoing` | -0.0492 | 2/4 |
| `NoInj` | `post_last_slot` | +0.1060 | 2/4 |
| `NoInj` | `history_any` | +0.0913 | 2/4 |

解释：

- `1141` 在 post-last 上是当前最像“事故模块能打赢 STID”的类型；
- `NoInj` 数量多但异质性强，直接二值 accident channel 会把大量弱影响样本也激活，导致负迁移；
- 下一版不应只用 binary accident，需要 type-conditioned router 或 type-specific residual prior。

### 2. 空间关系：downstream 更有希望，upstream/post-history 更容易失败

四县平均 `Gated - STID` MAE：

| Relation | Scope | Delta | Wins |
| --- | --- | ---: | ---: |
| downstream | `future_any` | -0.0309 | 3/4 |
| downstream | `ongoing` | -0.0730 | 3/4 |
| upstream | `post_last_slot` | +0.0719 | 1/4 |
| upstream | `ongoing` | +0.1322 | 2/4 |
| at_source | `history_any` | +0.1187 | 1/4 |

这里的 relation 定义为：

```text
signed_downstream_pm =
    (sensor_abs_pm - incident_abs_pm) * sign(Direction)

sign(Direction) = +1 for N/E, -1 for S/W
```

解释：

- downstream/ongoing 是 gated 当前最有结构的空间区域；
- upstream 和 at-source 的 history/post-last 很容易退化；
- V1 应显式加入 direction-aware / horizon-aware propagation，而不是只在 matched node 上做二值门控。

### 3. Impact 强度：高影响窗口最需要改，但当前并没有稳定赢

按 matched-control traffic-change 的绝对值分桶：

| Impact | Scope | Delta | Wins | Mean MAE |
| --- | --- | ---: | ---: | ---: |
| low | `future_any` | -0.0236 | 3/4 | 5.52 |
| high | `future_any` | -0.0004 | 2/4 | 23.67 |
| high | `post_last_slot` | +0.0825 | 2/4 | 24.24 |
| high | `history_any` | +0.1408 | 1/4 | 23.73 |
| high | `ongoing` | +0.1785 | 2/4 | 24.31 |

解释：

- gated 在低影响窗口更容易赢，但这些窗口本身误差很小，不足以支撑事故感知贡献；
- 真正重要的 high-impact windows 上，当前模型没有稳定赢 STID；
- V1 的训练目标必须显式偏向 high-impact matched-control windows，而不是让全量 MAE 主导。

### 4. Drop/Rise bias 很明显：模型在事故响应方向上回归到均值

按 matched-control traffic-change 的方向分：

| Direction | Scope | Delta | Wins | Bias |
| --- | --- | ---: | ---: | ---: |
| drop | `future_any` | -0.0220 | 2/4 | +3.9191 |
| drop | `post_last_slot` | -0.0046 | 2/4 | +3.6046 |
| rise | `future_any` | +0.0344 | 2/4 | -4.9353 |
| rise | `post_last_slot` | +0.0726 | 1/4 | -5.7949 |

Bias 是 `prediction - target`：

- drop 窗口 bias 为正，说明事故导致交通比正常更低时，模型普遍高估；
- rise 窗口 bias 为负，说明恢复/反弹窗口模型普遍低估；
- 这不是单纯 accident embedding 能解决的问题，需要方向化 residual：分别建模 congestion-drop 和 recovery-rise。

## Design Consequence

当前最清楚的结论不是“gated 已经成功”，而是：

> gated residual 是比 naive accident embedding 更安全的外壳；真正要打赢 STID，需要把 residual 训练目标和 router 输入改成 type-aware、direction-aware、impact-aware。

V1 不应再只使用 history accident binary mask。优先设计：

```text
Y_hat = Y_STID + M_local(type, relation, horizon, impact) *
        G(type, relation, pre_state, horizon) *
        R(type, relation, pre_state, horizon)
```

其中 `M_local` 由事件类型、上下游关系、距离和 horizon 决定；训练目标要额外强调 high-impact matched-control windows，并约束 no-event 不退化。

## Files

- `event_factor_metrics.csv`: full table for all models/factors/scopes.
- `event_factor_metrics_compact.csv`: compact gated-only table.
- `event_factor_gated_summary.csv`: cross-county summary for gated vs STID.
- `event_factor_metrics_summary.json`: metadata and relation definitions.
