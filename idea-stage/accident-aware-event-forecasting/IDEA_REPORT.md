# ARIS Idea Report: Accident-Aware Event Traffic Forecasting

**Generated**: 2026-05-27  
**Pipeline**: ARIS idea-creator style: landscape scan -> divergent idea generation -> first-pass filtering -> novelty check -> reviewer critique -> ranked candidates  
**Direction**: 事故感知的事件交通预测  
**User constraint**: 不把纯评估/诊断作为主方向，优先方法、框架、学习目标、迁移、检索、因果或不确定性建模。

## Landscape Summary

TraffiDent 把 traffic time series、incident records、road meta features 在 California 大规模对齐，支持 post-incident traffic forecasting、incident classification、global/local causal analysis。它给出的重要机会不是再做一个普通 PEMS-like benchmark，而是研究 incident 与 traffic 的双向关系。

IGSTGNN 已经把事故影响建模为 incident-context spatial fusion 和 temporal impact decay，因此“事故空间传播 + 时间衰减”本身不能作为新意，需要更强的差异点。ConFormer 把 accident/regulation 作为 node-time condition，并用 guided layer normalization 调制 Transformer；因此“把 accident channel 拼进模型”也不够新。

本项目已有初步结果显示，普通 STID+accident embedding 在 Los Angeles、Orange、Alameda、Contra Costa 四个 county 上没有稳定优于 STID。这说明事故信息不能被当作一个全局弱协变量使用，更可能需要局部激活、反事实残差、检索记忆或不确定性建模。

## Recommended Ideas

### Idea 1: Counterfactual Incident-Effect Forecaster

- **One-line**: 同时预测 factual traffic 和 “如果没有事故” 的 counterfactual baseline，把事故影响建模成 treatment effect residual。
- **Hypothesis**: 事故 embedding 不稳，是因为模型在学习平均交通，而不是学习事故相对正常状态造成的增量效应。
- **Method sketch**:  
  `Y_factual = Base(X)`，`Y_no_event = CounterfactualBase(X, matched controls)`，`Y_event_delta = EventEffect(E, node, horizon)`，最终 `Y = Y_no_event + Y_event_delta`。  
  用同节点、同星期、同时段、相近 pre-traffic 的 non-incident windows 构造 weak counterfactual target。
- **Closest work**: TraffiDent causal analysis, STEVE, IGSTGNN.
- **Novelty**: 8/10. 现有工作多把事故作为输入条件，较少直接建模事故造成的 counterfactual traffic effect。
- **Feasibility**: 中。已有 TraffiDent 和 test-results；难点是 matched control 的构造和防止伪因果。
- **Risk**: MEDIUM-HIGH.
- **Estimated effort**: 4-6 周。
- **Why worth doing**: 叙事最强，能把“事故感知预测”从 feature fusion 提升到 intervention-aware forecasting。

### Idea 2: Retrieval-Augmented Incident Memory

- **One-line**: 为每个新事故检索历史相似事故的 response trajectory，作为预测时的 external memory。
- **Hypothesis**: 事故是稀疏长尾事件，纯参数模型很难记住所有类型；TraffiDent 2022-2024 的历史事故库足以支持 case-based forecasting。
- **Method sketch**:  
  建一个 incident memory bank，每个 memory 包含 incident type、abs_pm / freeway、方向、road meta、pre-traffic state、time-of-day、post-incident response。预测时检索 top-k 相似事故，用 cross-attention 或 residual adapter 注入 STID/D2STGNN。
- **Closest work**: IGSTGNN incident-centered samples, ConFormer condition channel, memory-based traffic forecasting.
- **Novelty**: 8/10. 和 IGSTGNN/ConFormer 区分明显，突出“历史事故响应记忆”。
- **Feasibility**: 中高。可以先做 retrieval-only baseline，再接 STID adapter。
- **Risk**: MEDIUM.
- **Estimated effort**: 3-5 周。
- **Why worth doing**: 最贴合 TraffiDent 的数据优势，pilot 快，失败也能定位是 retrieval 维度还是事件稀疏性问题。

### Idea 3: Sparse Accident Expert Router

- **One-line**: 只在事故可能真实影响预测的局部时空区域激活 accident expert，避免事故模块污染普通样本。
- **Hypothesis**: STID+accident embedding 失败的核心原因是负迁移：大量无影响样本也被 accident feature 扰动。
- **Method sketch**:  
  Base forecaster 负责常规交通；accident expert 只预测局部 residual；router/gate 根据 incident distance、direction、road meta、pre-state anomaly、horizon 输出稀疏激活权重。
- **Closest work**: ConFormer conditional branch, mixture-of-experts ST forecasting, TraffiDent incident-enhanced STID baseline.
- **Novelty**: 7/10. 技术不激进，但问题切得准。
- **Feasibility**: 高。最适合基于当前 BasicTS + STID 快速实现。
- **Risk**: MEDIUM-LOW.
- **Estimated effort**: 2-4 周。
- **Why worth doing**: 工程最稳，能直接回答“为什么 naive accident embedding 不起作用”。
- **Current evidence**: 训练版 `STIDGatedAccident` 在四县整体 MAE 上稳定优于 `STIDAccident`，但只在 Alameda 优于纯 `STID`。事件切片后，`future_onset/future_any` 是最有希望的区域：gated 对纯 STID 平均 MAE 分别为 -0.0128/-0.0080，均为 3/4 county 胜出；`post_last/history_only` 仍不稳定。详见 `EVENT_METRIC_ANALYSIS.md`。

### Idea 4: Event-Conditioned Propagation Field

- **One-line**: 把事故建模成沿道路方向传播的时变扰动场，而不是二值 node-time channel。
- **Hypothesis**: 事故影响具有方向性、距离衰减和 horizon-dependent delay，固定图或简单 accident embedding 捕捉不到。
- **Method sketch**:  
  从事件源生成 anisotropic diffusion/advection kernel，边权由 upstream/downstream、abs_pm gap、road distance、历史速度、incident type 动态调制；对不同 horizon 学不同 delay kernel。
- **Closest work**: IGSTGNN ICSF/TIID, PDFormer propagation delay, ConFormer graph propagation.
- **Novelty**: 7/10. 传播/衰减已有相近工作，但如果强调方向、abs_pm、速度诱导 delay，会更有交通机理感。
- **Feasibility**: 中。
- **Risk**: MEDIUM.
- **Estimated effort**: 5-7 周。
- **Why worth doing**: 适合写成交通机理明确的方法论文，图和解释会好看。

### Idea 5: Uncertainty-First Accident Forecasting

- **One-line**: 不把事故感知只定义为均值预测更准，而是定义为事故窗口下分布、尾部风险和预测区间更可靠。
- **Hypothesis**: 事故对 mean forecast 的提升可能有限，但会显著改变 conditional variance 和 heavy-tail risk。
- **Method sketch**:  
  在 STID/D2STGNN 上接 quantile head、mixture density head 或 conformal residual head；事故上下文调制 interval width、tail mixture weight、calibration temperature。
- **Closest work**: incident-aware conformal STT, traffic accident risk forecasting, probabilistic traffic forecasting.
- **Novelty**: 7/10. 需要避免和 conformal 论文过近，重点放在 accident-conditioned traffic distribution。
- **Feasibility**: 中高。
- **Risk**: MEDIUM.
- **Estimated effort**: 3-5 周。
- **Why worth doing**: 如果 MAE 不明显涨，这条线仍可能产出强结果：事故主要体现为风险不确定性。

### Idea 6: Event-Aware Pretraining for Traffic Forecasting

- **One-line**: 利用 TraffiDent 2022-2024 做 self-supervised pretraining，学习事故响应表征，再 fine-tune 到 county-level forecasting。
- **Hypothesis**: 事故样本稀疏且异质，直接监督训练不稳；预训练能学习可迁移的事故-交通响应模式。
- **Method sketch**:  
  预训练任务包括 masked traffic reconstruction、masked incident reconstruction、future residual ranking、neighbor impact contrastive learning。下游接 STID/D2STGNN/ConFormer-style forecaster。
- **Closest work**: STEVE self-supervised confounder learning, masked spatiotemporal pretraining, TraffiDent.
- **Novelty**: 7/10.
- **Feasibility**: 中低。数据够，但工程量更大。
- **Risk**: MEDIUM-HIGH.
- **Estimated effort**: 6-10 周。
- **Why worth doing**: 适合作为中长期方向，能扩展到 weather/event/foundation-model 叙事。

## Lower-Priority Ideas

| Idea | Reason Not Top |
| --- | --- |
| Traffic-Accident Dual-Task Forecaster | 容易变成多任务拼接，和 TraffiDent incident classification 太近 |
| County-Invariant Accident Adapter | 有价值，但需要先证明事故效应确实可迁移 |
| Continuous-Time Incident Token Transformer | 表达优雅，但工程量和复现风险较高 |
| Incident-Induced Graph Rewiring | 新意不错，但容易和动态/因果图工作缠在一起，验证难 |
| Road-Attribute Constrained Incident Kernel | 可作为 Idea 4 的子模块，不建议单独成主线 |

## Novelty-Checked Ranking

The first ranking was too optimistic. After targeted novelty search, several directions are much closer to existing work than they first appeared:

- Counterfactual crash/post-crash prediction is close to crash HTE and MSCT.
- Generic retrieval-augmented traffic prediction is close to RAST.
- Generic sparse expert/router is close to traffic MoE work such as MH-MoE and TESTAM.
- Generic masked/event pretraining is close to STMAE/STEP-style traffic pretraining.

Therefore the safe main idea is not any single generic component. The safer novelty is their accident-specific composition:

> sparse node-horizon residual routing over a matched no-event baseline, optionally using retrieved historical incident residuals.

| Rank | Idea | Contribution Type | Novelty | Feasibility | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Sparse Accident Residual Router | accident-specific residual routing | 6.5 | High | Med-Low | 最适合作为主模型骨架 |
| 2 | Counterfactual-inspired Residual Target | learning objective / framing | 5.5 | Medium | High | 可做叙事，但不能强称 causal |
| 3 | Incident Response Memory | incident-specific retrieval prior | 5.5 | High | Medium | 先做 pilot/组件，不单独主线 |
| 4 | Event-Conditioned Propagation Field | dynamic graph prior | 5 | Medium | Medium-High | 作为 router prior，不单独主线 |
| 5 | Uncertainty-First Accident Forecasting | probabilistic forecasting | 5 | Med-High | Medium-High | 备选，不作为第一主线 |
| 6 | Event-Aware Pretraining | representation learning | 4 | Med-Low | High | 暂时放弃 |

## Suggested Main Project

**推荐组合**: Counterfactual-inspired Sparse Accident Residual Router + Incident Response Memory.

主论文可以这样讲：

1. 普通 accident embedding 把事故当作全局弱协变量，容易被常规交通模式淹没。
2. 事故预测应分解为 normal traffic baseline 和 incident-induced residual。
3. residual 不应全局激活，而应由 sparse router 在 node-horizon 局部选择性生效。
4. 历史相似事故 residual 可作为 retrieval prior，但必须避免 post-event leakage。
5. 用 TraffiDent 多 county 验证：整体不降，事故影响窗口、长 horizon、特定事故类型、上游邻域提升。

这个组合比纯 counterfactual 更安全，比纯 router 更有事故机制，也能避开 RAST/MoE/IGSTGNN 的直接覆盖。

**Hard evaluation bar**: 新模块必须以纯 `STID` 作为主要对手。打赢 `STIDAccident` 只能说明 naive accident embedding 被修正，不能单独构成方法有效性。每个新模块在训练前必须写清楚公式/架构图，避免只堆实验。

## Immediate Pilot Plan

| Pilot | Time | Success Signal |
| --- | --- | --- |
| Retrieval-only incident memory | done | NEGATIVE: top-k historical incident residual retrieval was worse than normal prior in all four counties |
| Linear sparse-residual proxy | done | POSITIVE FOR GATING: dense ridge residual was noisy, but impact-score gated residual improved all four counties |
| STID + gated accident residual router | done | MIXED: beats STIDAccident on all counties, but does not robustly beat pure STID; see `EVENT_METRIC_ANALYSIS.md` and `MODULE_ARCHITECTURE.md` |
| Counterfactual residual target construction | 2-4 天 | matched no-event baseline 能稳定解释 normal traffic，事故 residual 有非零结构 |

## Novelty Check Artifacts

- `NOVELTY_CHECK.md`: conservative novelty scoring and closest-prior table.
- `REVIEW_SUMMARY.md`: reviewer critique and final paper framing.
- `PILOT_RESULTS.md`: first training-free retrieval memory pilot. It was negative, so retrieval is downgraded to a baseline/component.
- `SPARSE_ROUTER_PILOT_RESULTS.md`: ridge residual expert pilot. It was mixed/mostly negative, supporting the need for sparse local gating.
- `EVENT_FACTOR_ANALYSIS.md`: event type, post-mile relation, and matched-control impact slices. It suggests V1 should be type-aware, direction-aware, and high-impact-window-aware to beat pure STID.
- `.aris/traces/novelty-check/2026-05-27_run01/trace.md`: ARIS trace record.

## Sources

- TraffiDent: https://arxiv.org/abs/2407.11477
- IGSTGNN: https://arxiv.org/abs/2602.02528
- ConFormer: https://arxiv.org/abs/2512.09398
- STEVE: https://arxiv.org/abs/2311.12472
- Incident-aware conformal STT: https://arxiv.org/abs/2603.16857
