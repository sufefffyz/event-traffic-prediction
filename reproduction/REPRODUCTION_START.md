# 复现启动分析：ConFormer / IGSTGNN / BasicTS

日期：2026-05-13

这份文档面向后续复现和重构，重点回答三个问题：

1. 事件/事故交通预测到底是什么任务，和普通交通预测有什么不同。
2. 两个仓库里实际走到 forward/training path 的模块有哪些，哪些地方目前可疑或会直接断。
3. 它们和 BasicTS 0.5.8 的训练框架接口差在哪里，迁移时该优先改什么。

## 1. 任务类型

普通交通预测一般是连续时间序列预测：给定历史窗口 `X[t-L+1:t]`，预测未来窗口 `Y[t+1:t+H]`。输入通常是速度/流量/占有率，加上 time-of-day、day-of-week 等周期性协变量，模型主要学习常规时空相关性和周期性。

事件/事故交通预测是在这个基础上显式建模非周期扰动。核心差别不是输出形状，而是样本条件和信息来源：

- 事件会造成非平稳分布漂移，历史交通曲线本身不一定能解释突发拥堵或恢复过程。
- 输入需要携带事故或事件上下文，例如事件类型、描述、发生位置、与传感器距离、影响区域、是否节假日、道路/传感器属性等。
- 模型评估更关注事件场景下的预测误差，而不是只在全量常规窗口上平均。
- “没有事件”本身也是一个重要条件，用来对比事件模块是否真的贡献了信息。

两个仓库对这个任务的落点不同：

| 仓库 | 任务形式 | 与普通交通预测的距离 |
| --- | --- | --- |
| `ConFormer` | 事故知情交通预测。输入仍是普通窗口 `(B,T,N,C)`，但额外用事故/区域二值通道生成条件 embedding，再用图传播和 GLN 调制 Transformer。 | 更接近普通 TSF，只是在窗口特征里加入 accident/region 条件。 |
| `IGSTGNN` | incident-guided forecasting。batch 里除了 `x_data/y_data`，还应有事件属性、事件位置、事件-传感器距离、传感器元信息。 | 和普通 TSF 差异更大，数据集和 runner 都需要支持结构化事件字段。 |

## 2. 代码入口与数据流

### ConFormer

入口是 `reproduction/ConFormer/model/train.py`。训练循环是手写 PyTorch，不经过 BasicTS：

- `get_dataloaders_from_index_data(...)` 读取 `data.npz`/`index.npz`，返回 `TensorDataset` 和 `DataLoader`。
- batch 形态是 `(x_batch, y_batch)`，其中 `x` 为 `(B, in_steps, N, C)`，`y` 为 `(B, out_steps, N, 1)`。
- scaler 只在训练集 `x_train[..., 0]` 上拟合，并只标准化输入第 0 个通道；`y` 保留原始尺度。
- 训练时 `model(x_batch)` 输出先 inverse transform，再和原始 `y_batch` 算 masked MAE。

数据 loader 里当前只稳定生成三类特征：traffic、time-of-day、day-of-week。`external.npz` 相关加载被注释掉，因此 accident/region 通道不会从 fallback 逻辑中生成。

### IGSTGNN

入口是 `reproduction/IGSTGNN/experiments/IGSTGNN/main.py`。它同样是独立 PyTorch 框架，不经过 BasicTS：

- `load_dataset(...)` 读取 `incident_data_train.npy`、`incident_data_val.npy`、`incident_data_test.npy`。
- 每个样本应包含 `x_data`、`y_data`、`event_features`、`event_position`、`event_distances`、`durations`。
- 如果打开 `--incident`，loader 应 yield 一个事件 batch 字典；否则 yield `(x, y)`。
- engine 在训练时尝试构造 `incident_data` 并传给 `model(X, label, incident_data=..., sensor_data=...)`。

当前 loader 和 engine 的事件字段命名不一致：loader 产出 `event_features/event_position/event_distances`，engine 读取 `incident_features/incident_position/incident_distances`。这是复现前必须修的第一类硬断点。

### BasicTS 0.5.8

BasicTS 的标准 TSF 流程由 config 驱动：

- dataset 默认读取 `datasets/<DATA_NAME>/data.dat` 和 `desc.json`。
- `TimeSeriesForecastingDataset.__getitem__` 返回 `{'inputs': history_data, 'target': future_data}`。
- `SimpleTimeSeriesForecastingRunner.forward(...)` 负责 scaler、feature selection、调用模型、postprocess 和指标。
- 模型标准签名是 `model(history_data=..., future_data=..., batch_seen=..., epoch=..., train=...)`。

这意味着 ConFormer 可以用较薄的 wrapper 接进 BasicTS；IGSTGNN 需要自定义 dataset 和 runner，因为事件字段不属于 BasicTS 默认 `inputs/target` 二元结构。

## 3. 实际生效模块

### ConFormer 主路径

只看当前 forward 主路径，实际生效模块是：

1. `input_proj`：把历史交通输入 `x[..., :input_dim]` 展平成每个节点的历史表示。
2. `tod_embedding` / `dow_embedding`：如果对应维度大于 0，从最后一个历史步的时间通道取 embedding。
3. `node_embedding`：节点静态 embedding。
4. `acc_embedding` / `reg_embedding`：如果启用，从 `x[..., 3]` 和 `x[..., 4]` 聚合成二值条件。
5. `graph_propagate`：对条件特征做 GCN/自适应图传播，得到 GLN 的条件 `c`。
6. `SelfAttentionLayer`：每层用 `GLN(c)` 生成 shift/scale/gate，调制 attention 和 MLP。
7. `output_proj`：输出 `(B, out_steps, N, output_dim)`。

当前可疑或会断的点：

- `ConFormer.yaml` 中有 `adaptive_embedding_dim`，但 `ConFormer.__init__` 没有这个参数；更像是应映射到 `node_embedding_dim`。
- 配置启用了 `tod/dow/acc/reg` embedding，但 dataloader fallback 只会生成 traffic/tod/dow；如果真实 `data.npz` 没有第 3/4 通道，`acc_embedding` 和 `reg_embedding` 会索引越界。
- `supports` 默认值是整数 `1`，但 `GCN.__init__` 调用 `len(supports)`；如果配置没有传入邻接矩阵 list，会直接报错。
- `external.npz`、`dom`、`use_mixed_proj` 等路径当前没有形成有效训练路径。
- `SelfAttentionLayer` 的 `mask` 参数被传给 `AttentionLayer` 的 `qkv_bias` 位置；默认值下影响不大，但语义是错位的。

结论：ConFormer 的核心思想在代码里是“事故/区域条件特征 -> 图传播 -> GLN 调制 Transformer”。但复现前要先确认真实数据是否包含 5 通道，或者先关闭 accident/region embedding 做普通窗口 smoke test。

### IGSTGNN 主路径

只看当前 forward 主路径，实际生效模块是：

1. `_prepare_inputs`：假设输入前三个通道为 traffic、time-in-day、day-in-week，其中 `num_feat` 个通道作为主交通特征，后两个通道查时间 embedding。
2. `embedding`：把主交通特征投影到 hidden dim。
3. `IncidentsIcsfModule`：如果传入 `incident_data`，嵌入事件属性、位置、时间、距离和传感器信息，并改写最后一个历史步的节点表示。
4. `DynamicGraphConstructor`：从历史 hidden、时间 embedding 和节点 embedding 构造动态邻接。
5. 静态图构造：由两组 node embedding 生成静态邻接。
6. `DecoupleLayer` 堆叠：每层包含 `EstimationGate`、`DifBlock` 和 `InhBlock`。
7. `STLocalizedConv`：在动态图和静态图上做局部时空卷积。
8. `DifForecast` / `InhForecast`：分别生成差异/固有分支的预测 hidden。
9. `_apply_incident_decay`：试图对多步预测施加 temporal incident impact decay。

当前可疑或会断的点：

- `main.py` 导入 `src.models.igstgnn`，但文件名是 `IGSTGNN.py`；在大小写敏感文件系统会失败。
- loader 产出 `event_*` 键，engine 读取 `incident_*` 键。
- `load_dataset(... incidents_sensor=args.use_sensor_info)` 传入的是 bool，但 `IncidentDataLoader` 后续把它当成包含 `sensor_type/surface/...` 的 dict 使用。
- `IncidentsIcsfModule` 初始化时固定 `use_sensor_info=True`，且在无 `sensor_data` 分支后仍使用 `sensor_embed`，会出现未定义变量或维度不一致。
- `forward` 在 `incident_data is None` 时不会定义 `incident_effect`，但后面无条件调用 `_apply_incident_decay(...)`。
- `_apply_incident_decay` 里 `incident_trans1` 把 incident effect 从 hidden dim 投到 `gap`，然后试图加到 forecast hidden。第一次调用时 `forecast_hidden` 最后一维是 256，`gap` 默认是 3，会形状不匹配。
- `lambda_incident`、`durations`、`spatial_attention` 目前没有真正参与预测结果。
- `model_args['use_pre'] = False` 被强制写死，因此预定义路网不直接作为 support 使用；预定义 adjacency 主要用于动态图 mask。

结论：IGSTGNN 的论文模块意图更完整，但当前代码主路径有多个硬断点。复现它需要先做“最小可跑修复”，然后再判断 ICSF/TIID 是否按预期贡献性能。

## 4. 和 BasicTS 训练框架的差异

| 维度 | ConFormer / IGSTGNN 当前代码 | BasicTS 0.5.8 |
| --- | --- | --- |
| 配置 | `argparse` 或 yaml + 手写脚本 | EasyDict config，`CFG.RUNNER` / `CFG.DATASET` / `CFG.MODEL` |
| 数据格式 | ConFormer: `(x,y)`；IGSTGNN: 事件样本 dict 或 `(x,y)` | 默认 `{'inputs': ..., 'target': ...}` |
| scaler | 在训练脚本/engine 内手动 inverse 后算 loss | runner 里统一 preprocessing/postprocessing |
| 模型签名 | ConFormer: `forward(x)`；IGSTGNN: `forward(history_data, label=None, incident_data=None, sensor_data=None)` | `forward(history_data, future_data, batch_seen, epoch, train)` |
| loss/metrics | 手动 masked MAE/MAPE/RMSE | config 里注册 loss/metrics，并自动做 train/val/test meter |
| curriculum learning | IGSTGNN 用 iteration 级 warm/cl step | BasicTS TSF runner 支持 epoch 级 CL |
| checkpoint/log | 各自手写保存路径 | BasicTS/EasyTorch 统一 checkpoint、日志、TensorBoard |
| 事件字段 | IGSTGNN 需要额外事件/距离/传感器字段 | 默认 runner 不会传这些字段，需要自定义 runner |

迁移判断：

- ConFormer 优先用 wrapper 接入 BasicTS：dataset 仍可返回 `inputs/target`，模型 wrapper 把 `history_data` 转给 `ConFormer.forward(x)` 即可。
- IGSTGNN 优先写自定义 `IncidentForecastingDataset` 和 `IncidentForecastingRunner`：dataset 返回 `inputs/target/incident_data/sensor_data`，runner 在 `forward` 中把额外字段送入模型。
- scaler 要统一策略。更建议 BasicTS 里只规范化 target channel，并保留事件/时间/类别字段不被 z-score 破坏。

## 5. 建议复现顺序

### Phase 0：静态 smoke test

ConFormer：

1. 修正配置参数名：确认 `adaptive_embedding_dim` 是否应为 `node_embedding_dim`。
2. 确认 `data.npz` 的通道数；如果只有三通道，先关闭 `acc_embedding_dim/reg_embedding_dim`。
3. 明确 `supports`：要么加载 `adj.npy` 为 list，要么设置空 list 并只用自适应图。
4. 用一个 batch 跑通 `model(x)`、inverse transform、masked MAE。

IGSTGNN：

1. 修正导入大小写或增加 lower-case alias。
2. 统一 loader 和 engine 的事件键名。
3. 明确 sensor metadata 的加载方式；没有 sensor 时要让 ICSF 走无传感器路径。
4. 修 `_apply_incident_decay` 的维度，让 hidden-level decay 和 output-level decay 分开，或者先只保留一个可解释的 decay 点。
5. 用一个 batch 跑通 `model(X, incident_data=...)` 和 loss。

### Phase 1：保留原框架的 faithful reproduction

先不要急着移植到 BasicTS。对两个 reproduction 目录做最小补丁，让原始训练脚本能单数据集跑通，并记录每个补丁原因。这样能区分“论文复现问题”和“BasicTS 移植问题”。

### Phase 2：模块消融

ConFormer 优先消融：

- `acc_embedding/reg_embedding`
- `graph_propagate`
- `GLN`
- `tod/dow`
- `node_embedding`

IGSTGNN 优先消融：

- `incident_data` on/off
- sensor metadata on/off
- ICSF on/off
- temporal decay on/off
- dynamic graph vs static graph
- Dif/Inh branches

### Phase 3：BasicTS 化

建议先迁移 ConFormer，再迁移 IGSTGNN：

1. 在 BasicTS 中增加 ConFormer wrapper 和 config，保持 BasicTS 标准 TSF runner。
2. 为事件数据增加 dataset/scaler 约定，明确哪些通道可归一化，哪些字段是类别或距离。
3. 增加 IGSTGNN 专用 runner，把事件字段纳入 `forward_return` 或 model kwargs。
4. 统一 metrics/horizon evaluation/checkpoint，让两个方法可以和 BasicTS 里已有基线横向比较。

## 6. 当前结论

三个目标的当前答案：

1. 任务类型不是换一个模型做普通交通预测，而是事件条件下的交通预测。ConFormer 把事件条件压成窗口通道，IGSTGNN 把事件作为结构化 batch 上下文。
2. ConFormer 的有效核心是 accident/region/time/node 条件 embedding + graph propagation + GLN Transformer；IGSTGNN 的有效核心应是 ICSF + 动静态图 ST backbone + incident decay，但当前实现有硬断点，需要先修。
3. BasicTS 的主框架优势是统一配置、数据、scaler、runner、checkpoint 和 metrics；外部两库的差异主要集中在模型 forward 签名、事件字段 batch 结构、scaler 策略和 curriculum learning 粒度。

