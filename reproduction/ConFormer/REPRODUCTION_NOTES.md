# ConFormer Reproduction Notes

This repo keeps the official ConFormer entrypoint:

```bash
cd reproduction/ConFormer/model
python train.py -d <dataset> -g <gpu_id>
```

## BA/SD Settings

Paper-aligned settings used for BA and SD:

- input length 12, prediction length 12
- 15 minute sampling interval, so `steps_per_day=96`
- chronological 6:2:2 split through `index.npz`
- US Accidents enabled through `acc_embedding_dim`
- regulation embedding disabled because the paper marks BA/SD regulation as unavailable
- z-score normalization on the training traffic channel, using the released loader style
- masked RMSE/MAE/MAPE through the official metric functions

AI-supplemented settings:

- The official repository only ships a TKY YAML block, so BA/SD YAML blocks are added here.
- `num_layers` and `feed_forward_dim` are inferred to match the paper-reported parameter counts under the released model code:
  - SD: `num_layers=2`, `feed_forward_dim=331`
  - BA: `num_layers=2`, `feed_forward_dim=239`
  - Expected trainable parameters under the released code are approximately SD 757,842 and BA 783,850.
- BA `data.npz` has traffic, time-of-day, and day-of-week channels but no accident channel. When accident embedding is enabled, the loader appends `accident.h5` in memory as channel 3.

These supplements are intentionally minimal: they keep the official `train.py` path and only fill the missing BA/SD configuration and accident-channel bridge needed by the released data layout.
