# Post-Hoc Residual Pilot

**Last update**: 2026-05-28 14:35 Asia/Shanghai
**Script**: `reproduction/analysis/traffident_posthoc_residual_pilot.py`
**Output**: `reproduction/analysis/traffident_posthoc_residual_pilot/`

## Status

Negative.

This pilot does not justify implementing the current V1 residual router inside
BasicTS yet.

## Protocol

This is an exploratory calibration/holdout pilot, not an official test result:

- source: saved `STID` `test_results`;
- calibration: first half of each county test split;
- evaluation: second half of each county test split;
- model: pooled weighted ridge residual over four counties;
- correction target: `target - STID_prediction`;
- inference features: incident type, downstream/upstream relation, distance,
  event age, pre-state statistics, horizon, county id;
- matched-control impact: used only for calibration weights and evaluation
  groups, not as an inference feature.

Reason for this protocol: saved STID train/val predictions are not currently
available. Therefore this run is only a low-cost sanity check.

## Summary

Four-county mean results on the evaluation half:

| Group | STID MAE | Posthoc MAE | Delta | Wins |
| --- | ---: | ---: | ---: | ---: |
| all_eval | 10.7641 | 10.7664 | +0.0023 | 0/4 |
| no_event | 10.7570 | 10.7570 | +0.0000 | 0/4 |
| history_any | 12.5802 | 13.6237 | +1.0435 | 0/4 |
| post_last_slot | 12.8312 | 13.9199 | +1.0887 | 0/4 |
| ongoing | 13.1748 | 14.1742 | +0.9994 | 0/4 |
| high_impact_history_any | 24.0563 | 24.4282 | +0.3719 | 1/4 |
| high_impact_post_last | 24.3922 | 24.7449 | +0.3527 | 0/4 |
| impact_drop_history_any | 13.9882 | 14.8783 | +0.8901 | 0/4 |
| impact_rise_history_any | 14.1373 | 14.6746 | +0.5373 | 0/4 |

Interpretation:

- the linear post-hoc residual corrector does not beat pure STID on any key
  event group;
- no-event windows are unchanged, as intended, because correction is only
  applied to history-event windows;
- the correction worsens observed-event windows substantially, especially
  `history_any`, `post_last_slot`, and `ongoing`;
- high-impact windows remain the main unsolved target.

## Per-County Notes

Only one useful sign appears: Alameda high-impact history windows improve
slightly (`delta=-0.0334`). This is too weak and isolated to justify full model
implementation.

Examples of failures:

- LosAngeles `post_last_slot`: `+0.9486` MAE worse than STID;
- Orange `post_last_slot`: `+1.0959`;
- Alameda `ongoing`: `+0.9265`;
- ContraCosta `high_impact_post_last`: `+0.8060`.

## Decision

Do not implement this exact V1 as a BasicTS neural module yet.

The current feature set and linear residual objective are not enough. The next
step should be one of:

1. generate proper STID train/val predictions and repeat the pilot without test
   calibration;
2. change the target from raw residual to residual sign/magnitude with stronger
   shrinkage;
3. restrict correction to a smaller high-confidence subset, e.g. `1141` +
   downstream + high-impact;
4. add a learned no-change gate with a strong penalty for worsening STID.

The main standard remains unchanged: a candidate must beat pure `STID` on
observed-event or high-impact windows before full training is justified.
