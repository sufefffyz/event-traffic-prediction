# V1 Module Design: Directional Impact Router

**Last update**: 2026-05-28 13:35 Asia/Shanghai
**Status**: design only, not implemented; first post-hoc ridge residual pilot is negative
**Primary baseline**: pure `STID`

## Objective

V0 `STIDGatedAccident` proves that sparse gating is safer than direct accident
embedding, but it does not robustly beat pure `STID`. V1 should target the real
failure mode found in `EVENT_FACTOR_ANALYSIS.md`:

- high-impact event windows have large error and do not consistently improve;
- drop windows are often overestimated;
- rise/recovery windows are often underestimated;
- useful gains concentrate in specific incident types and downstream/ongoing
  geometry.

The target is not generic accident fusion. The target is:

```text
MAE(V1) - MAE(STID) < 0
```

on observed-event or matched-impact windows, while keeping no-event windows near
STID.

## No-Leakage Rule

Matched-control impact is a training/evaluation label, not an inference input.

Allowed inference inputs:

- traffic history;
- time-of-day/day-of-week;
- node identity;
- accident records already observed in the history window;
- incident type, distance, signed post-mile relation, and event age for observed
  incidents;
- horizon index.

Disallowed inference inputs:

- future target traffic;
- matched-control future change;
- future accident onset unless a separate future-known-event setting is declared.

## Model Decomposition

Use a frozen or strongly regularized STID base:

```text
Y_base = STID_frozen(X)
Y_hat = Y_base + Delta_event
```

The residual branch is only responsible for event correction:

```text
Delta_event[i,k] =
    M_local[i,k] * s_impact[i,k] * g[i,k] * r_dir[i,k]
```

where:

- `M_local[i,k]`: deterministic local mask from observed incident geometry;
- `s_impact[i,k]`: predicted high-impact score from history/meta only;
- `g[i,k]`: learned residual gate;
- `r_dir[i,k]`: directional residual, separated into drop/rise/neutral modes.

## Feature Construction

For each sample, node `i`, horizon `k`, construct:

```text
phi[i,k] = concat(
    h_stid[i],
    Emb(type_i),
    Emb(relation_i),
    signed_downstream_pm_i,
    distance_pm_i,
    event_age_slots_i,
    pre_last_i,
    pre_mean_i,
    pre_slope_i,
    horizon_emb(k)
)
```

`relation_i` uses:

```text
signed_downstream_pm =
    (sensor_abs_pm - incident_abs_pm) * sign(Direction)

sign(Direction) = +1 for N/E, -1 for S/W
```

Buckets:

```text
downstream: signed_downstream_pm > 0.025
upstream:   signed_downstream_pm < -0.025
at_source:  |signed_downstream_pm| <= 0.025
```

## Router Heads

Impact predictor:

```text
s_impact[i,k] = sigmoid(MLP_impact(phi[i,k]))
```

Gate:

```text
g[i,k] = sigmoid(MLP_gate(phi[i,k]))
```

Direction mixture:

```text
pi_drop, pi_rise, pi_neutral = softmax(MLP_dir(phi[i,k]))
```

Residual heads:

```text
r_dir[i,k] =
    pi_drop    * r_drop(phi[i,k])
  + pi_rise    * r_rise(phi[i,k])
  + pi_neutral * r_neutral(phi[i,k])
```

Final:

```text
Y_hat[i,k] =
    Y_base[i,k] + M_local[i,k] * s_impact[i,k] * g[i,k] * r_dir[i,k]
```

## Local Mask

V1 should not use a single history accident binary. Use node-horizon geometry:

```text
M_local[i,k] =
    1[event observed in history for node i]
  * 1[distance_pm_i <= d_max]
  * K_relation(relation_i, k)
```

Initial safe rule:

```text
K_relation(downstream, k) = 1
K_relation(at_source, k)  = 1
K_relation(upstream, k)   = 1[k <= 3]
```

Reason: previous slices show downstream/ongoing is more promising, while
upstream/post-history often hurts STID.

## Training Loss

Use saved or frozen STID predictions as the baseline teacher. This keeps the
comparison target explicit.

```text
L = L_all
  + lambda_event * L_event
  + lambda_high  * L_high_impact
  + lambda_dir   * L_direction
  + lambda_noevent * max(0, MAE_noevent(V1) - MAE_noevent(STID) + margin)
```

Definitions:

```text
L_all = masked_mae(Y_hat, Y)

L_event = masked_mae(Y_hat[event_observed], Y[event_observed])

L_high_impact =
    weighted_mae(Y_hat, Y, weight = 1 + alpha * 1[abs(matched_control_change) > q75])

L_direction =
    CE(pi_direction, sign(matched_control_change))
```

The matched-control labels are only used in training loss and analysis. They
are not passed to the model as features.

## Evaluation Gate

V1 is only considered worth training longer if a low-cost pilot beats STID on
at least one of these:

| Slice | Required signal |
| --- | --- |
| `post_last_slot` | `delta_vs_STID_MAE < 0` in at least 3/4 counties |
| `ongoing` | `delta_vs_STID_MAE < 0` in at least 3/4 counties |
| high-impact windows | mean `delta_vs_STID_MAE < 0` |
| no-event windows | degradation <= 0.01 MAE |

Beating `STIDAccident` alone is not sufficient.

## Low-Cost Pilot Before Full Training

Before implementing a full BasicTS runner, run a post-hoc residual pilot:

1. Load saved STID predictions and targets.
2. Build `phi[i,k]` from traffic history and matched incident metadata.
3. Train a small ridge/MLP residual model only on train or validation windows.
4. Apply residual to STID test predictions.
5. Evaluate the same event-factor slices.

This directly tests whether the proposed features can correct STID. If the
post-hoc residual cannot beat STID on high-impact/post-last slices, a full
neural implementation is unlikely to help.

Pilot result:

- `POSTHOC_RESIDUAL_PILOT.md` reports a negative result.
- The weighted ridge residual worsened every key four-county mean event group,
  including `post_last_slot`, `ongoing`, and high-impact windows.
- Do not implement this exact V1 until the residual target/gate is revised or
  proper STID train/val predictions are generated.

## Implementation Notes

Minimal BasicTS path if pilot passes:

- keep STID base initialized from a trained checkpoint;
- freeze base for the first V1 run;
- load sidecar event features from `matched_incidents.csv` and
  `sensor_meta_feature.csv`;
- save `test_results` exactly as current BasicTS runs do;
- log all metrics to WandB and write event-factor metrics after test.
