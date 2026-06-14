# Distribution Audit And Causal-Inspired Forecasting Notes

**Date**: 2026-06-14

## Setting

Dataset: `TraffiDent_D5_2023Q1_OfficialAll`.

The audit follows the current project rule: use the official TraffiDent/XTraffic
matching script output, preserve all official incident classes, and expand each
matched incident only to sensors on the same freeway and direction within
0.5 post-mile. Incident duration is used when available.

Window definition:

$$
X_{r} = Y_{t-11:t}, \qquad
Y_{r} = Y_{t+1:t+12}.
$$

Each node-window is assigned to `no_event`, `event_any`, `history_any`,
`future_any`, `future_onset`, `history_only`, `ongoing`, or `post_last_slot`
according to whether the expanded incident mask intersects the history/future
parts of the 12-12 window.

Raw channels:

- flow: channel 0
- occupancy: channel 1
- speed: channel 2

## Data Scale

| Item | Value |
| --- | ---: |
| nodes | 565 |
| timesteps | 25,920 |
| node-windows | 14,631,240 |
| matched unique incidents | 2,168 |
| event slot-node coverage | 0.0030 |
| event_any node-windows | 151,372 |
| no_event node-windows | 14,479,868 |

## Main Distribution Result

Comparison is against `no_event`. SMD is standardized mean difference.

| Slice | Metric | Mean Diff | SMD | KS |
| --- | --- | ---: | ---: | ---: |
| event_any | full_mean_flow | +25.88 | +0.161 | 0.068 |
| event_any | full_mean_occupancy | +0.00959 | +0.182 | 0.106 |
| event_any | full_mean_speed | -1.26 | -0.041 | 0.081 |
| future_onset | full_mean_flow | +32.38 | +0.200 | 0.086 |
| future_onset | full_mean_occupancy | +0.01172 | +0.218 | 0.124 |
| future_onset | full_mean_speed | -1.32 | -0.043 | 0.088 |
| ongoing | full_mean_flow | +10.30 | +0.066 | 0.043 |
| ongoing | full_mean_occupancy | +0.00494 | +0.098 | 0.065 |
| post_last_slot | full_mean_flow | +12.14 | +0.078 | 0.045 |
| post_last_slot | full_mean_occupancy | +0.00557 | +0.109 | 0.069 |

The strongest difference is not the history-to-future transition. It is the
traffic-state level of the whole 12-12 window. Event windows are more likely to
occur under higher flow and higher occupancy, with only weak average speed
shift.

Transition metrics are much weaker:

| Slice | Metric | Mean Diff | SMD | KS |
| --- | --- | ---: | ---: | ---: |
| event_any | future_minus_history_flow | -1.04 | -0.033 | 0.014 |
| event_any | future_minus_history_occupancy | -0.00052 | -0.026 | 0.011 |
| event_any | future_minus_history_speed | +0.025 | +0.007 | 0.022 |
| future_onset | future_minus_history_flow | +0.30 | +0.009 | 0.031 |
| future_onset | future_minus_history_occupancy | +0.00036 | +0.018 | 0.040 |
| future_onset | future_minus_history_speed | -0.124 | -0.033 | 0.036 |

## Interpretation

The audit supports three conclusions:

1. Accident-containing windows are distributionally different from no-event
   windows, especially in occupancy and flow.
2. The difference is mostly a state-selection effect: incidents are observed
   in windows that already look more loaded/congested.
3. A single event-conditioned mean residual is unlikely to work because the
   12-step future-minus-history shift is near zero on average and highly
   overlapping with no-event windows.

This explains why previous mean-residual pilots were unstable. The accident
signal is real, but it is not a simple deterministic correction

$$
\hat{Y} = \hat{Y}^{0} + \Delta(E).
$$

It is closer to a change in the conditional distribution:

$$
P(Y_{t+1:t+12} \mid X_{t-11:t}, E)
\neq
P(Y_{t+1:t+12} \mid X_{t-11:t}),
$$

where the most visible shift is in density, tail risk, and calibration, not
always in the conditional mean.

## Causal-Inspired Forecasting Direction

The cleanest object is not "use accidents as another feature", but a
counterfactual conditional distribution:

$$
\Delta_{r,i,h}
=
Y_{r,i,h}^{\mathrm{event}}
-
Y_{r,i,h}^{\mathrm{no\ event}}.
$$

Because true counterfactuals are unobserved, the practical design should treat
matched controls as noisy support, not ground truth.

### Candidate 1: Incident-Calibrated Probabilistic Forecaster

Freeze or regularize the mean forecaster:

$$
\mu_{r,i,h}=f_{\mathrm{base}}(X)_{r,i,h}.
$$

Use incidents to model scale / quantiles:

$$
\sigma_{r,i,h}
=
\mathrm{softplus}
\left(
g_{\theta}
\left[
z^{\mathrm{traffic}}_{r,i},
z^{\mathrm{event}}_{r,i,h}
\right]
\right).
$$

Primary claim: accident metadata improves NLL, pinball loss, coverage, or
interval width on ongoing/post-event windows while keeping MAE/RMSE unchanged.

This is currently the most evidence-aligned direction.

### Candidate 2: Causal Support Prior + Robust Residual Head

Use distance, direction, duration, type, and relative time to define where an
event could matter:

$$
M_{r,i,h}
=
\mathbb{I}(d_{i,m}<d_0)
\mathbb{I}(\mathrm{dir}_{i}=\mathrm{dir}_{m})
\kappa_{\tau}(t_{r,h}-o_m).
$$

Then learn only a conservative residual inside the support:

$$
\hat{Y}_{r,i,h}
=
\hat{Y}^{0}_{r,i,h}
+
M_{r,i,h}\,a_{r,i,h}\,\mathrm{sign}_{r,i,h}.
$$

The residual target should be loss-aware sign / risk first, not raw magnitude.

### Candidate 3: Representation Balancing For Event Windows

Estimate a propensity score for event exposure:

$$
e_{r,i} = P(M_{r,i}=1 \mid X_{r,i}, \mathrm{time}, \mathrm{road}).
$$

Then train the event module with inverse propensity or overlap weights:

$$
w_{r,i}
=
\frac{M_{r,i}}{e_{r,i}+\epsilon}
+
\frac{1-M_{r,i}}{1-e_{r,i}+\epsilon}.
$$

Purpose: reduce the state-selection bias found in this audit, where event
windows already have higher flow/occupancy.

### Candidate 4: Distributional Contrastive Objective

Instead of forcing point forecasts to differ, make the model learn when two
windows belong to different traffic-state distributions:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{forecast}}
+
\lambda
\mathcal{L}_{\mathrm{dist}}
\left(
h(X,E),
h(X,E=0)
\right).
$$

The contrast should be conditioned on time-of-day, day-of-week, node, and
road segment to avoid learning trivial periodic differences.

## Recommended Next Step

Before another heavy model run, implement a low-cost post-hoc experiment:

1. train a traffic-state propensity model for `event_any` using only pre-window
   traffic/time/road features;
2. compare event vs no-event distributions after propensity matching or
   weighting;
3. if flow/occupancy differences remain, use the residual difference as an
   incident-effect target;
4. if they disappear, shift the claim to accident-conditioned uncertainty /
   risk calibration rather than causal mean improvement.

The next deployable model should therefore be:

```text
STID mean backbone
+ incident-conditioned probabilistic/risk head
+ propensity-balanced training or evaluation
```

not another direct accident embedding into the mean predictor.

