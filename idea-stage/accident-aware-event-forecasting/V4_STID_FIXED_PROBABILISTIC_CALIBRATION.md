# V4 STID-Fixed Probabilistic Calibration

**Status**: protocol proposal, pilot script pending server run  
**Goal**: test whether incident information improves probabilistic forecasts
when the STID mean forecast is fixed.  
**Hard baseline**: pure `STID` mean + traffic/time uncertainty calibration.

## Motivation

Prior experiments do not support a broad claim that accident labels improve
mean MAE/RMSE. The more plausible claim is probabilistic:

```text
Incident information may not move the conditional mean reliably, but it may
change predictive uncertainty, interval coverage, and tail risk.
```

So the mean stays:

$$
\mu_{r,h,i}
=
\hat{Y}^{\mathrm{STID}}_{r,h,i}
$$

and the new module predicts a distribution around this mean:

$$
Y_{r,h,i}
\mid
X_r,\mathcal{I}_r
\sim
\mathcal{N}
\left(
\mu_{r,h,i},
\sigma^2_{r,h,i}
\right)
$$

or equivalently a set of quantiles:

$$
Q_{\tau,r,h,i}
=
\mu_{r,h,i}
+
z_{\tau}\sigma_{r,h,i}
$$

where \(z_{\tau}\) is the standard normal \(\tau\)-quantile.

## Prediction Target

The fixed-mean residual is:

$$
\epsilon_{r,h,i}
=
Y_{r,h,i}
-
\hat{Y}^{\mathrm{STID}}_{r,h,i}
$$

The probabilistic calibrator predicts:

$$
\log \sigma_{r,h,i}
=
f_{\theta}
\left(
\phi_{\mathrm{traffic}}(r,i),
\phi_{\mathrm{time}}(r),
\phi_{\mathrm{incident}}(r,i),
h
\right)
$$

This is deliberately not a residual mean corrector.

## Baselines

| Model | Features | Purpose |
| --- | --- | --- |
| `constant_sigma` | county + horizon | no event / traffic conditioning |
| `traffic_time_sigma` | history traffic stats + TOD/DOW + horizon | strong non-incident uncertainty baseline |
| `incident_field_sigma` | incident type / spatial / temporal field + horizon | incident-only ablation |
| `full_sigma` | traffic/time + incident field + horizon | tests incident incremental value |

The decisive comparison is:

$$
\mathrm{full\_sigma}
\quad
\text{vs.}
\quad
\mathrm{traffic\_time\_sigma}
$$

not `full_sigma` vs `constant_sigma`.

## Training Loss

Use Gaussian negative log likelihood with fixed mean:

$$
\mathcal{L}_{\mathrm{NLL}}
=
\frac{1}{2}
\left(
\frac{\epsilon_{r,h,i}}{\sigma_{r,h,i}}
\right)^2
+
\log\sigma_{r,h,i}
+
\frac{1}{2}\log(2\pi)
$$

This rewards both sharpness and calibration.

## Quantile Evaluation

From Gaussian scale:

$$
Q_{\tau,r,h,i}
=
\mu_{r,h,i}
+
z_{\tau}\sigma_{r,h,i}
$$

Evaluate pinball loss:

$$
\rho_{\tau}(y-Q_{\tau})
=
\max
\left(
\tau(y-Q_{\tau}),
(\tau-1)(y-Q_{\tau})
\right)
$$

Lower is better.

## Interval Evaluation

For a central \(1-\alpha\) interval:

$$
L_{\alpha/2}
=
\mu
+
z_{\alpha/2}\sigma
$$

$$
U_{1-\alpha/2}
=
\mu
+
z_{1-\alpha/2}\sigma
$$

Coverage:

$$
\mathrm{Coverage}_{1-\alpha}
=
\mathbb{E}
\left[
\mathbb{I}
\left(
L_{\alpha/2}
\le
Y
\le
U_{1-\alpha/2}
\right)
\right]
$$

Width:

$$
\mathrm{Width}_{1-\alpha}
=
\mathbb{E}
\left[
U_{1-\alpha/2}
-
L_{\alpha/2}
\right]
$$

A useful model should improve calibration without simply making intervals very
wide.

## Event-Sensitive Slices

Report metrics on:

- `all_eval_sample`;
- `no_event_sample`;
- `future_any`;
- `future_onset`;
- `ongoing`;
- `post_last_slot`;
- `UnknInj/future_any`;
- `UnknInj/future_onset`;
- `UnknInj/ongoing`;
- `UnknInj/post_last_slot`.

The important question is:

$$
\Delta_{\mathrm{incident}}
=
\mathrm{Metric}(\mathrm{full\_sigma})
-
\mathrm{Metric}(\mathrm{traffic\_time\_sigma})
$$

where lower is better for NLL and pinball, and coverage error should decrease
without excessive width inflation.

## Expected Outcomes

Positive evidence would be:

$$
\mathrm{NLL}_{\mathrm{full}}
<
\mathrm{NLL}_{\mathrm{traffic}}
$$

and:

$$
\mathrm{Pinball}_{\mathrm{full}}
<
\mathrm{Pinball}_{\mathrm{traffic}}
$$

especially on `UnknInj` slices, with reasonable coverage.

Negative evidence would be:

```text
full_sigma only matches traffic_time_sigma, while incident_field_sigma is weak.
```

That would mean accident labels still do not add enough information beyond
traffic state, even in probabilistic forecasting.

## Non-Claims

- This does not claim mean MAE improvement.
- This does not claim causal accident effect.
- This does not claim a fully deployable probabilistic model until the
  `history` scope is positive.
- `history_future` is an oracle diagnostic and must be labeled as such.

## Pilot Script

The low-cost pilot should live at:

```text
reproduction/analysis/traffident_probabilistic_calibration_pilot.py
```

It should use saved pure-STID predictions and targets, fit on the first half of
the test result, and evaluate on the second half, matching the previous
post-hoc pilots.
