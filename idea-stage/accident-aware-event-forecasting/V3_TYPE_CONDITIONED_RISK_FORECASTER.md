# V3 Type-Conditioned Incident Risk Forecaster

**Status**: ARIS method proposal, not implemented yet  
**Goal**: use incident information to predict risk / uncertainty under
incidents, rather than forcing a mean residual correction.  
**Hard baseline**: pure `STID`.

## Target

Previous pilots show that accident-aware mean correction is unstable:

- `DecayKernel` is worse than pure `STID`;
- `BiasOnly` is safer than the decay kernel but still weak;
- `SignReliability` collapses to a no-op under validation;
- matched-control type slicing shows that only `UnknInj` consistently raises
  STID MAE and tail90 risk.

Therefore the next target should not be:

$$
\hat{Y}_{h,i}
=
\hat{Y}^{0}_{h,i}
+
\Delta_{h,i}
$$

The next target is incident-conditioned risk:

$$
\pi_{r,i}
=
P
\left(
E_{r,i} > q^{(c)}_{0.9}
\mid
X_r,\mathcal{I}_r
\right)
$$

where \(E_{r,i}\) is the STID error for sample window \(r\), node \(i\), and
\(q^{(c)}_{0.9}\) is the county-specific no-event tail threshold.

## Status

COHERENT AFTER REFRAMING.

The object changes from conditional mean forecasting to conditional tail-risk
forecasting. This is supported by the matched-control audit: incident labels
change the error distribution more consistently than they improve mean
prediction.

## Invariant Object

The invariant object is the conditional risk of STID failure:

$$
T_{r,i}
=
\mathbb{I}
\left(
E_{r,i} > q^{(c)}_{0.9}
\right)
$$

with:

$$
E_{r,i}
=
\frac{1}{H}
\sum_{h=1}^{H}
\left|
Y_{r,h,i}
-
\hat{Y}^{0}_{r,h,i}
\right|
$$

The model predicts \(T_{r,i}\) or a calibrated proxy for it. It does not need to
change the STID mean forecast in the first version.

## Assumptions

- A pure STID model provides the normal-traffic mean prediction
  \(\hat{Y}^{0}\).
- No-event controls define the reference error distribution for each county.
- Incident type matters: `UnknInj`, `NoInj`, and `1141` should not share one
  undifferentiated accident embedding.
- Incident spatial / temporal kernels are candidate support priors, not direct
  residual magnitudes.
- The first deployable model should avoid future traffic leakage. Future
  incident labels are allowed only for oracle analysis or explicitly marked
  oracle variants.

## Notation

- \(r\): prediction sample window.
- \(i\): target sensor node.
- \(h \in \{1,\ldots,H\}\): forecast horizon.
- \(c\): county.
- \(m\): incident record.
- \(\mathcal{I}_r\): incident records visible to sample \(r\).
- \(\hat{Y}^{0}_{r,h,i}\): pure STID prediction.
- \(Y_{r,h,i}\): ground-truth traffic target.
- \(E_{r,i}\): mean absolute STID error across horizons.
- \(T_{r,i}\): tail90 failure label.
- \(q^{(c)}_{0.9}\): 90th percentile of \(E_{r,i}\) over no-event controls.

## Tail Label

For each county \(c\), compute:

$$
q^{(c)}_{0.9}
=
\mathrm{Quantile}_{0.9}
\left(
\left\{
E_{r,i}
:
(r,i)\in\mathcal{C}^{(c)}_{\mathrm{no\text{-}event}}
\right\}
\right)
$$

Then define:

$$
T_{r,i}
=
\mathbb{I}
\left(
E_{r,i} > q^{(c)}_{0.9}
\right)
$$

This gives a binary target for whether STID enters its no-event top-10% error
tail.

## Type-Conditioned Incident Field

For each candidate incident \(m\), construct:

$$
z_{h,i,m}
=
\left[
e_{\mathrm{type}}(c_m),
\phi_s(s_{i,m}),
\phi_t(\delta_{h,m})
\right]
$$

where:

$$
\delta_{h,m}=t_{r,h}-o_m
$$

\(o_m\) is incident onset / report time, \(s_{i,m}\) is the incident-node
spatial relation, and \(c_m\) is the incident type.

Use the spatial-temporal kernel only as a support prior:

$$
K_{h,i,m}
=
\exp
\left(
-\frac{d_{i,m}}{\lambda_s}
\right)
\cdot
\left[
\mathbb{I}(\delta_{h,m}\ge0)
\exp
\left(
-\frac{\delta_{h,m}}{\lambda_t^+}
\right)
+
\mathbb{I}(\delta_{h,m}<0)
\exp
\left(
-\frac{-\delta_{h,m}}{\lambda_t^-}
\right)
\right]
$$

Aggregate:

$$
F_{r,h,i}
=
\sum_{m\in\mathcal{I}_r}
K_{h,i,m}
\psi(z_{h,i,m})
$$

The type prior should be initialized or regularized according to the audit:

$$
b_{\mathrm{UnknInj}}>0,\quad
b_{\mathrm{NoInj}}\approx 0,\quad
b_{1141}\le 0
$$

This is not a hard rule. It is a weak prior reflecting the observed tail-risk
direction.

## Risk Head

Let \(Z_{r,i}\) be the STID hidden state or a frozen summary of recent traffic.
The risk head predicts:

$$
\pi_{r,i}
=
\sigma
\left(
f_{\theta}
\left(
Z_{r,i},
\mathrm{Pool}_{h}(F_{r,h,i}),
\eta_{r,i}
\right)
\right)
$$

where \(\eta_{r,i}\) can include time-of-day, day-of-week, county, and normal
traffic volatility features.

The primary loss is weighted binary cross entropy:

$$
\mathcal{L}_{\mathrm{risk}}
=
-
\alpha T_{r,i}\log \pi_{r,i}
-
(1-T_{r,i})\log(1-\pi_{r,i})
$$

Because tail90 positives are sparse, \(\alpha\) should be chosen from the
training positive rate or replaced by focal loss:

$$
\mathcal{L}_{\mathrm{focal}}
=
-
\alpha
T_{r,i}
(1-\pi_{r,i})^\gamma
\log \pi_{r,i}
-
(1-T_{r,i})
\pi_{r,i}^{\gamma}
\log(1-\pi_{r,i})
$$

## Optional Uncertainty Head

A second head can predict uncertainty inflation instead of changing the mean:

$$
\log \sigma_{r,i}
=
\log \sigma^{0}_{r,i}
+
g_{\theta}
\left(
Z_{r,i},
\mathrm{Pool}_{h}(F_{r,h,i})
\right)
$$

where \(\sigma^{0}_{r,i}\) is the no-event calibrated STID error scale. A
prediction interval is:

$$
\left[
\hat{Y}^{0}_{r,h,i}
-
z_{\alpha}\sigma_{r,i},
\hat{Y}^{0}_{r,h,i}
+
z_{\alpha}\sigma_{r,i}
\right]
$$

This version should be evaluated by coverage, interval width, ECE, Brier score,
AUROC, and AUPRC, not only MAE.

## No-Mean-Correction Rule

The first version should keep:

$$
\hat{Y}_{r,h,i}
=
\hat{Y}^{0}_{r,h,i}
$$

and only output:

$$
\pi_{r,i},\quad \sigma_{r,i}
$$

Mean correction may be reintroduced only after risk calibration is positive:

$$
\hat{Y}_{r,h,i}
=
\hat{Y}^{0}_{r,h,i}
+
\mathbb{I}(\pi_{r,i}>\tau)
\Delta_{r,h,i}
$$

This prevents the module from repeating the failure mode of previous residual
pilots.

## Architecture

```mermaid
flowchart LR
    X["traffic history"] --> STID["frozen or warm-start STID"]
    STID --> Y0["mean forecast Y0"]
    STID --> Z["traffic state Z"]

    INC["incident records"] --> TYPE["type encoder"]
    INC --> SPACE["space relation encoder"]
    INC --> TIME["relative time encoder"]

    TYPE --> FIELD["type-conditioned incident field"]
    SPACE --> FIELD
    TIME --> FIELD
    FIELD --> POOL["node/window pooling"]

    Z --> RISK["tail-risk head"]
    POOL --> RISK
    RISK --> PI["P tail90"]

    Z --> UNC["uncertainty head"]
    POOL --> UNC
    UNC --> SIGMA["sigma inflation"]

    Y0 --> OUT["mean kept unchanged"]
    PI --> OUT
    SIGMA --> OUT
```

## Pilot Plan

1. Build a post-hoc risk dataset from saved pure-STID `test_results`.
2. Train a lightweight classifier for \(T_{r,i}\) using type-conditioned event
   field features plus normal traffic volatility features.
3. Compare against:
   - no-event base rate;
   - county base rate;
   - type-only logistic regression;
   - decay-kernel score;
   - STID residual magnitude proxy.
4. Report AUROC, AUPRC, Brier, ECE, and recall at fixed false-positive rate.
5. Only if calibration is positive, implement a BasicTS module.

## Success Criteria

Minimum success:

$$
\mathrm{AUPRC}_{\mathrm{type+risk}}
>
\mathrm{AUPRC}_{\mathrm{county\ base}}
$$

and:

$$
\mathrm{ECE}_{\mathrm{type+risk}}
<
\mathrm{ECE}_{\mathrm{type\ only}}
$$

Stronger success:

$$
P(T=1\mid\pi>\tau)
-
P(T=1)
>0
$$

on `UnknInj` event slices across at least three counties.

## Non-Claims

- This does not claim to improve mean MAE.
- This does not claim a causal accident effect.
- This does not claim `1141` is harmless; it only says `1141` does not behave
  like the current positive tail-risk class under the four-county Q1 audit.
- This does not yet solve incident-node mapping. Spatial relation quality still
  matters for any deployable model.

## Decision

The next low-cost experiment should be a post-hoc type-conditioned tail-risk
pilot. If that pilot fails, the accident labels in the current benchmark are
probably insufficient for a robust forecasting improvement claim, and the main
paper direction should return to dataset analysis or stronger event alignment.
