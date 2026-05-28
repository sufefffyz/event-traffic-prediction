# Module Architecture Notes

**Last update**: 2026-05-28  
**Current module**: `STIDGatedAccident`  
**Code**: `BasicTS/baselines/STIDGatedAccident/arch/stid_gated_accident_arch.py`

## Research Bar

Beating `STIDAccident` is not enough. It only proves that sparse gating is less
harmful than direct accident embedding.

The useful bar is:

```text
delta_vs_STID = MAE(model) - MAE(STID) < 0
```

The next module should be considered positive only if it beats pure `STID`
under at least one meaningful setting:

- all-window MAE without degrading normal traffic;
- observed-event windows such as `post_last_slot`, `history_any`, or `ongoing`;
- matched-impact event windows where incidents actually change traffic;
- a clearly declared future-known-event setting, if future incident information
  is provided to the model without leakage.

`future_onset` alone is not enough for the current v0 module, because v0 only
uses accident activity in the input history. If history has no accident, the
event branch is masked to zero and any gain is indirect base-branch retraining,
not true future-event awareness.

## Current V0 Formula

Notation:

- `X_{b,l,i,c}`: input history for batch `b`, lag `l`, node `i`, channel `c`;
- `A_{b,l,i}`: binary accident channel in the input history;
- `H`: forecast horizon length;
- `L`: history length;
- `h_i`: STID hidden state for node `i`;
- `m_i`: accident-active mask from history;
- `g_{i,k}`: learned gate for node `i`, horizon `k`;
- `r_{i,k}`: learned residual for node `i`, horizon `k`.

Base STID branch:

```text
z_i = concat(
    Conv(vec(X_{:, i, flow/time/day})),
    e_node(i),
    e_time(t),
    e_day(t)
)

h_i = MLP_base(z_i)
Y_base[i, 1:H] = W_base h_i
```

History accident activation:

```text
m_i = 1[ sum_{l=1}^{L} A_{l,i} > 0 ]
e_acc(i) = Emb_acc(m_i)
```

Residual router:

```text
u_i = MLP_router(concat(h_i, e_acc(i)))
g_{i,k} = sigmoid(W_gate,k u_i + b_gate,k)
r_{i,k} = W_res,k u_i
```

Sparse residual forecast:

```text
Y_hat[i,k] = Y_base[i,k] + m_i * g_{i,k} * r_{i,k}
```

Implementation details:

- `b_gate` is initialized to `-2.0`, so the initial gate is small.
- `m_i` hard-masks the residual branch to zero if no accident is observed in
  the history window.
- This preserves a no-change path, but it also means v0 cannot intentionally
  respond to accidents that start only in the future horizon.

## Architecture Diagram

![STIDGatedAccident architecture](/Users/richardo/Desktop/STproject/event-traffic-prediction/idea-stage/accident-aware-event-forecasting/figures/stid-gated-accident-architecture.png)

Source files:

- `figures/stid-gated-accident-architecture.mmd`
- `figures/stid-gated-accident-architecture.md`
- `figures/stid-gated-accident-architecture.png`

## V0 Diagnosis

Current result:

- beats `STIDAccident` on all four counties;
- beats pure `STID` only on Alameda under all-window MAE;
- event-slice gains over pure `STID` are mostly in `future_onset/future_any`;
- `post_last_slot`, `history_only`, and `history_any` are not robust.

This suggests the v0 gate is a useful anti-negative-transfer device, but not yet
a strong accident-effect model.

## V1 Design Requirements

The next module should explicitly target the pure-STID gap:

1. **Use a normal baseline**: keep `Y_base` as a pure-STID-compatible branch.
2. **Predict residuals only where justified**: route by node, horizon, and local
   incident relevance rather than a dense accident embedding.
3. **Add local incident geometry**: include `abs_pm` gap, freeway, direction,
   upstream/downstream relation, and distance-to-incident if available.
4. **Separate observed-event and future-known-event settings**: do not claim
   future accident awareness unless future incident information is provided at
   prediction time.
5. **Optimize an event-aware auxiliary objective**: encourage improvements on
   matched-impact event windows while constraining no-event degradation.

Candidate V1 objective:

```text
L = L_all
  + lambda_event * L_event_impact
  + lambda_normal * max(0, MAE_no_event(model) - MAE_no_event(STID) + margin)
```

Where `L_event_impact` is computed on event windows with matched-control traffic
change above a threshold. This directly encodes the requirement: improve where
accidents matter, and do not pay for it by hurting normal traffic.
