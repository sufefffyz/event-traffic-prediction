# Oracle Future Accident Router

**Status**: implementation ready, training pending
**Code**: `BasicTS/baselines/STIDOracleFutureAccident`
**Purpose**: aggressive upper-bound diagnostic

## Setting

This module intentionally reads the future accident sequence from
`future_data[..., accident_feature_index]`. It does not read future target
flow. The result should be interpreted as an oracle / known-future-event
upper bound, not as a deployable forecasting protocol.

The experiment answers:

> If the model is told which sensor-time slots will have accident records in
> the forecast horizon, can it finally beat pure STID on event windows?

## Formula

Let pure STID produce:

```text
Z = STIDEncoder(X_hist)
Y_base = W_base(Z)
```

The oracle branch builds future event features for each horizon `h` and node
`i`:

```text
e_{h,i}         = 1[future accident at horizon h, node i]
started_{h,i}   = 1[sum_{k<=h} e_{k,i} > 0]
remaining_{h,i} = 1[sum_{k>=h} e_{k,i} > 0]
any_i           = 1[sum_h e_{h,i} > 0]
```

The prediction is:

```text
F = EventConv([e, started, remaining, any])
H_h = concat(Z, F_h)
Y_hat_h = Y_base_h + any_i * sigmoid(G(H_h)) * R(H_h)
```

`any_i` keeps no-future-event windows identical to STID, while allowing all
horizons in a future-event window to be corrected.

## Why This Is More Aggressive

Previous modules used only history accident states:

```text
history accident -> residual
```

This one uses the future accident sequence directly:

```text
future accident sequence -> horizon-wise residual
```

So it can affect `future_onset` windows where history has no accident but the
forecast horizon does. This is exactly the slice where earlier gated results
showed the cleanest weak positive signal.

## Success Criterion

The first check is not paper-valid realism. It is signal existence:

```text
MAE(STIDOracleFutureAccident) - MAE(STID) < 0
```

Priority slices:

- `future_onset`
- `future_any`
- `ongoing`
- `high_impact_*`

If this model cannot beat STID on those slices, future accident labels alone
are probably not enough under the current node-event mapping. If it can, the
next research step is to replace oracle future events with predicted /
retrieved event likelihoods.
