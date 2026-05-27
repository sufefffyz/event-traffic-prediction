# Pilot Results: Sparse Residual Router

**Date**: 2026-05-27  
**Script**: `reproduction/analysis/sparse_residual_router_pilot.py`  
**Data**: TraffiDent 2023Q1 county BasicTS data  
**Model**: ridge residual experts over persistence, using event/node/pre-state features  

## Pilot Question

Before building a neural sparse accident residual router, test whether a simple residual expert can predict accident-window residuals better than persistence, and whether sparse gating avoids negative transfer.

The pilot uses:

- event type;
- county;
- freeway;
- distance / abs_pm gap;
- time-of-day / day-of-week;
- pre-event last, mean, std, slope;
- road width / lane width / speed limit when available.

Target:

```text
residual_h = true_future_h - persistence_h
prediction_h = persistence_h + ridge_residual_h
```

## Results

| County | Persistence | Dense Ridge | Impact-Gated Ridge q75 | High-Impact-Train Gated q75 | County-Specific Ridge | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LosAngeles | 15.35 | 15.70 | 14.55 | 14.55 | 15.67 | gated better |
| Orange | 15.30 | 16.01 | 15.05 | 15.51 | 16.25 | gated better |
| Alameda | 16.26 | 17.29 | 15.70 | 16.04 | 17.48 | gated better |
| ContraCosta | 19.48 | 19.15 | 18.24 | 18.24 | 20.93 | gated better |

Full outputs:

- `reproduction/analysis/sparse_residual_router_pilot/sparse_residual_router_pilot_summary.csv`
- `reproduction/analysis/sparse_residual_router_pilot/sparse_residual_router_pilot_by_horizon.csv`
- `reproduction/analysis/sparse_residual_router_pilot/sparse_residual_router_pilot_meta.json`

## Interpretation

This is a positive pilot for the sparse-router framing.

The dense/global residual expert still injects noise into LA, Orange, and Alameda. However, once residual correction is only applied to windows predicted as high-impact by a train-only residual-magnitude gate, MAE improves over persistence in all four counties.

This supports the core mechanism:

```text
prediction = normal_forecast + sparse_gate(event, node, history) * accident_residual
```

This supports the reviewer's warning:

> The accident module must be sparse and local; dense/global residual correction will cause negative transfer.

## Decision Update

Do not implement a dense residual expert as the main model.

Implement the next BasicTS/STID prototype as a gated residual module:

- base forecast branch: normal STID forecast;
- gate branch: predicts whether an accident window is high-impact;
- residual branch: predicts accident residual only when the gate is active;
- no-change option: the model should be able to keep the base forecast unchanged.

Still useful follow-up variants:

1. **Impact-gated residual expert**: train residual only on high-impact event windows and force gate sparsity.
2. **Type/county-specific residual experts**: avoid pooling all incident regimes.
3. **Node-horizon router**: route residuals only when predicted impact probability is high.
4. **Persistence-aware objective**: optimize only residual corrections that beat persistence, with a no-change option.
