# Pilot Results: Sparse Residual Router

**Date**: 2026-05-27  
**Script**: `reproduction/analysis/sparse_residual_router_pilot.py`  
**Data**: TraffiDent 2023Q1 county BasicTS data  
**Model**: ridge residual expert over persistence, using event/node/pre-state features  

## Pilot Question

Before building a neural sparse accident residual router, test whether a simple residual expert can predict accident-window residuals better than persistence.

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

| County | Persistence MAE | Global Residual Mean | Ridge Residual Expert | Decision |
| --- | ---: | ---: | ---: | --- |
| LosAngeles | 15.35 | 15.66 | 15.70 | worse |
| Orange | 15.30 | 15.49 | 16.01 | worse |
| Alameda | 16.26 | 16.54 | 17.29 | worse |
| ContraCosta | 19.48 | 19.74 | 19.15 | better |

Full outputs:

- `reproduction/analysis/sparse_residual_router_pilot/sparse_residual_router_pilot_summary.csv`
- `reproduction/analysis/sparse_residual_router_pilot/sparse_residual_router_pilot_by_horizon.csv`
- `reproduction/analysis/sparse_residual_router_pilot/sparse_residual_router_pilot_meta.json`

## Interpretation

This is a mixed/mostly negative pilot.

The result says a global linear residual expert is not enough. It tends to inject residual noise into LA, Orange, and Alameda. ContraCosta improves slightly, which suggests some county/event regimes may have predictable residual structure, but the signal is not universal.

This supports the reviewer's warning:

> The accident module must be sparse and local; dense/global residual correction will cause negative transfer.

## Decision Update

Do not implement a dense residual expert as the main model.

Next version should test one of:

1. **Impact-gated residual expert**: train residual only on high-impact event windows and force gate sparsity.
2. **Type/county-specific residual experts**: avoid pooling all incident regimes.
3. **Node-horizon router**: route residuals only when predicted impact probability is high.
4. **Persistence-aware objective**: optimize only residual corrections that beat persistence, with a no-change option.

