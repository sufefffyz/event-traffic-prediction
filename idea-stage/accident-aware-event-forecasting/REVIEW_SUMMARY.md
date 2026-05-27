# ARIS Review Summary

**Date**: 2026-05-27  
**Reviewer route**: secondary Codex agent, xhigh reasoning  
**Reviewer status**: one full research-review response received; novelty-check response pending at timeout  

## Main Reviewer Verdict

Do not pitch the project as a stronger accident embedding model. The user's own STID vs STID+accident runs already make that story weak. The viable story is:

> Accident information should be modeled as local sparse residuals relative to normal traffic, not as a global feature concatenated into every sample.

## Strongest Objections

| Idea | Reviewer Objection |
| --- | --- |
| Counterfactual | True counterfactual is unobserved; matched controls may be seasonality proxies, not no-accident worlds |
| Retrieval | May collapse to kNN seasonal analog; post-event leakage is a serious risk |
| Sparse router | Could be dismissed as MoE/gating unless it routes accident residuals at node-horizon level |
| Propagation field | Too close to IGSTGNN/ConFormer unless direction/delay is validated |
| Uncertainty | Could simply widen intervals; must prove calibration-sharpness tradeoff |
| Pretraining | Crowded and expensive; likely learns road/time identity rather than accident response |

## Reviewer Top 3

1. **Sparse Accident Expert Router** as the implementable model skeleton.
2. **Counterfactual-inspired residual forecasting** as the paper framing and loss target.
3. **Retrieval-Augmented Incident Memory** as the fastest pilot and optional prior.

## Recommended Paper Framing

**Title**: Counterfactual Sparse Residual Routing for Accident-Aware Traffic Forecasting

**Core claim**: Naive accident embeddings fail because incidents are sparse local treatments, not global covariates. Modeling accident effects as sparse, horizon-dependent residuals over a matched no-event baseline improves event-affected forecasts without degrading normal traffic prediction.

## Method Skeleton

```text
base_pred = BaseForecaster(history)
normal_prior = MatchedNoEventPrior(history, node, time)
memory_delta = IncidentMemory(event, node, horizon)
gate = SparseRouter(event, node_meta, distance, direction, horizon, pre_state)
delta = AccidentResidualExpert(history, event, memory_delta)
pred = base_pred + gate * delta
```

## Experiments Required To Survive Review

- Compare against STID, STIDAccident, D2STGNN, ConFormer, IGSTGNN when runnable.
- Include matched-control baseline and retrieval-only baseline.
- Show overall metric does not degrade.
- Report event-window and impact-zone metrics.
- Include random event time/location/type placebo.
- Ablate sparse gate, dense gate, direction, abs_pm distance, event type, retrieval.

