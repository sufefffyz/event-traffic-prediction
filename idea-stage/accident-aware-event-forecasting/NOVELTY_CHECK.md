# Novelty Check: Accident-Aware Event Traffic Forecasting

**Date**: 2026-05-27  
**Mode**: ARIS novelty-check + research-review  
**Status**: round-1 completed, one novelty reviewer still pending at timeout  

## Core Claims Checked

1. Accident-aware forecasting should model accident effects as residuals relative to normal traffic, not as global accident embeddings.
2. A sparse accident expert/router can reduce negative transfer from rare events.
3. Historical incident response retrieval can provide useful priors for new accidents.
4. Directional/distance-aware propagation fields can model upstream/downstream accident effects.
5. Accident-aware uncertainty calibration may be more meaningful than mean-only forecasting.
6. Event-aware pretraining may learn transferable accident-response representations.

## Closest Prior Work

| Paper | Year | What It Already Covers | Risk To Our Ideas |
| --- | --- | --- | --- |
| [TraffiDent / XTraffic](https://arxiv.org/abs/2407.11477) | 2024/2026 | Large-scale traffic + incident + road meta dataset; post-incident forecasting; incident classification; causal analysis | Dataset/task novelty is mostly taken |
| [IGSTGNN](https://arxiv.org/abs/2602.02528) | 2026 | Incident-context spatial fusion and temporal incident impact decay | Generic propagation/decay modules are crowded |
| [ConFormer](https://arxiv.org/abs/2512.09398) | 2025/2026 | Accident/regulation condition channels, graph propagation, guided normalization | Accident-conditioned Transformer is crowded |
| [Crash HTE / doubly robust causal ML](https://arxiv.org/abs/2401.00781) | 2024 | Heterogeneous causal effects of crashes, matched counterfactual outcomes | Pure counterfactual crash effect claim is risky |
| [MSCT](https://arxiv.org/abs/2407.14065) | 2024 | Counterfactual post-crash traffic prediction with time-varying confounding | Counterfactual Transformer framing is directly close |
| [RAST](https://arxiv.org/abs/2508.16623) | 2025/2026 | General retrieval-augmented spatiotemporal traffic prediction | Generic retrieval memory is not enough |
| MH-MoE / TESTAM / ST-MoE | 2024-2025 | Traffic mixture-of-experts and pattern-based gating | Generic sparse expert/router is not enough |
| [Incident-aware conformal STT](https://arxiv.org/abs/2603.16857) | 2026 | Incident-aware dynamic graph and adaptive conformal prediction | Uncertainty-first direction has close prior |
| STMAE / STEP / masked ST pretraining | 2022-2024 | Masked spatial-temporal pretraining for traffic | Generic event-aware pretraining novelty is weak |
| [STEVE](https://arxiv.org/abs/2311.12472) | 2025 | Basis confounder bank for weather/accident/holiday confounders | Confounder bank framing is crowded |

## Scored Ideas

Scores are conservative after novelty search. 10 means strong, 1 means weak.

| Idea | Novelty | Feasibility | Impact | Risk | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Counterfactual Incident-Effect Forecaster | 5.5 | 6 | 8 | 8 | Proceed with caution, do not claim strong causality |
| Retrieval-Augmented Incident Memory | 5.5 | 8 | 7 | 6 | Use as pilot/component, not standalone main idea |
| Sparse Accident Expert Router | 6.5 | 8 | 7 | 5 | Proceed as main model skeleton if made accident-specific |
| Event-Conditioned Propagation Field | 5 | 6 | 7 | 7 | Use as inductive bias, not standalone idea |
| Uncertainty-First Accident Forecasting | 5 | 7 | 6 | 7 | Backup direction; requires sharpness-calibration proof |
| Event-Aware Pretraining | 4 | 5 | 7 | 8 | Deprioritize or abandon as first project |

## Key Novelty Adjustments

### 1. Counterfactual direction must be renamed

The original title "Counterfactual Incident-Effect Forecaster" is too close to MSCT and crash HTE work. The safer framing is:

> Counterfactual-inspired sparse residual forecasting.

Do not claim true causal effect unless we explicitly control weather, report lag, roadwork, multi-incident overlap, and time-varying confounders. The current project can still use matched no-event windows, but only as a weak normal-traffic baseline.

### 2. Retrieval must be incident-specific

RAST already covers general retrieval-augmented spatiotemporal forecasting. Our delta must be:

- retrieve historical **incident response residuals**, not generic traffic patterns;
- use incident type, abs_pm/freeway, direction, road meta, and pre-event state;
- prohibit post-event leakage;
- compare against RAST-style generic retrieval and seasonal analog retrieval.

### 3. Router must be residual and local

Traffic MoE and memory gating already exist. The router cannot be generic "which expert is best". It should route:

- only accident-induced residuals;
- over node-horizon pairs, not whole samples;
- with placebo negatives from random event time/type/location;
- with a normal-window degradation penalty.

### 4. Propagation field should be a router prior

IGSTGNN/ConFormer already cover incident propagation. Use directional/abs_pm/delay kernels as the prior for router sparsity and residual support, not as the headline method.

### 5. Uncertainty is backup

Incident-aware conformal STT already occupies this neighborhood. Only keep this if mean metrics fail but event-window calibration and sharpness improve without simply widening intervals.

## Final Recommended Direction

**Working title**: Counterfactual Sparse Residual Routing for Accident-Aware Traffic Forecasting

Core idea:

> Accidents are sparse local perturbations, not global covariates. Instead of concatenating accident indicators to all traffic states, learn a normal-traffic baseline and route accident-induced residual experts only to affected node-horizon pairs.

Method components:

1. **Normal baseline**: STID/D2STGNN predicts regular traffic.
2. **Matched no-event prior**: same node, same weekday/time, similar pre-traffic, no nearby incidents.
3. **Sparse residual router**: predicts affected node-horizon weights from incident type, direction, abs_pm/road distance, road meta, report delay proxy, and pre-state anomaly.
4. **Accident residual expert**: predicts local delta over the normal baseline.
5. **Incident response memory**: retrieves top-k historical residual trajectories as a prior or adapter input.
6. **Placebo regularization**: shuffled time/location/type should not activate the router.

## Required Experiments

| Experiment | Purpose |
| --- | --- |
| STID / STIDAccident / D2STGNN / ConFormer / IGSTGNN | Establish strong baselines |
| matched-control baseline | Show normal baseline difficulty |
| retrieval-only residual baseline | Fast pilot for memory usefulness |
| sparse router vs dense accident expert | Prove sparse routing matters |
| random event time/location/type | Rule out event leakage and time-pattern cheating |
| no direction / no abs_pm / no event type | Prove traffic-specific signals matter |
| normal-window degradation check | Prove accident module does not hurt normal traffic |
| event-window, impact-zone, type-specific metrics | Show accident-specific gains |

## Decision

Proceed with **Sparse Accident Residual Router + counterfactual-inspired residual target + retrieval memory pilot**.

Abandon as first project:

- generic event-aware pretraining;
- standalone propagation field;
- standalone uncertainty-first forecasting.
