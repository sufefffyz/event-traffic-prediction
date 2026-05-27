# Pilot Results: Incident Response Memory

**Date**: 2026-05-27  
**Script**: `reproduction/analysis/incident_memory_retrieval_pilot.py`  
**Server output**: `reproduction/analysis/incident_memory_retrieval_pilot/`  
**Data**: TraffiDent 2023Q1 county BasicTS data  
**Counties**: LosAngeles, Orange, Alameda, ContraCosta  

## Pilot Question

Can a simple historical incident-response memory outperform normal matched traffic priors?

This is the cheapest check for the retrieval idea. If naive historical incident residual retrieval already beats normal priors, retrieval memory is worth making a method component. If not, retrieval should not be the main idea.

## Methods

| Method | Meaning |
| --- | --- |
| `persistence` | Repeat the last observed value. |
| `normal_prior` | Same node, same weekday/time if available, otherwise same time-of-day, excluding nearby incident windows. |
| `global_residual` | `normal_prior + average train incident residual`. |
| `type_residual` | `normal_prior + average train incident residual for the same incident type`. |
| `retrieval_residual` | `normal_prior + top-k historical incident residuals` using pre-state, incident type, freeway, abs_pm, and time-of-day distance. |

## Summary

| County | Train Events | Test Events | Persistence MAE | Normal Prior MAE | Retrieval MAE | Retrieval vs Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LosAngeles | 7740 | 3111 | 15.35 | 17.82 | 19.30 | worse |
| Orange | 2064 | 885 | 15.30 | 18.36 | 19.88 | worse |
| Alameda | 1530 | 594 | 16.26 | 17.09 | 17.90 | worse |
| ContraCosta | 805 | 365 | 19.48 | 24.26 | 25.63 | worse |

Full files:

- `reproduction/analysis/incident_memory_retrieval_pilot/retrieval_pilot_summary.csv`
- `reproduction/analysis/incident_memory_retrieval_pilot/retrieval_pilot_by_horizon.csv`
- `reproduction/analysis/incident_memory_retrieval_pilot/retrieval_pilot_event_counts.json`

## Interpretation

The naive retrieval idea is negative. Historical incident residuals, retrieved with simple metadata and pre-state similarity, do not improve over the matched normal prior and are worse than persistence in all four counties.

This does **not** kill all accident-aware modeling, but it does kill the idea that a simple incident memory bank can be the main contribution. If retrieval is used later, it needs a stronger representation:

- retrieve local residual shapes after removing persistence;
- constrain retrieval to direction/abs_pm neighborhoods;
- learn a residual metric rather than hand-designed distance;
- use retrieval as a weak prior for a sparse router, not as a direct predictor.

## Decision Update

Retrieval-Augmented Incident Memory should be downgraded from "candidate method" to "optional component / baseline". The main direction remains:

> Sparse accident residual routing over a normal-traffic baseline.

