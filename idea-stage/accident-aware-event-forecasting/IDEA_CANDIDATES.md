# Idea Candidates: Accident-Aware Event Forecasting

## Active Candidate

**Counterfactual-inspired Sparse Accident Residual Router**

- **One-line**: Learn accident-induced residuals over a normal-traffic baseline and route them sparsely to affected node-horizon pairs.
- **Novelty score**: 6.5/10 after check.
- **Reviewer assessment**: Strongest feasible method skeleton, but must avoid being generic MoE/gating.
- **Pilot result**: POSITIVE FOR GATING; MIXED AS A TRAINED MODEL. A dense/global ridge residual expert mostly worsened persistence, but an impact-score gated residual expert improved all four counties. The trained BasicTS `STIDGatedAccident` beats `STIDAccident` on all four counties, but beats pure `STID` only on Alameda under overall MAE. Event-sensitive metrics show the useful signal is mainly in `future_onset/future_any`, not all post-incident windows.
- **Estimated effort**: 2-6 weeks for a BasicTS prototype and first full run.
- **Why selected first**: Best balance between feasibility, current evidence, and paper narrative.

## Candidates

### Candidate 1: Incident Response Memory

- **One-line**: Retrieve historical incident residual trajectories for new accidents and use them as a prior for forecasting.
- **Novelty score**: 5.5/10.
- **Reviewer assessment**: Generic retrieval is crowded because of RAST; novelty depends on incident-response residual retrieval with no post-event leakage.
- **Pilot result**: NEGATIVE. Retrieval residual was worse than the matched normal prior in all four counties and worse than persistence.
- **Estimated effort**: 1-2 days for pilot, 1-2 weeks as model component.
- **Why not selected first**: Too close to generic retrieval-augmented traffic prediction if used alone, and the first pilot did not show signal.

### Candidate 2: Counterfactual-Inspired Residual Target

- **One-line**: Use matched no-event windows as a weak normal baseline and learn accident residuals instead of raw flow.
- **Novelty score**: 5.5/10.
- **Reviewer assessment**: Strong framing but close to crash treatment-effect and MSCT work; avoid strong causal claims.
- **Pilot result**: Pending; residual construction can be paired with the retrieval pilot.
- **Estimated effort**: 2-4 days for target construction, 2-4 weeks for model integration.
- **Why not selected first**: Counterfactual validity is the biggest review risk.

### Candidate 3: Directional Propagation Prior

- **One-line**: Use upstream/downstream, abs_pm gap, road distance, and horizon delay as a prior for where accident residuals can propagate.
- **Novelty score**: 5/10.
- **Reviewer assessment**: Too close to IGSTGNN/ConFormer if standalone, useful as a sparse-router prior.
- **Pilot result**: Not started.
- **Estimated effort**: 1-2 weeks as a module.
- **Why not selected first**: Needs validation that direction/delay is real, not just a plausible diagram.

## Killed Or Deferred Ideas

### Event-Aware Pretraining

- **Kill reason**: Masked ST pretraining is crowded; likely learns road/time identity before accident response.
- **Kill date**: 2026-05-27.
- **Source**: Novelty check, STMAE/STEP prior work.

### Standalone Uncertainty-First Forecasting

- **Kill reason**: Incident-aware conformal STT is close; keep as backup if mean metrics fail.
- **Kill date**: 2026-05-27.
- **Source**: Novelty check.

### Standalone Event Propagation Field

- **Kill reason**: IGSTGNN and ConFormer already occupy propagation/decay/conditional graph framing.
- **Kill date**: 2026-05-27.
- **Source**: Reviewer critique.

## Idea Switch Log

| Date | From | To | Reason |
| --- | --- | --- | --- |
| 2026-05-27 | Generic accident-aware forecasting | Sparse accident residual router | Naive STID+accident was weak and prior work already covers generic condition fusion |
| 2026-05-27 | Counterfactual forecaster | Counterfactual-inspired residual target | Crash HTE and MSCT make strong causal framing risky |
| 2026-05-27 | Retrieval as main method | Incident response memory as pilot/component | RAST makes generic retrieval too crowded |
| 2026-05-27 | Incident response memory as candidate | Incident response memory as baseline/component only | Retrieval-only pilot was negative |
| 2026-05-27 | Dense residual expert | Sparse/local gated residual expert | Ridge residual pilot showed negative transfer in three of four counties; impact-score gated residual improved all four |
