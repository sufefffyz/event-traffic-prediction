# Review Trace: accident-aware event forecasting

**Date**: 2026-05-27  
**Skill**: novelty-check + research-review  
**Route**: local literature search + Zotero + web + secondary Codex agents  

## Prompt Summary

The user requested ARIS automation for accident-aware event traffic forecasting and objected that the earlier response lacked scoring and novelty check. The workflow evaluated six candidate ideas:

1. Counterfactual Incident-Effect Forecaster
2. Retrieval-Augmented Incident Memory
3. Sparse Accident Expert Router
4. Event-Conditioned Propagation Field
5. Uncertainty-First Accident Forecasting
6. Event-Aware Pretraining

## Sources Checked

- TraffiDent / XTraffic
- IGSTGNN
- ConFormer
- STEVE
- Crash heterogeneous treatment effects with doubly robust causal ML
- MSCT counterfactual post-crash traffic prediction
- RAST retrieval-augmented spatiotemporal traffic prediction
- Incident-aware conformal STT
- MH-MoE / TESTAM / traffic MoE
- STMAE and masked traffic pretraining

## Agents

- Research-review agent: `019e6999-1bbc-7223-964a-7fea5ec3c033`, completed.
- Novelty-check agent: `019e6998-e28d-7221-b057-46f0b0d803a3`, timed out during this update; pending response can be merged later if it completes.

## Decisions

- Downgrade counterfactual idea from high novelty to proceed-with-caution because of crash HTE and MSCT.
- Downgrade retrieval idea because of RAST/MRATP; keep only incident-response residual retrieval.
- Keep sparse router as main implementation skeleton, but require accident-specific node-horizon residual routing and placebo tests.
- Use propagation field only as an inductive prior.
- Deprioritize event-aware pretraining as first project.

## Output Files

- `idea-stage/accident-aware-event-forecasting/IDEA_REPORT.md`
- `idea-stage/accident-aware-event-forecasting/NOVELTY_CHECK.md`
- `idea-stage/accident-aware-event-forecasting/REVIEW_SUMMARY.md`

