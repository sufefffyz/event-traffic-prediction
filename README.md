# Event Traffic Prediction

This repository is organized around BasicTS as the primary research framework for future event-based traffic forecasting work. ConFormer and IGSTGNN are kept as reproduction/reference implementations.

## Directory Layout

| Directory | Role | Notes |
| --- | --- | --- |
| `BasicTS/` | Primary framework | Main codebase for future method development, dataset integration, training, evaluation, and experiment management. |
| `reproduction/ConFormer/` | Reproduction/reference code | Imported to reproduce and inspect ConFormer. Keep changes minimal and document any reproduction patches. |
| `reproduction/IGSTGNN/` | Reproduction/reference code | Imported to reproduce and inspect IGSTGNN. Keep changes minimal and document any reproduction patches. |

## Source Provenance

The imported projects are vendored as plain source directories. Their original `.git` metadata has been removed, so this repository is independent and does not track the upstream repositories as remotes or submodules.

| Directory | Source | Imported revision |
| --- | --- | --- |
| `BasicTS/` | local `STproject/BasicTS` checkout | branch `v0.5.8`, commit `63fa719`, package version `0.5.8` |
| `reproduction/ConFormer/` | `https://github.com/Dreamzz5/ConFormer.git` | `7d6ab5a6a71bc78a3e0c7835b6a860637d2cc18b` |
| `reproduction/IGSTGNN/` | `https://github.com/fanlixiang/IGSTGNN.git` | `d8ae94aa3d7dd3f17f03696425515afad3bf398e` |

## Development Rules

- Use `BasicTS/` as the base for new research code.
- Treat `reproduction/` as baseline reproduction material, not the main implementation surface.
- Preserve upstream licenses and attribution files inside each imported codebase.
- If a reproduction directory needs fixes, prefer small, well-documented patches.

## Reproduction Notes

- Start from `reproduction/REPRODUCTION_START.md` for the current task definition, effective-module audit, known code breakpoints, and BasicTS migration plan.
