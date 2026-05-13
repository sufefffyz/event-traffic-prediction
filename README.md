# Event Traffic Prediction

This repository collects codebases for event-based traffic forecasting experiments in a single independent project.

## Imported Codebases

The external projects are vendored as plain source directories under `external/`. Their original `.git` metadata has been removed, so this repository is not connected to the upstream repositories or configured as submodules.

| Directory | Source | Imported revision |
| --- | --- | --- |
| `external/ConFormer` | `https://github.com/Dreamzz5/ConFormer.git` | `7d6ab5a6a71bc78a3e0c7835b6a860637d2cc18b` |
| `external/IGSTGNN` | `https://github.com/fanlixiang/IGSTGNN.git` | `d8ae94aa3d7dd3f17f03696425515afad3bf398e` |
| `external/BasicTS` | local `STproject/BasicTS` checkout | branch `v0.5.8`, commit `63fa719`, package version `0.5.8` |

## Notes

- Keep upstream licenses and attribution files with each imported codebase.
- Treat `external/` as vendored third-party source unless this project later adds integration code.
