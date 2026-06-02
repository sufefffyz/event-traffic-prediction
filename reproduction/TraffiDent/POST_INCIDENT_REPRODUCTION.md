# TraffiDent Post-Incident Forecasting Reproduction

This note tracks the first reproduction pass for the TraffiDent Section 4.2 /
Appendix A.8 post-incident forecasting experiment.

## Paper protocol

- Data period: first 3 months of TraffiDent traffic volume data.
- Split: chronological 6/2/2 train/validation/test.
- Forecasting horizons: 5, 15, and 30 minutes, i.e. `t=1,3,6` for 5-minute data.
- Incident sample definition: match each incident to the closest sensor on the
  same freeway using `Abs PM`; map the incident timestamp to a 5-minute slot;
  use the next slot as the post-incident forecasting origin.
- `General` is all test samples. `Incident` is the matched sensor/node at those
  post-incident origins.

## Important ambiguity

The paper text says the experiment uses San Bernardino with 561 mainline
sensors, while Table 3 and Appendix A.8 say D5/Monterey. In the released
metadata available on the server:

- District 5 has 565 sensors total and 421 `Mainline` sensors.
- Monterey County has 166 sensors total and 123 `Mainline` sensors.
- San Bernardino County has 893 sensors total and 452 `Mainline` sensors.

Therefore the first reproduction uses `District == 5` with all sensor types as
the primary paper-table candidate, because this is the only local slice close to
the reported 561 count and matches the D5 wording. This is an adapter choice and
must be reported with the result.

## First runnable path

The first runnable path uses AGCRN, one of the Table 3 baselines. It is the
lightest first pass for verifying the data slice, post-incident case selection,
saved predictions, and metric table before expanding to DCRNN, GWNet, STGODE,
DSTAGNN, and D2STGNN.

The mapped settings are:

- model: AGCRN, Table 3 baseline
- seed: `2023`, matching the LargeST public run scripts
- max epochs: `100`, patience: `30`
- batch size: `64`
- input channels: `[flow, time_of_day, day_of_week]`, target: `flow`
- optimizer: Adam, learning rate `1e-3`, weight decay `0`

```bash
screen -dmS traffident_post_incident_agcrn_d5 bash reproduction/server_scripts/prepare_and_run_traffident_post_incident_agcrn_d5.sh
```

Expected outputs:

- Dataset: `/data/yuzhang_fei/TraffiDent/basicts/TraffiDent_D5_2023Q1`
- Training log: `reproduction/logs/traffident_post_incident_agcrn_d5_100ep_g1.log`
- Table CSV: `reproduction/analysis/traffident_post_incident_table/TraffiDent_D5_2023Q1/post_incident_forecasting_table.csv`
- Case CSV: `reproduction/analysis/traffident_post_incident_table/TraffiDent_D5_2023Q1/post_incident_cases.csv`

## AGCRN D5 result, 2026-06-02

This run completed on the server after fixing the GPU mapping in
`reproduction/server_scripts/prepare_and_run_traffident_post_incident_agcrn_d5.sh`.
The effective command was:

```bash
python BasicTS/experiments/train.py -c baselines/AGCRN/TraffiDent_D5.py -g 1
```

Artifacts:

- Training log:
  `reproduction/logs/traffident_post_incident_agcrn_d5_100ep_g1_fixed_20260602.log`
- Test results:
  `BasicTS/checkpoints/AGCRN/TraffiDent_D5_2023Q1_100_12_12_paper/5c041ba989f9f9e4aa501e34c882a39f/test_results`
- Table CSV:
  `reproduction/analysis/traffident_post_incident_table/TraffiDent_D5_2023Q1/post_incident_forecasting_table.csv`

Confirmed settings:

- dataset: `TraffiDent_D5_2023Q1`
- node count: `565`
- timesteps: `25920`
- chronological split: train `15537`, validation `5179`, test `5180`
- input/output length: `12/12`
- model input channels: `[flow, time_of_day, day_of_week]`
- target channel: `flow`
- scaler: global train-window z-score via `IndexedNPZStandardScaler`
- missing traffic values: interpolated, `nan_before=2827720`, `nan_after=0`
- incident matching: same freeway, nearest `Abs PM`, maximum distance `0.5`
- matched incident labels: `NoInj`, `UnknInj`, `1141`
- matched incidents in this slice: `613`
- active event slots in the prepared data: `1218`

The best-validation checkpoint gives:

| Split | node windows | valid@t1 | MAE@t1 | RMSE@t1 | MAPE@t1 | valid@t3 | MAE@t3 | RMSE@t3 | MAPE@t3 | valid@t6 | MAE@t6 | RMSE@t6 | MAPE@t6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| General | 2926700 | 2351935 | 10.5107 | 20.6547 | 23.1438 | 2351954 | 11.4204 | 22.9364 | 24.6104 | 2351968 | 12.2740 | 24.9094 | 26.4230 |
| Incident | 123 | 97 | 8.6245 | 13.8256 | 17.3443 | 97 | 11.4535 | 19.3297 | 20.0653 | 95 | 11.3142 | 23.8639 | 10.5446 |

Interpretation:

- This run does not reproduce a strong "incident windows are harder than
  general windows" conclusion. Incident MAE is lower than General at `t=1` and
  `t=6`, and roughly equal at `t=3`.
- The incident sample is very small: only `123` node-windows, with
  `95-97` valid labels at the reported horizons. This is too narrow for a
  strong negative or positive conclusion.
- The most likely next check is the sample-definition gap: the current adapter
  uses one matched sensor and the next post-incident slot. The paper may have
  used a broader affected window, a different county/district slice, mainline
  filtering, or multiple nearby sensors.

Recommended next steps:

1. Reconcile the D5 / Monterey / San Bernardino ambiguity before expanding the
   claim.
2. Audit the post-incident sample construction against the released TraffiDent
   code and appendix wording.
3. If the sample definition is confirmed, run one more quick Table 3 baseline
   or one additional seed to check whether this AGCRN result is model/seed
   specific.
