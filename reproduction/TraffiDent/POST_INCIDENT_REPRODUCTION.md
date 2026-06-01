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
