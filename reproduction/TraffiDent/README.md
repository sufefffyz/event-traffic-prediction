# TraffiDent BasicTS Preparation

This folder prepares county-level TraffiDent subsets for the local BasicTS
STID and STID+accident experiments.

Official part:

- Incident matching follows `XTraffic/process/traffic_incident_match.py`:
  same `Fwy`, closest `Abs PM`, `distance <= 0.5`.

Adapter part:

- County subset: `sensor_meta_feature.csv` with `County == target` and
  `Type == Mainline`.
- BasicTS tensor: `[flow, time_of_day, day_of_week, accident_binary]`.
- Split: chronological sliding-window `6/2/2` with `input_len=12`,
  `output_len=12`.
- Default period: 2023 Q1 (`months=1,2,3`) for an initial paper-style run.
