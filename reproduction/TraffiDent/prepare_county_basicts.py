#!/usr/bin/env python3
"""Prepare TraffiDent county subsets for BasicTS STID experiments.

The incident-to-sensor matching follows the official XTraffic script:
group incidents and sensors by freeway (`Fwy`), then choose the sensor
with the smallest absolute post-mile distance (`Abs PM`) under a threshold.

The county/Mainline subset, BasicTS feature tensor, and split index are
adapter code for these reproduction experiments.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


COUNTY_LABELS = {
    "LosAngeles": "Los Angeles",
    "Orange": "Orange",
    "Alameda": "Alameda",
    "ContraCosta": "Contra Costa",
}
ACCIDENT_TYPES = ("NoInj", "UnknInj", "1141")
STEPS_PER_DAY = 288


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice TraffiDent counties into BasicTS data.npz/index.npz files."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"),
        help="Path to the downloaded TraffiDent/XTraffic zip archive.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/basicts"),
        help="Directory where prepared BasicTS datasets will be written.",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--months",
        default="1,2,3",
        help="Comma separated months. Default is Q1 2023, the paper-style short horizon.",
    )
    parser.add_argument(
        "--counties",
        default="LosAngeles,Orange,Alameda,ContraCosta",
        help=f"Comma separated county slugs. Choices: {','.join(COUNTY_LABELS)}",
    )
    parser.add_argument(
        "--sensor-type",
        default="Mainline",
        help="Sensor Type filter in sensor_meta_feature.csv.",
    )
    parser.add_argument(
        "--traffic-channel",
        type=int,
        default=0,
        help="Traffic channel to forecast. TraffiDent channel 0 is flow.",
    )
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument(
        "--split-ratio",
        default="0.6,0.2,0.2",
        help="Chronological train/val/test split over sliding windows.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.5,
        help="Official matching threshold on Abs PM distance.",
    )
    parser.add_argument(
        "--event-window-slots",
        type=int,
        default=2,
        help="Number of 5-minute slots marked active from each matched incident time.",
    )
    parser.add_argument(
        "--event-types",
        default="accident",
        choices=["accident", "all"],
        help="Use accident-like records only or all TraffiDent incident classes.",
    )
    parser.add_argument(
        "--match-scope",
        default="subset",
        choices=["subset", "all"],
        help=(
            "subset: match incidents to available county/Mainline nodes; "
            "all: match using all sensors, then keep matches landing in the subset."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite datasets that already have data.npz and index.npz.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> List[float]:
    parsed = [float(part) for part in value.split(",") if part.strip()]
    if len(parsed) != 3 or not np.isclose(sum(parsed), 1.0):
        raise ValueError(f"Expected three split ratios summing to 1.0, got {parsed}")
    return parsed


def parse_counties(value: str) -> List[str]:
    counties = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(counties) - set(COUNTY_LABELS))
    if unknown:
        raise ValueError(f"Unknown county slugs: {unknown}; choices={sorted(COUNTY_LABELS)}")
    return counties


def period_label(months: Sequence[int]) -> str:
    if list(months) == [1, 2, 3]:
        return "Q1"
    if len(months) == 12 and list(months) == list(range(1, 13)):
        return "FullYear"
    return "M" + "-".join(f"{month:02d}" for month in months)


def dataset_name(slug: str, year: int, months: Sequence[int]) -> str:
    return f"TraffiDent_{slug}_{year}{period_label(months)}"


def find_member(zf: zipfile.ZipFile, pattern: str) -> str:
    regex = re.compile(pattern)
    matches = [name for name in zf.namelist() if regex.search(name)]
    if not matches:
        raise FileNotFoundError(f"No zip member matched pattern: {pattern}")
    return sorted(matches, key=len)[0]


def month_member(zf: zipfile.ZipFile, year: int, month: int) -> str:
    if year == 2023:
        name = f"p{month:02d}_done.npy"
        if name in zf.namelist():
            return name
    return find_member(zf, rf"{year}_p{month:02d}\.npy$")


def read_csv_member(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as probe:
        head = probe.read(4096)
    sep = "\t" if head.count(b"\t") > head.count(b",") else ","
    with zf.open(name) as fh:
        return pd.read_csv(fh, sep=sep, low_memory=False)


def load_metadata(zf: zipfile.ZipFile) -> pd.DataFrame:
    meta_name = find_member(zf, r"sensor_meta_feature\.csv$")
    meta = read_csv_member(zf, meta_name)
    meta = meta.reset_index().rename(columns={"index": "global_index"})
    meta["station_id"] = meta["station_id"].astype(str)
    meta["Fwy"] = pd.to_numeric(meta["Fwy"], errors="coerce")
    meta["Abs PM"] = pd.to_numeric(meta["Abs PM"], errors="coerce")
    return meta


def load_incidents(zf: zipfile.ZipFile, year: int, event_types: str) -> pd.DataFrame:
    incident_name = find_member(zf, rf"incidents_y{year}\.csv$")
    incidents = read_csv_member(zf, incident_name)
    incidents = incidents.copy()
    incidents["Fwy"] = pd.to_numeric(incidents["Fwy"], errors="coerce")
    incidents["Abs PM"] = pd.to_numeric(incidents["Abs PM"], errors="coerce")
    incidents["dt"] = pd.to_datetime(incidents["dt"], errors="coerce")
    incidents = incidents.dropna(subset=["incident_id", "Fwy", "Abs PM", "dt"])
    incidents["Fwy"] = incidents["Fwy"].astype(int)
    if event_types == "accident":
        incidents = incidents[incidents["Type"].isin(ACCIDENT_TYPES)].copy()
    return incidents


def official_style_match(
    incidents: pd.DataFrame,
    candidate_meta: pd.DataFrame,
    distance_threshold: float,
) -> pd.DataFrame:
    rows = []
    sensors = candidate_meta.dropna(subset=["Fwy", "Abs PM"]).copy()
    sensors["Fwy"] = sensors["Fwy"].astype(int)

    for fwy, incident_group in incidents.groupby("Fwy"):
        sensor_group = sensors[sensors["Fwy"] == fwy]
        if sensor_group.empty:
            continue
        incident_part = incident_group[
            ["incident_id", "Abs PM", "dt", "Type", "Fwy"]
        ].rename(columns={"Abs PM": "incident_abs_pm"})
        sensor_part = sensor_group[
            ["station_id", "global_index", "Abs PM", "County", "Type", "Fwy"]
        ].rename(columns={"Abs PM": "sensor_abs_pm", "Type": "sensor_type"})
        merged = sensor_part.merge(incident_part, on="Fwy", how="inner")
        merged["distance"] = (merged["sensor_abs_pm"] - merged["incident_abs_pm"]).abs()
        merged = merged[merged["distance"] <= distance_threshold].sort_values("distance")
        rows.append(merged.groupby("incident_id", as_index=False).head(1))

    if not rows:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "station_id",
                "global_index",
                "dt",
                "Type",
                "Fwy",
                "sensor_abs_pm",
                "incident_abs_pm",
                "distance",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_time_index(year: int, months: Sequence[int]) -> pd.DatetimeIndex:
    pieces = []
    for month in months:
        days = calendar.monthrange(year, month)[1]
        pieces.append(
            pd.date_range(
                f"{year}-{month:02d}-01 00:00:00",
                periods=days * STEPS_PER_DAY,
                freq="5min",
            )
        )
    return pieces[0].append(pieces[1:]) if len(pieces) > 1 else pieces[0]


def add_time_features(flow: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    num_steps, num_nodes = flow.shape
    tod = ((times.hour * 60 + times.minute) / (24 * 60)).astype(np.float32).to_numpy()
    dow = times.dayofweek.astype(np.float32).to_numpy()
    data = np.empty((num_steps, num_nodes, 4), dtype=np.float32)
    data[:, :, 0] = flow.astype(np.float32, copy=False)
    data[:, :, 1] = tod[:, None]
    data[:, :, 2] = dow[:, None]
    data[:, :, 3] = 0.0
    return data


def load_county_flow(
    zf: zipfile.ZipFile,
    year: int,
    months: Sequence[int],
    global_indices: np.ndarray,
    traffic_channel: int,
    total_nodes: int,
) -> np.ndarray:
    month_arrays = []
    for month in months:
        member = month_member(zf, year, month)
        print(f"[traffic] loading {member}", flush=True)
        with zf.open(member) as fh:
            month_data = np.load(fh)
        if month_data.shape[0] == total_nodes:
            month_data = np.transpose(month_data, (1, 0, 2))
        if month_data.ndim != 3 or month_data.shape[1] != total_nodes:
            raise ValueError(
                f"Unexpected traffic shape for {member}: {month_data.shape}; "
                f"expected [T,{total_nodes},C] or [{total_nodes},T,C]"
            )
        month_arrays.append(month_data[:, global_indices, traffic_channel].astype(np.float32))
        del month_data
    return np.concatenate(month_arrays, axis=0)


def fill_event_channel(
    data: np.ndarray,
    matched: pd.DataFrame,
    node_lookup: Dict[int, int],
    times: pd.DatetimeIndex,
    event_window_slots: int,
) -> None:
    if matched.empty:
        return
    start = times[0]
    end = times[-1] + pd.Timedelta(minutes=5)
    active = matched[(matched["dt"] >= start) & (matched["dt"] < end)].copy()
    if active.empty:
        return
    slots = ((active["dt"] - start) / pd.Timedelta(minutes=5)).astype(int)
    for row, slot in zip(active.itertuples(index=False), slots):
        local_idx = node_lookup.get(int(row.global_index))
        if local_idx is None:
            continue
        slot_start = max(0, int(slot))
        slot_end = min(data.shape[0], slot_start + event_window_slots)
        data[slot_start:slot_end, local_idx, 3] = 1.0


def build_index(num_steps: int, input_len: int, output_len: int, split_ratio: Sequence[float]):
    num_samples = num_steps - input_len - output_len
    if num_samples <= 0:
        raise ValueError(
            f"Not enough timesteps: T={num_steps}, input_len={input_len}, output_len={output_len}"
        )
    starts = np.arange(num_samples, dtype=np.int64)
    index = np.stack(
        [starts, starts + input_len, starts + input_len + output_len],
        axis=-1,
    )
    train_end = int(split_ratio[0] * len(index))
    val_end = int((split_ratio[0] + split_ratio[1]) * len(index))
    return {
        "train": index[:train_end],
        "val": index[train_end:val_end],
        "test": index[val_end:],
    }


def write_dataset(
    out_dir: Path,
    data: np.ndarray,
    index: Dict[str, np.ndarray],
    meta: pd.DataFrame,
    matched: pd.DataFrame,
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "data.npz", data=data)
    np.savez_compressed(out_dir / "index.npz", **index)
    meta.to_csv(out_dir / "sensor_meta_feature.csv", index=False)
    matched.to_csv(out_dir / "matched_incidents.csv", index=False)
    with (out_dir / "preprocess_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def prepare_county(
    zf: zipfile.ZipFile,
    slug: str,
    meta: pd.DataFrame,
    incidents: pd.DataFrame,
    args: argparse.Namespace,
    months: Sequence[int],
    split_ratio: Sequence[float],
) -> None:
    county_name = COUNTY_LABELS[slug]
    selected_meta = meta[(meta["County"] == county_name) & (meta["Type"] == args.sensor_type)].copy()
    selected_meta = selected_meta.sort_values("global_index").reset_index(drop=True)
    if selected_meta.empty:
        raise ValueError(f"No sensors for county={county_name}, Type={args.sensor_type}")

    name = dataset_name(slug, args.year, months)
    out_dir = args.output_root / name
    if not args.overwrite and (out_dir / "data.npz").exists() and (out_dir / "index.npz").exists():
        print(f"[skip] {name} already exists. Use --overwrite to rewrite.", flush=True)
        return

    print(f"[county] {slug}: {len(selected_meta)} {args.sensor_type} sensors", flush=True)
    times = build_time_index(args.year, months)
    active_incidents = incidents[
        (incidents["dt"] >= times[0]) & (incidents["dt"] < times[-1] + pd.Timedelta(minutes=5))
    ].copy()
    candidate_meta = selected_meta if args.match_scope == "subset" else meta
    matched = official_style_match(active_incidents, candidate_meta, args.distance_threshold)
    selected_indices = set(selected_meta["global_index"].astype(int))
    matched = matched[matched["global_index"].astype(int).isin(selected_indices)].copy()

    global_indices = selected_meta["global_index"].to_numpy(dtype=np.int64)
    flow = load_county_flow(
        zf,
        args.year,
        months,
        global_indices,
        args.traffic_channel,
        total_nodes=len(meta),
    )
    if len(times) != flow.shape[0]:
        raise ValueError(f"Time index length {len(times)} does not match flow length {flow.shape[0]}")
    data = add_time_features(flow, times)
    node_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(global_indices)}
    fill_event_channel(data, matched, node_lookup, times, args.event_window_slots)
    index = build_index(data.shape[0], args.input_len, args.output_len, split_ratio)

    summary = {
        "dataset": name,
        "county": county_name,
        "sensor_type": args.sensor_type,
        "num_nodes": int(data.shape[1]),
        "num_timesteps": int(data.shape[0]),
        "features": ["flow", "time_of_day", "day_of_week", "accident_binary"],
        "traffic_channel": args.traffic_channel,
        "year": args.year,
        "months": list(months),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "split_ratio": list(split_ratio),
        "split_sizes": {key: int(len(value)) for key, value in index.items()},
        "official_matching": {
            "source": "XTraffic/process/traffic_incident_match.py",
            "rule": "same Fwy, nearest Abs PM within distance threshold",
            "distance_threshold": args.distance_threshold,
            "match_scope": args.match_scope,
        },
        "adapter_choices": {
            "county_subset": "sensor_meta_feature.csv County field",
            "sensor_filter": f"Type == {args.sensor_type}",
            "event_types": args.event_types,
            "accident_types": list(ACCIDENT_TYPES) if args.event_types == "accident" else "all",
            "event_window_slots": args.event_window_slots,
        },
        "matched_incidents": int(len(matched)),
        "event_active_slots": int(data[:, :, 3].sum()),
    }
    write_dataset(out_dir, data, index, selected_meta, matched, summary)
    print(
        f"[done] {name}: data={data.shape}, matched={len(matched)}, "
        f"event_slot_nodes={int(data[:, :, 3].sum())}, out={out_dir}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    months = parse_csv_ints(args.months)
    split_ratio = parse_csv_floats(args.split_ratio)
    counties = parse_counties(args.counties)
    args.output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        meta = load_metadata(zf)
        incidents = load_incidents(zf, args.year, args.event_types)
        print(
            f"[source] sensors={len(meta)}, incidents={len(incidents)}, "
            f"year={args.year}, months={months}",
            flush=True,
        )
        for slug in counties:
            prepare_county(zf, slug, meta, incidents, args, months, split_ratio)


if __name__ == "__main__":
    main()
