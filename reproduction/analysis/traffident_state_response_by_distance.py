#!/usr/bin/env python3
"""Distance-aware TraffiDent traffic-state response diagnostic.

This descriptive analysis expands each official matched incident to nearby
sensors on the same freeway and direction, then compares raw
flow/occupancy/speed windows with same-node same-time-of-week no-event
controls. It is not a training script and does not change the BasicTS dataset.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7
TYPE_ORDER = ["Hazard", "NoInj", "Other", "UnknInj", "1141", "AHazard", "Fire", "CarFire"]
RELATION_ORDER = ["upstream", "at_source", "downstream"]
EPS_PM = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/basicts/TraffiDent_D5_2023Q1_OfficialAll"),
        help="Official-script BasicTS adapter dataset with sensor_meta_feature.csv and matched_incidents.csv.",
    )
    parser.add_argument("--zip-path", type=Path, default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction/analysis/traffident_d5_state_response_by_distance"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument(
        "--channel-map",
        default="flow:0,occupancy:1,speed:2",
        help=(
            "Raw XTraffic channel mapping. Current data statistics indicate "
            "channel 0=flow, 1=occupancy/fraction, 2=speed."
        ),
    )
    parser.add_argument("--pre-slots", type=int, default=12)
    parser.add_argument("--post-slots", type=int, default=24)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--distance-bins", default="0,0.05,0.15,0.30,0.50")
    parser.add_argument("--same-direction-only", action="store_true", default=True)
    parser.add_argument("--allow-opposite-direction", dest="same_direction_only", action="store_false")
    parser.add_argument("--event-window-slots", type=int, default=2)
    parser.add_argument("--use-duration", action="store_true", default=True)
    parser.add_argument("--no-duration", dest="use_duration", action="store_false")
    parser.add_argument("--max-duration-slots", type=int, default=288)
    parser.add_argument("--min-controls", type=int, default=1)
    parser.add_argument("--control-weeks", default="-4,-3,-2,-1,1,2,3,4")
    parser.add_argument("--fill-missing", default="interpolate", choices=["interpolate", "zero", "none"])
    parser.add_argument("--min-type-cases", type=int, default=20)
    return parser.parse_args()


def parse_months(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_channel_map(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, idx = part.split(":", 1)
        out[name.strip()] = int(idx)
    for required in ("flow", "occupancy", "speed"):
        if required not in out:
            raise ValueError(f"--channel-map must include {required!r}; got {out}")
    return out


def parse_distance_bins(value: str) -> list[float]:
    bins = [float(part) for part in value.split(",") if part.strip()]
    if len(bins) < 2 or bins[0] != 0 or any(b <= a for a, b in zip(bins, bins[1:])):
        raise ValueError(f"Invalid distance bins: {bins}")
    return bins


def distance_bin_label(abs_pm: float, bins: Sequence[float]) -> str:
    for left, right in zip(bins, bins[1:]):
        if left <= abs_pm <= right:
            return f"{left:.2f}-{right:.2f}"
    return f">{bins[-1]:.2f}"


def build_time_index(year: int, months: Sequence[int]) -> pd.DatetimeIndex:
    pieces = []
    for month in months:
        days = calendar.monthrange(year, month)[1]
        pieces.append(
            pd.date_range(
                f"{year}-{month:02d}-01 00:00:00",
                periods=days * SLOTS_PER_DAY,
                freq="5min",
            )
        )
    return pieces[0].append(pieces[1:]) if len(pieces) > 1 else pieces[0]


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


def read_csv_member(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with zf.open(name) as probe:
        head = probe.read(4096)
    sep = "\t" if head.count(b"\t") > head.count(b",") else ","
    with zf.open(name) as fh:
        return pd.read_csv(fh, sep=sep, low_memory=False, **kwargs)


def load_raw_incidents(zf: zipfile.ZipFile, year: int) -> pd.DataFrame:
    incident_name = find_member(zf, rf"incidents_y{year}\.csv$")
    raw = read_csv_member(zf, incident_name, dtype={"incident_id": str})
    raw = raw.copy()
    raw["incident_id_clean"] = raw["incident_id"].astype(str).str.strip()
    raw["duration_minutes"] = pd.to_numeric(raw.get("duration"), errors="coerce")
    raw["raw_dt"] = pd.to_datetime(raw.get("dt"), errors="coerce")
    raw["raw_fwy"] = pd.to_numeric(raw.get("Fwy"), errors="coerce")
    raw["raw_incident_abs_pm"] = pd.to_numeric(raw.get("Abs PM"), errors="coerce")
    keep = [
        "incident_id_clean",
        "duration_minutes",
        "raw_dt",
        "raw_fwy",
        "raw_incident_abs_pm",
    ]
    for col in ["Freeway_direction", "DESCRIPTION", "LOCATION", "AREA", "Type"]:
        if col in raw.columns:
            keep.append(col)
    return raw[keep].drop_duplicates("incident_id_clean")


def load_raw_channels(
    zip_path: Path,
    year: int,
    months: Sequence[int],
    global_indices: np.ndarray,
    channel_map: dict[str, int],
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {name: [] for name in channel_map}
    min_nodes = int(global_indices.max()) + 1
    with zipfile.ZipFile(zip_path) as zf:
        for month in months:
            member = month_member(zf, year, month)
            print(f"[traffic] loading {member}", flush=True)
            with zf.open(member) as fh:
                month_data = np.load(fh)
            if month_data.ndim != 3:
                raise ValueError(f"Unexpected traffic rank for {member}: {month_data.shape}")
            if month_data.shape[1] >= min_nodes:
                pass
            elif month_data.shape[0] >= min_nodes:
                month_data = np.transpose(month_data, (1, 0, 2))
            else:
                raise ValueError(
                    f"Unexpected traffic shape for {member}: {month_data.shape}; "
                    f"neither axis can contain max global index {min_nodes - 1}"
                )
            for name, channel_idx in channel_map.items():
                parts[name].append(month_data[:, global_indices, channel_idx].astype(np.float32, copy=False))
            del month_data
    return {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}


def fill_missing(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none" or not np.isnan(values).any():
        return values.astype(np.float32, copy=False)
    if mode == "zero":
        return np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)
    if mode == "interpolate":
        return (
            pd.DataFrame(values)
            .interpolate(axis=0, limit_direction="both")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    raise ValueError(f"Unknown fill mode: {mode}")


def direction_sign(direction: object) -> int:
    text = str(direction).strip().upper()
    if text in {"N", "E"}:
        return 1
    if text in {"S", "W"}:
        return -1
    return 1


def relation_bucket(signed_downstream_pm: float) -> str:
    if signed_downstream_pm > EPS_PM:
        return "downstream"
    if signed_downstream_pm < -EPS_PM:
        return "upstream"
    return "at_source"


def slot_of(dt: pd.Timestamp, start_time: pd.Timestamp) -> int:
    return int((dt - start_time) / pd.Timedelta(minutes=5))


def ordered_types(types: Iterable[object]) -> list[str]:
    present = set(str(x) for x in types)
    ordered = [event_type for event_type in TYPE_ORDER if event_type in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def prepare_incidents(dataset_dir: Path, zip_path: Path, year: int, times: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(dataset_dir / "sensor_meta_feature.csv").copy()
    meta["node_idx"] = np.arange(len(meta), dtype=np.int64)
    meta["global_index"] = meta["global_index"].astype(int)
    meta["Fwy"] = pd.to_numeric(meta["Fwy"], errors="coerce")
    meta["Abs PM"] = pd.to_numeric(meta["Abs PM"], errors="coerce")
    meta["Direction"] = meta["Direction"].astype(str).str.strip().str.upper()

    matched = pd.read_csv(dataset_dir / "matched_incidents.csv", dtype={"incident_id": str})
    matched = matched.copy()
    matched["incident_id_clean"] = matched["incident_id"].astype(str).str.strip()
    matched["dt"] = pd.to_datetime(matched["dt"], errors="coerce")
    matched["Fwy"] = pd.to_numeric(matched["Fwy"], errors="coerce")
    matched["incident_abs_pm"] = pd.to_numeric(matched["incident_abs_pm"], errors="coerce")
    matched["global_index"] = matched["global_index"].astype(int)

    local = meta[["global_index", "Fwy", "Direction", "Abs PM", "node_idx"]].copy()
    matched = matched.merge(local, on=["global_index", "Fwy"], how="left", suffixes=("", "_matched_sensor"))

    with zipfile.ZipFile(zip_path) as zf:
        raw_incidents = load_raw_incidents(zf, year)
    matched = matched.merge(raw_incidents, on="incident_id_clean", how="left")
    matched["event_dt"] = matched["raw_dt"].fillna(matched["dt"])
    matched["event_fwy"] = matched["raw_fwy"].fillna(matched["Fwy"])
    matched["event_abs_pm"] = matched["raw_incident_abs_pm"].fillna(matched["incident_abs_pm"])
    matched["event_direction"] = matched["Freeway_direction"].fillna(matched["Direction"])
    matched["event_type"] = matched["Type_y"].fillna(matched["Type_x"] if "Type_x" in matched else matched["Type"])

    start = times[0]
    end = times[-1] + pd.Timedelta(minutes=5)
    incidents = matched.dropna(subset=["event_dt", "event_fwy", "event_abs_pm", "event_type"]).copy()
    incidents = incidents[(incidents["event_dt"] >= start) & (incidents["event_dt"] < end)].copy()
    incidents["event_slot"] = ((incidents["event_dt"] - start) / pd.Timedelta(minutes=5)).astype(int)
    incidents["event_direction"] = incidents["event_direction"].astype(str).str.strip().str.upper()
    incidents["event_fwy"] = incidents["event_fwy"].astype(int)
    incidents = incidents.drop_duplicates(
        subset=["incident_id_clean", "event_dt", "event_fwy", "event_abs_pm", "event_type", "event_direction"]
    ).reset_index(drop=True)
    return incidents, meta


def duration_slots(row: object, args: argparse.Namespace) -> int:
    base_slots = max(1, int(args.event_window_slots))
    if not args.use_duration:
        return base_slots
    duration = getattr(row, "duration_minutes", np.nan)
    if pd.isna(duration) or float(duration) <= 0:
        return base_slots
    slots = int(np.ceil(float(duration) / 5.0))
    return max(base_slots, min(int(args.max_duration_slots), slots))


def affected_nodes(meta: pd.DataFrame, incident: object, args: argparse.Namespace) -> pd.DataFrame:
    direction = str(incident.event_direction).strip().upper()
    same = meta["Fwy"].to_numpy(dtype=float) == float(incident.event_fwy)
    if args.same_direction_only and direction:
        same &= meta["Direction"].astype(str).to_numpy() == direction
    candidates = meta.loc[same, ["node_idx", "global_index", "station_id", "Direction", "Abs PM"]].copy()
    candidates = candidates.rename(columns={"Abs PM": "sensor_abs_pm"})
    if candidates.empty:
        return candidates
    signs = np.asarray([direction_sign(direction if args.same_direction_only else x) for x in candidates["Direction"]])
    candidates["signed_downstream_pm"] = (candidates["sensor_abs_pm"].astype(float) - float(incident.event_abs_pm)) * signs
    candidates["abs_distance_pm"] = candidates["signed_downstream_pm"].abs()
    candidates = candidates[candidates["abs_distance_pm"] <= float(args.max_distance)].copy()
    candidates["relation"] = candidates["signed_downstream_pm"].map(relation_bucket)
    return candidates


def build_expanded_event_mask(
    incidents: pd.DataFrame,
    meta: pd.DataFrame,
    num_steps: int,
    num_nodes: int,
    args: argparse.Namespace,
) -> np.ndarray:
    mask = np.zeros((num_steps, num_nodes), dtype=bool)
    for incident in incidents.itertuples(index=False):
        slot = int(incident.event_slot)
        if slot < 0 or slot >= num_steps:
            continue
        span = duration_slots(incident, args)
        nodes = affected_nodes(meta, incident, args)
        if nodes.empty:
            continue
        left = max(0, slot)
        right = min(num_steps, slot + span)
        mask[left:right, nodes["node_idx"].to_numpy(dtype=np.int64)] = True
    return mask


def collect_controls(
    channels: dict[str, np.ndarray],
    expanded_event_mask: np.ndarray,
    node_idx: int,
    slot: int,
    pre_slots: int,
    post_slots: int,
    control_weeks: Sequence[int],
) -> list[dict[str, np.ndarray]]:
    controls: list[dict[str, np.ndarray]] = []
    for week_offset in control_weeks:
        control_slot = slot + week_offset * SLOTS_PER_WEEK
        left = control_slot - pre_slots
        right = control_slot + post_slots
        if left < 0 or right >= expanded_event_mask.shape[0]:
            continue
        if expanded_event_mask[left : right + 1, node_idx].any():
            continue
        window: dict[str, np.ndarray] = {}
        ok = True
        for name, values in channels.items():
            series = values[left : right + 1, node_idx]
            if np.isnan(series).any():
                ok = False
                break
            window[name] = series.astype(np.float64, copy=False)
        if ok:
            controls.append(window)
    return controls


def state_scores(delta: dict[str, np.ndarray], scales: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    z_speed = delta["speed"] / max(scales["speed"], 1e-6)
    z_flow = delta["flow"] / max(scales["flow"], 1e-6)
    z_occ = delta["occupancy"] / max(scales["occupancy"], 1e-6)
    congestion_shift = -z_speed - z_flow + z_occ
    displacement = np.sqrt(z_speed**2 + z_flow**2 + z_occ**2)
    return congestion_shift, displacement


def relation_distance_label(relation: str, distance_bin: str) -> str:
    if relation == "at_source":
        return "at_source"
    return f"{relation}:{distance_bin}"


def summarize_cases(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["event_type", "relation", "distance_bin"]
    summary = (
        cases.groupby(group_cols, dropna=False)
        .agg(
            n_event_nodes=("incident_id", "count"),
            n_events=("incident_id", "nunique"),
            n_nodes=("node_idx", "nunique"),
            mean_signed_pm=("signed_downstream_pm", "mean"),
            mean_abs_pm=("abs_distance_pm", "mean"),
            median_duration_min=("duration_minutes", "median"),
            post_0_60_delta_speed=("post_0_60_delta_speed", "mean"),
            post_0_60_delta_flow=("post_0_60_delta_flow", "mean"),
            post_0_60_delta_occupancy=("post_0_60_delta_occupancy", "mean"),
            post_0_60_congestion_shift=("post_0_60_congestion_shift", "mean"),
            post_0_60_state_displacement=("post_0_60_state_displacement", "mean"),
            frac_congestion_shift_positive=("post_0_60_congestion_shift", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    type_summary = (
        cases.groupby("event_type")
        .agg(
            n_event_nodes=("incident_id", "count"),
            n_events=("incident_id", "nunique"),
            n_nodes=("node_idx", "nunique"),
            post_0_60_delta_speed=("post_0_60_delta_speed", "mean"),
            post_0_60_delta_flow=("post_0_60_delta_flow", "mean"),
            post_0_60_delta_occupancy=("post_0_60_delta_occupancy", "mean"),
            post_0_60_congestion_shift=("post_0_60_congestion_shift", "mean"),
            post_0_60_state_displacement=("post_0_60_state_displacement", "mean"),
            frac_congestion_shift_positive=("post_0_60_congestion_shift", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    type_summary["event_type"] = pd.Categorical(
        type_summary["event_type"],
        categories=ordered_types(type_summary["event_type"]),
        ordered=True,
    )
    return summary, type_summary.sort_values("event_type").reset_index(drop=True)


def plot_relation_curves(delta_df: pd.DataFrame, cases: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("delta_speed", "Speed residual"),
        ("delta_flow", "Flow residual"),
        ("delta_occupancy", "Occupancy residual"),
        ("congestion_shift", "Congestion-state shift"),
    ]
    relations = [r for r in RELATION_ORDER if r in set(delta_df["relation"])]
    fig, axes = plt.subplots(len(metrics), len(relations), figsize=(4.1 * len(relations), 9.4), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, None]
    for col, relation in enumerate(relations):
        part = delta_df[delta_df["relation"] == relation]
        med_duration = cases.loc[cases["relation"] == relation, "duration_minutes"].replace(0, np.nan).median()
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            curve = (
                part.groupby("rel_min")[metric]
                .agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
                .reset_index()
                .sort_values("rel_min")
            )
            x = curve["rel_min"].to_numpy()
            y = curve["mean"].to_numpy()
            ax.plot(x, y, color="#3a6ea5", linewidth=2.0)
            ax.fill_between(x, curve["q25"].to_numpy(), curve["q75"].to_numpy(), color="#3a6ea5", alpha=0.16)
            if np.isfinite(med_duration):
                ax.axvspan(0, min(float(med_duration), float(x.max())), color="#c43c39", alpha=0.08)
            ax.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
            ax.axhline(0, color="#777777", linestyle=":", linewidth=1.0)
            ax.grid(True, color="#ececec", linewidth=0.8)
            if row == 0:
                n_cases = int(cases.loc[cases["relation"] == relation, "incident_id"].count())
                ax.set_title(f"{relation} (event-nodes={n_cases})")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(metrics) - 1:
                ax.set_xlabel("Minutes relative to incident")
    fig.suptitle("Traffic-state residuals by upstream/downstream relation", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "d5_state_response_curves_by_relation.png", dpi=180)
    fig.savefig(output_dir / "d5_state_response_curves_by_relation.pdf")
    plt.close(fig)


def plot_signed_distance_heatmap(delta_df: pd.DataFrame, output_dir: Path, value_col: str, filename: str, title: str) -> None:
    labels = delta_df["relation_distance"].dropna().unique().tolist()
    upstream = sorted([x for x in labels if x.startswith("upstream")], reverse=True)
    downstream = sorted([x for x in labels if x.startswith("downstream")])
    ordered = upstream + (["at_source"] if "at_source" in labels else []) + downstream
    rel_mins = sorted(delta_df["rel_min"].unique())
    matrix = np.full((len(ordered), len(rel_mins)), np.nan)
    pivot = delta_df.groupby(["relation_distance", "rel_min"])[value_col].mean().reset_index()
    label_to_i = {label: idx for idx, label in enumerate(ordered)}
    rel_to_j = {rel: idx for idx, rel in enumerate(rel_mins)}
    for row in pivot.itertuples(index=False):
        matrix[label_to_i[str(row.relation_distance)], rel_to_j[int(row.rel_min)]] = float(getattr(row, value_col))
    vmax = max(float(np.nanpercentile(np.abs(matrix), 95)), 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(11, max(4.0, 0.36 * len(ordered) + 2.0)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(0, len(rel_mins), max(1, len(rel_mins) // 8)))
    ax.set_xticklabels([str(rel_mins[i]) for i in ax.get_xticks().astype(int)])
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered)
    if 0 in rel_to_j:
        ax.axvline(rel_to_j[0], color="#111111", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Minutes relative to incident")
    ax.set_ylabel("Signed relation and distance bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=180)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_type_relation_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    eligible = summary[summary["n_event_nodes"] >= 10].copy()
    if eligible.empty:
        return
    eligible["relation_distance"] = [
        relation_distance_label(str(r), str(d)) for r, d in zip(eligible["relation"], eligible["distance_bin"])
    ]
    event_types = ordered_types(eligible["event_type"])
    labels = eligible["relation_distance"].dropna().unique().tolist()
    upstream = sorted([x for x in labels if x.startswith("upstream")], reverse=True)
    downstream = sorted([x for x in labels if x.startswith("downstream")])
    columns = upstream + (["at_source"] if "at_source" in labels else []) + downstream
    matrix = np.full((len(event_types), len(columns)), np.nan)
    counts = np.zeros((len(event_types), len(columns)), dtype=np.int64)
    type_to_i = {name: idx for idx, name in enumerate(event_types)}
    col_to_j = {name: idx for idx, name in enumerate(columns)}
    for row in eligible.itertuples(index=False):
        i = type_to_i[str(row.event_type)]
        j = col_to_j[str(row.relation_distance)]
        matrix[i, j] = float(row.post_0_60_congestion_shift)
        counts[i, j] = int(row.n_event_nodes)
    vmax = max(float(np.nanpercentile(np.abs(matrix), 90)), 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(max(9.0, 1.2 * len(columns)), max(4.8, 0.55 * len(event_types) + 2.0)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.set_xlabel("Relation and distance bin")
    ax.set_ylabel("Incident type")
    ax.set_title("Post 0-60 min congestion-state shift by type and signed distance")
    for i in range(len(event_types)):
        for j in range(len(columns)):
            if counts[i, j] > 0 and np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}\n(n={counts[i, j]})", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="post_0_60_congestion_shift")
    fig.tight_layout()
    fig.savefig(output_dir / "d5_state_type_relation_distance_heatmap.png", dpi=180)
    fig.savefig(output_dir / "d5_state_type_relation_distance_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    channel_map = parse_channel_map(args.channel_map)
    distance_bins = parse_distance_bins(args.distance_bins)
    control_weeks = [int(part) for part in args.control_weeks.split(",") if part.strip()]
    times = build_time_index(args.year, months)

    incidents, meta = prepare_incidents(args.dataset_dir, args.zip_path, args.year, times)
    if incidents.empty:
        raise RuntimeError("No incidents available after official matched incident merge.")
    global_indices = meta["global_index"].to_numpy(dtype=np.int64)
    channels = load_raw_channels(args.zip_path, args.year, months, global_indices, channel_map)
    channels = {name: fill_missing(values, args.fill_missing) for name, values in channels.items()}
    num_steps = next(iter(channels.values())).shape[0]
    num_nodes = next(iter(channels.values())).shape[1]
    if len(times) != num_steps:
        raise ValueError(f"Time index length {len(times)} != raw traffic length {num_steps}")
    scales = {name: float(np.nanstd(values)) for name, values in channels.items()}
    expanded_event_mask = build_expanded_event_mask(incidents, meta, num_steps, num_nodes, args)

    rel_slots = np.arange(-args.pre_slots, args.post_slots + 1)
    pre_mask = (rel_slots >= -args.pre_slots) & (rel_slots <= -1)
    post_0_60_mask = (rel_slots >= 0) & (rel_slots <= min(12, args.post_slots))

    case_rows: list[dict] = []
    delta_rows: list[dict] = []
    for incident in incidents.itertuples(index=False):
        slot = int(incident.event_slot)
        left = slot - args.pre_slots
        right = slot + args.post_slots
        if left < 0 or right >= num_steps:
            continue
        nodes = affected_nodes(meta, incident, args)
        if nodes.empty:
            continue
        span_slots = duration_slots(incident, args)
        duration_min = float(span_slots * 5)
        if not pd.isna(getattr(incident, "duration_minutes", np.nan)):
            duration_min = float(getattr(incident, "duration_minutes"))
        for node in nodes.itertuples(index=False):
            node_idx = int(node.node_idx)
            event_window = {name: values[left : right + 1, node_idx].astype(np.float64, copy=False) for name, values in channels.items()}
            if any(np.isnan(values).any() for values in event_window.values()):
                continue
            controls = collect_controls(
                channels,
                expanded_event_mask,
                node_idx,
                slot,
                args.pre_slots,
                args.post_slots,
                control_weeks,
            )
            if len(controls) < args.min_controls:
                continue
            control_mean = {
                name: np.nanmean(np.vstack([control[name] for control in controls]), axis=0)
                for name in channels
            }
            delta = {name: event_window[name] - control_mean[name] for name in channels}
            congestion_shift, displacement = state_scores(delta, scales)
            relation = str(node.relation)
            dist_bin = distance_bin_label(float(node.abs_distance_pm), distance_bins)
            rel_dist = relation_distance_label(relation, dist_bin)
            case_rows.append(
                {
                    "incident_id": str(incident.incident_id_clean),
                    "event_type": str(incident.event_type),
                    "event_dt": pd.Timestamp(incident.event_dt).isoformat(),
                    "event_fwy": int(incident.event_fwy),
                    "event_direction": str(incident.event_direction),
                    "event_abs_pm": float(incident.event_abs_pm),
                    "duration_minutes": duration_min,
                    "node_idx": node_idx,
                    "global_index": int(node.global_index),
                    "station_id": str(node.station_id),
                    "sensor_direction": str(node.Direction),
                    "sensor_abs_pm": float(node.sensor_abs_pm),
                    "signed_downstream_pm": float(node.signed_downstream_pm),
                    "abs_distance_pm": float(node.abs_distance_pm),
                    "relation": relation,
                    "distance_bin": dist_bin,
                    "relation_distance": rel_dist,
                    "n_controls": int(len(controls)),
                    "pre_delta_speed": float(delta["speed"][pre_mask].mean()),
                    "pre_delta_flow": float(delta["flow"][pre_mask].mean()),
                    "pre_delta_occupancy": float(delta["occupancy"][pre_mask].mean()),
                    "post_0_60_delta_speed": float(delta["speed"][post_0_60_mask].mean()),
                    "post_0_60_delta_flow": float(delta["flow"][post_0_60_mask].mean()),
                    "post_0_60_delta_occupancy": float(delta["occupancy"][post_0_60_mask].mean()),
                    "post_0_60_congestion_shift": float(congestion_shift[post_0_60_mask].mean()),
                    "post_0_60_state_displacement": float(displacement[post_0_60_mask].mean()),
                    "description": str(getattr(incident, "DESCRIPTION", "")),
                    "location": str(getattr(incident, "LOCATION", "")),
                }
            )
            for rel, rel_min, idx in zip(rel_slots, rel_slots * 5, range(len(rel_slots))):
                delta_rows.append(
                    {
                        "incident_id": str(incident.incident_id_clean),
                        "event_type": str(incident.event_type),
                        "node_idx": node_idx,
                        "rel_slot": int(rel),
                        "rel_min": int(rel_min),
                        "signed_downstream_pm": float(node.signed_downstream_pm),
                        "abs_distance_pm": float(node.abs_distance_pm),
                        "relation": relation,
                        "distance_bin": dist_bin,
                        "relation_distance": rel_dist,
                        "delta_speed": float(delta["speed"][idx]),
                        "delta_flow": float(delta["flow"][idx]),
                        "delta_occupancy": float(delta["occupancy"][idx]),
                        "congestion_shift": float(congestion_shift[idx]),
                        "state_displacement": float(displacement[idx]),
                    }
                )

    cases = pd.DataFrame(case_rows)
    delta_df = pd.DataFrame(delta_rows)
    if cases.empty or delta_df.empty:
        raise RuntimeError("No event-node windows with controls were collected.")

    summary, type_summary = summarize_cases(cases)
    cases.to_csv(args.output_dir / "state_response_cases.csv", index=False)
    delta_df.to_csv(args.output_dir / "state_response_delta_windows.csv", index=False)
    summary.to_csv(args.output_dir / "state_response_by_type_relation_distance.csv", index=False)
    type_summary.to_csv(args.output_dir / "state_response_by_type.csv", index=False)

    plot_relation_curves(delta_df, cases, args.output_dir)
    plot_signed_distance_heatmap(
        delta_df,
        args.output_dir,
        "congestion_shift",
        "d5_state_congestion_shift_signed_distance_heatmap",
        "Congestion-state shift by signed distance and relative time",
    )
    plot_signed_distance_heatmap(
        delta_df,
        args.output_dir,
        "delta_speed",
        "d5_state_speed_residual_signed_distance_heatmap",
        "Speed residual by signed distance and relative time",
    )
    plot_type_relation_heatmap(summary, args.output_dir)

    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "zip_path": str(args.zip_path),
        "year": args.year,
        "months": months,
        "channel_map": channel_map,
        "channel_scales": scales,
        "same_direction_only": bool(args.same_direction_only),
        "signed_downstream_pm_rule": "(sensor_abs_pm - incident_abs_pm) * sign(event_direction), sign=+1 for N/E and -1 for S/W",
        "distance_bins": distance_bins,
        "max_distance": args.max_distance,
        "use_duration": bool(args.use_duration),
        "event_window_slots": args.event_window_slots,
        "max_duration_slots": args.max_duration_slots,
        "control_weeks": control_weeks,
        "matched_unique_incidents": int(len(incidents)),
        "event_node_cases": int(len(cases)),
        "events_with_cases": int(cases["incident_id"].nunique()),
        "nodes_with_cases": int(cases["node_idx"].nunique()),
        "type_summary": type_summary.to_dict(orient="records"),
        "outputs": [
            "state_response_cases.csv",
            "state_response_delta_windows.csv",
            "state_response_by_type_relation_distance.csv",
            "state_response_by_type.csv",
            "d5_state_response_curves_by_relation.png",
            "d5_state_congestion_shift_signed_distance_heatmap.png",
            "d5_state_speed_residual_signed_distance_heatmap.png",
            "d5_state_type_relation_distance_heatmap.png",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
