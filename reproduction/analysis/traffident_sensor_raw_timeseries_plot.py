#!/usr/bin/env python3
"""Plot raw flow time series for a TraffiDent sensor with incident markers.

This diagnostic complements event-aligned aggregate plots by showing the
original 5-minute flow sequence for one matched sensor. It selects the most
frequently matched sensor by default, marks all official matched incidents on
the raw timeline, and zooms into one incident with same-time-of-week controls.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7


TYPE_COLORS = {
    "Hazard": "#4e79a7",
    "NoInj": "#e15759",
    "Other": "#59a14f",
    "UnknInj": "#f28e2b",
    "1141": "#b07aa1",
    "AHazard": "#76b7b2",
    "Fire": "#edc948",
    "CarFire": "#9c755f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/basicts/TraffiDent_D5_2023Q1_OfficialAll"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction/analysis/traffident_d5_sensor_raw_timeseries"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument("--station-id", type=int, default=None)
    parser.add_argument("--incident-id", type=int, default=None)
    parser.add_argument("--zoom-hours", type=float, default=24.0)
    parser.add_argument("--control-hours", type=float, default=6.0)
    parser.add_argument("--event-window-slots", type=int, default=2)
    parser.add_argument(
        "--raw-incident-zip",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"),
        help="Official XTraffic zip used to recover incident duration and descriptions.",
    )
    parser.add_argument("--raw-incident-file", default=None, help="Defaults to incidents_y{year}.csv.")
    parser.add_argument(
        "--control-weeks",
        default="-4,-3,-2,-1,1,2,3,4",
        help="Same-time-of-week offsets used as no-event controls.",
    )
    return parser.parse_args()


def build_time_index(year: int, months: list[int]) -> pd.DatetimeIndex:
    pieces = []
    for month in months:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        pieces.append(pd.date_range(start, end, freq="5min", inclusive="left"))
    return pieces[0].append(pieces[1:]) if len(pieces) > 1 else pieces[0]


def slot_of(dt: pd.Timestamp, start_time: pd.Timestamp) -> int:
    return int((dt - start_time) / pd.Timedelta(minutes=5))


def is_clear(event_channel: np.ndarray, node_idx: int, left: int, right: int) -> bool:
    if left < 0 or right >= event_channel.shape[0]:
        return False
    return bool(np.nanmax(event_channel[left : right + 1, node_idx]) == 0.0)


def collect_controls(
    flow: np.ndarray,
    event_channel: np.ndarray,
    node_idx: int,
    slot: int,
    half_slots: int,
    control_weeks: list[int],
) -> list[dict]:
    controls = []
    for week_offset in control_weeks:
        control_slot = slot + week_offset * SLOTS_PER_WEEK
        left = control_slot - half_slots
        right = control_slot + half_slots
        if left < 0 or right >= flow.shape[0]:
            continue
        if not is_clear(event_channel, node_idx, left, right):
            continue
        window = flow[left : right + 1, node_idx]
        if np.isnan(window).any():
            continue
        controls.append({"week_offset": week_offset, "slot": control_slot, "window": window})
    return controls


def load_raw_incident_metadata(zip_path: Path, member_name: str) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame(columns=["incident_id_clean", "duration_minutes", "DESCRIPTION", "LOCATION"])
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            raw = pd.read_csv(f, sep="\t", dtype={"incident_id": str}, low_memory=False)
    raw["incident_id_clean"] = raw["incident_id"].astype(str).str.strip()
    raw["duration_minutes"] = pd.to_numeric(raw.get("duration"), errors="coerce")
    keep_cols = ["incident_id_clean", "duration_minutes"]
    for col in ["DESCRIPTION", "LOCATION", "AREA", "Freeway_direction"]:
        if col in raw.columns:
            keep_cols.append(col)
    return raw[keep_cols].drop_duplicates("incident_id_clean")


def duration_for(row: pd.Series | object, fallback_minutes: float) -> float:
    if isinstance(row, pd.Series):
        value = row.get("duration_minutes", np.nan)
    else:
        value = getattr(row, "duration_minutes", np.nan)
    if pd.isna(value) or float(value) <= 0:
        return fallback_minutes
    return float(value)


def choose_incident(
    station_events: pd.DataFrame,
    flow: np.ndarray,
    event_channel: np.ndarray,
    node_idx: int,
    start_time: pd.Timestamp,
    half_slots: int,
    control_weeks: list[int],
) -> tuple[pd.Series, list[dict], float | None]:
    candidates = []
    for _, row in station_events.iterrows():
        slot = slot_of(row["dt"], start_time)
        left = slot - half_slots
        right = slot + half_slots
        if left < 0 or right >= flow.shape[0]:
            continue
        incident_window = flow[left : right + 1, node_idx]
        if np.isnan(incident_window).any():
            continue
        controls = collect_controls(flow, event_channel, node_idx, slot, half_slots, control_weeks)
        if not controls:
            continue
        control_mean = np.nanmean(np.vstack([c["window"] for c in controls]), axis=0)
        post = slice(half_slots, min(half_slots + 12, len(incident_window)))
        post_delta = float(np.nanmean(incident_window[post] - control_mean[post]))
        candidates.append((post_delta, row, controls))

    if candidates:
        post_delta, row, controls = min(candidates, key=lambda item: item[0])
        return row, controls, post_delta
    return station_events.iloc[0], [], None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    months = [int(part) for part in args.months.split(",") if part.strip()]
    control_weeks = [int(part) for part in args.control_weeks.split(",") if part.strip()]
    times = build_time_index(args.year, months)

    data = np.load(args.dataset_dir / "data.npz")["data"]
    flow = data[:, :, 0].astype(np.float64, copy=False)
    event_channel = data[:, :, 3].astype(np.float32, copy=False)
    meta = pd.read_csv(args.dataset_dir / "sensor_meta_feature.csv")
    matched = pd.read_csv(args.dataset_dir / "matched_incidents.csv")
    matched["dt"] = pd.to_datetime(matched["dt"], errors="coerce")
    matched["incident_id_clean"] = matched["incident_id"].astype(str).str.strip()
    raw_member = args.raw_incident_file or f"incidents_y{args.year}.csv"
    raw_incidents = load_raw_incident_metadata(args.raw_incident_zip, raw_member)
    matched = matched.merge(raw_incidents, on="incident_id_clean", how="left")
    matched = matched.dropna(subset=["dt"]).copy()

    if len(times) != flow.shape[0]:
        raise ValueError(f"Time index length {len(times)} != data length {flow.shape[0]}")

    station_id = args.station_id
    if station_id is None:
        station_id = int(matched["station_id"].value_counts().idxmax())

    meta_match = meta.index[meta["station_id"].astype(int) == int(station_id)]
    if len(meta_match) == 0:
        raise ValueError(f"station_id={station_id} not found in sensor_meta_feature.csv")
    node_idx = int(meta_match[0])
    sensor_meta = meta.loc[node_idx].to_dict()

    station_events = matched[matched["station_id"].astype(int) == int(station_id)].sort_values("dt").copy()
    if station_events.empty:
        raise ValueError(f"station_id={station_id} has no matched incidents")
    matched_sensor_type = ""
    if "sensor_type" in station_events.columns:
        sensor_type_values = station_events["sensor_type"].dropna().astype(str)
        if not sensor_type_values.empty:
            matched_sensor_type = sensor_type_values.iloc[0]

    if args.incident_id is not None:
        chosen = station_events[station_events["incident_id"].astype(int) == int(args.incident_id)]
        if chosen.empty:
            raise ValueError(f"incident_id={args.incident_id} not found for station_id={station_id}")
        chosen_row = chosen.iloc[0]
        half_control_slots = int(round(args.control_hours * 12))
        selected_slot = slot_of(chosen_row["dt"], times[0])
        controls = collect_controls(flow, event_channel, node_idx, selected_slot, half_control_slots, control_weeks)
        selected_post_delta = None
    else:
        half_control_slots = int(round(args.control_hours * 12))
        chosen_row, controls, selected_post_delta = choose_incident(
            station_events,
            flow,
            event_channel,
            node_idx,
            times[0],
            half_control_slots,
            control_weeks,
        )

    selected_dt = pd.Timestamp(chosen_row["dt"])
    selected_slot = slot_of(selected_dt, times[0])
    selected_incident_id = int(chosen_row["incident_id"])
    flow_series = flow[:, node_idx]
    event_series = event_channel[:, node_idx]

    # Save raw sensor series for reproducibility.
    sensor_df = pd.DataFrame({"time": times, "flow": flow_series, "event_active": event_series})
    sensor_df.to_csv(args.output_dir / f"sensor_{station_id}_raw_flow_q1.csv", index=False)

    event_rows = []
    for event in station_events.itertuples(index=False):
        slot = slot_of(pd.Timestamp(event.dt), times[0])
        duration_minutes = duration_for(event, 5 * args.event_window_slots)
        event_rows.append(
            {
                "incident_id": int(event.incident_id),
                "time": pd.Timestamp(event.dt).isoformat(),
                "slot": slot,
                "type": str(event.Type),
                "duration_minutes": duration_minutes,
                "flow_at_event": float(flow_series[slot]) if 0 <= slot < len(flow_series) else np.nan,
                "distance": float(event.distance),
                "description": str(getattr(event, "DESCRIPTION", "")),
                "location": str(getattr(event, "LOCATION", "")),
            }
        )
    pd.DataFrame(event_rows).to_csv(args.output_dir / f"sensor_{station_id}_matched_incidents.csv", index=False)

    zoom_slots = int(round(args.zoom_hours * 12))
    zoom_left = max(0, selected_slot - zoom_slots)
    zoom_right = min(len(times) - 1, selected_slot + zoom_slots)
    rel_slots = np.arange(-half_control_slots, half_control_slots + 1)
    selected_left = selected_slot - half_control_slots
    selected_right = selected_slot + half_control_slots
    selected_window = flow[selected_left : selected_right + 1, node_idx]

    control_rows = []
    for rel, value in zip(rel_slots, selected_window):
        control_rows.append(
            {
                "kind": "selected_incident",
                "week_offset": 0,
                "rel_min": int(rel * 5),
                "flow": float(value),
            }
        )
    for control in controls:
        for rel, value in zip(rel_slots, control["window"]):
            control_rows.append(
                {
                    "kind": "control",
                    "week_offset": int(control["week_offset"]),
                    "rel_min": int(rel * 5),
                    "flow": float(value),
                }
            )
    selected_controls_name = f"sensor_{station_id}_incident_{selected_incident_id}_controls.csv"
    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(args.output_dir / selected_controls_name, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.4, 1.2, 1.1]})
    fig.suptitle(
        f"Raw flow for station {station_id} "
        f"(local node {node_idx}, global {int(sensor_meta['global_index'])}, {sensor_meta.get('County', '')})",
        fontsize=13,
    )

    # Full Q1 raw timeline.
    axes[0].plot(times, flow_series, color="#2f6f9f", linewidth=0.55)
    for event in station_events.itertuples(index=False):
        color = TYPE_COLORS.get(str(event.Type), "#555555")
        axes[0].axvline(pd.Timestamp(event.dt), color=color, alpha=0.35, linewidth=0.8)
    axes[0].set_ylabel("Flow")
    axes[0].set_title(f"Full 2023 Q1 raw 5-min flow with {len(station_events)} matched incidents")
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Zoom around selected incident.
    zoom_times = times[zoom_left : zoom_right + 1]
    axes[1].plot(zoom_times, flow_series[zoom_left : zoom_right + 1], color="#2f6f9f", linewidth=1.1)
    zoom_start = times[zoom_left]
    zoom_end = times[zoom_right]
    zoom_events = station_events[(station_events["dt"] >= zoom_start) & (station_events["dt"] <= zoom_end)]
    for event in zoom_events.itertuples(index=False):
        color = TYPE_COLORS.get(str(event.Type), "#555555")
        event_time = pd.Timestamp(event.dt)
        duration_minutes = duration_for(event, 5 * args.event_window_slots)
        event_end = event_time + pd.Timedelta(minutes=duration_minutes)
        axes[1].axvline(event_time, color=color, linestyle="--", alpha=0.75, linewidth=1.0)
        axes[1].axvspan(event_time, event_end, color=color, alpha=0.12)
        axes[1].text(event_time, axes[1].get_ylim()[1], str(event.Type), color=color, rotation=90, va="top", fontsize=8)
    axes[1].set_ylabel("Flow")
    axes[1].set_title(
        f"Zoom around selected incident {int(chosen_row['incident_id'])}: "
        f"{chosen_row['Type']} at {selected_dt:%Y-%m-%d %H:%M}"
    )
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # Same sensor incident vs same-time-of-week control windows.
    rel_min = rel_slots * 5
    for control in controls:
        axes[2].plot(rel_min, control["window"], color="#b0b0b0", alpha=0.55, linewidth=0.9)
    if controls:
        control_mean = np.nanmean(np.vstack([c["window"] for c in controls]), axis=0)
        axes[2].plot(rel_min, control_mean, color="#2f6f9f", linewidth=2.0, label="Control mean")
    axes[2].plot(rel_min, selected_window, color="#c43c39", linewidth=2.2, label="Selected incident")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1.0)
    selected_duration = duration_for(chosen_row, 5 * args.event_window_slots)
    axes[2].axvspan(
        0,
        min(selected_duration, float(rel_min[-1])),
        color=TYPE_COLORS.get(str(chosen_row["Type"]), "#555555"),
        alpha=0.14,
        label=f"Reported duration ({selected_duration:.0f} min)",
    )
    axes[2].set_xlabel("Minutes relative to selected incident")
    axes[2].set_ylabel("Flow")
    axes[2].set_title("Same sensor: selected raw window vs same-time-of-week no-event controls")
    axes[2].legend(frameon=False, loc="best")

    for ax in axes:
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_base = args.output_dir / f"sensor_{station_id}_incident_{selected_incident_id}_raw_timeseries"
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "station_id": int(station_id),
        "node_idx": int(node_idx),
        "global_index": int(sensor_meta["global_index"]),
        "county": str(sensor_meta.get("County", "")),
        "fwy": str(sensor_meta.get("Fwy", "")),
        "sensor_type": matched_sensor_type or str(sensor_meta.get("Sensor Type", "")),
        "selection_rule": "minimum post-60-min flow delta vs same-time-of-week controls"
        if args.incident_id is None
        else "user-specified incident_id",
        "matched_incidents_for_sensor": int(len(station_events)),
        "selected_incident_id": selected_incident_id,
        "selected_incident_type": str(chosen_row["Type"]),
        "selected_incident_time": selected_dt.isoformat(),
        "selected_incident_distance": float(chosen_row["distance"]),
        "selected_incident_duration_minutes": float(selected_duration),
        "selected_incident_description": str(chosen_row.get("DESCRIPTION", "")),
        "selected_incident_location": str(chosen_row.get("LOCATION", "")),
        "selected_post_delta_vs_controls": selected_post_delta,
        "n_controls": int(len(controls)),
        "outputs": [
            out_base.with_suffix(".png").name,
            out_base.with_suffix(".pdf").name,
            f"sensor_{station_id}_raw_flow_q1.csv",
            f"sensor_{station_id}_matched_incidents.csv",
            selected_controls_name,
        ],
    }
    with (args.output_dir / f"manifest_sensor_{station_id}_incident_{selected_incident_id}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
