#!/usr/bin/env python3
"""Visualize event-aligned TraffiDent flow impact.

The plot aligns each matched incident at t=0 for its matched sensor, then
compares the observed flow trajectory with same-node same-time-of-week
no-event control windows. This is a descriptive diagnostic, not a causal
estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7


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
        default=Path("reproduction/analysis/traffident_d5_incident_flow_impact"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument("--pre-slots", type=int, default=12, help="Slots before incident, 12 = 60 minutes.")
    parser.add_argument("--post-slots", type=int, default=24, help="Slots after incident, 24 = 120 minutes.")
    parser.add_argument(
        "--control-weeks",
        default="-4,-3,-2,-1,1,2,3,4",
        help="Same-time-of-week offsets used as controls.",
    )
    parser.add_argument(
        "--min-type-cases",
        type=int,
        default=20,
        help="Minimum incident windows for a type-specific panel.",
    )
    return parser.parse_args()


def build_time_index(year: int, months: list[int]) -> pd.DatetimeIndex:
    pieces = []
    for month in months:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        pieces.append(pd.date_range(start, end, freq="5min", inclusive="left"))
    return pieces[0].append(pieces[1:]) if len(pieces) > 1 else pieces[0]


def window_is_clear(event_channel: np.ndarray, node_idx: int, start: int, end: int) -> bool:
    return bool(np.nanmax(event_channel[start : end + 1, node_idx]) == 0.0)


def summarize_group(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = df.groupby(["kind", "event_type", "rel_slot"], sort=True)[value_col]
    out = grouped.agg(
        n="count",
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    return out


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    months = [int(part) for part in args.months.split(",") if part.strip()]
    control_weeks = [int(part) for part in args.control_weeks.split(",") if part.strip()]

    data = np.load(dataset_dir / "data.npz")["data"]
    flow = data[:, :, 0].astype(np.float64, copy=False)
    event_channel = data[:, :, 3].astype(np.float32, copy=False)
    meta = pd.read_csv(dataset_dir / "sensor_meta_feature.csv")
    matched = pd.read_csv(dataset_dir / "matched_incidents.csv")
    matched["dt"] = pd.to_datetime(matched["dt"], errors="coerce")

    times = build_time_index(args.year, months)
    if len(times) != flow.shape[0]:
        raise ValueError(f"Time index length {len(times)} != data length {flow.shape[0]}")
    start_time = times[0]
    global_to_local = {int(row.global_index): int(idx) for idx, row in meta.iterrows()}

    rel_slots = np.arange(-args.pre_slots, args.post_slots + 1)
    rows: list[dict] = []
    case_rows: list[dict] = []

    for incident in matched.itertuples(index=False):
        if pd.isna(incident.dt):
            continue
        node_idx = global_to_local.get(int(incident.global_index))
        if node_idx is None:
            continue
        slot = int((incident.dt - start_time) / pd.Timedelta(minutes=5))
        left = slot - args.pre_slots
        right = slot + args.post_slots
        if left < 0 or right >= flow.shape[0]:
            continue

        event_window = flow[left : right + 1, node_idx]
        if np.isnan(event_window).any():
            continue

        event_type = str(incident.Type)
        case_rows.append(
            {
                "incident_id": incident.incident_id,
                "event_type": event_type,
                "station_id": incident.station_id,
                "node_idx": node_idx,
                "global_index": int(incident.global_index),
                "slot": slot,
                "dt": incident.dt,
                "n_controls": 0,
            }
        )
        case_idx = len(case_rows) - 1

        for rel, value in zip(rel_slots, event_window):
            rows.append(
                {
                    "kind": "incident",
                    "event_type": "ALL",
                    "incident_type": event_type,
                    "rel_slot": int(rel),
                    "rel_min": int(rel * 5),
                    "flow": float(value),
                    "incident_id": incident.incident_id,
                }
            )
            rows.append(
                {
                    "kind": "incident",
                    "event_type": event_type,
                    "incident_type": event_type,
                    "rel_slot": int(rel),
                    "rel_min": int(rel * 5),
                    "flow": float(value),
                    "incident_id": incident.incident_id,
                }
            )

        controls_added = 0
        for week_offset in control_weeks:
            control_slot = slot + week_offset * SLOTS_PER_WEEK
            c_left = control_slot - args.pre_slots
            c_right = control_slot + args.post_slots
            if c_left < 0 or c_right >= flow.shape[0]:
                continue
            if not window_is_clear(event_channel, node_idx, c_left, c_right):
                continue
            control_window = flow[c_left : c_right + 1, node_idx]
            if np.isnan(control_window).any():
                continue
            controls_added += 1
            for rel, value in zip(rel_slots, control_window):
                rows.append(
                    {
                        "kind": "control",
                        "event_type": "ALL",
                        "incident_type": event_type,
                        "rel_slot": int(rel),
                        "rel_min": int(rel * 5),
                        "flow": float(value),
                        "incident_id": incident.incident_id,
                    }
                )
                rows.append(
                    {
                        "kind": "control",
                        "event_type": event_type,
                        "incident_type": event_type,
                        "rel_slot": int(rel),
                        "rel_min": int(rel * 5),
                        "flow": float(value),
                        "incident_id": incident.incident_id,
                    }
                )
        case_rows[case_idx]["n_controls"] = controls_added

    long_df = pd.DataFrame(rows)
    cases = pd.DataFrame(case_rows)
    if long_df.empty:
        raise RuntimeError("No aligned windows were collected.")

    type_counts = matched["Type"].value_counts().rename_axis("event_type").reset_index(name="matched_incidents")
    used_type_counts = cases["event_type"].value_counts().rename_axis("event_type").reset_index(name="used_incident_windows")
    type_counts = type_counts.merge(used_type_counts, on="event_type", how="left").fillna({"used_incident_windows": 0})
    type_counts["used_incident_windows"] = type_counts["used_incident_windows"].astype(int)

    summary = summarize_group(long_df, "flow")
    summary["rel_min"] = summary["rel_slot"] * 5
    pivot = summary.pivot_table(index=["event_type", "rel_slot"], columns="kind", values="mean").reset_index()
    pivot["delta_incident_minus_control"] = pivot.get("incident", np.nan) - pivot.get("control", np.nan)
    pivot["rel_min"] = pivot["rel_slot"] * 5

    impact_windows = []
    for event_type, group in pivot.groupby("event_type"):
        pre = group[(group["rel_slot"] >= -args.pre_slots) & (group["rel_slot"] <= -1)]
        early = group[(group["rel_slot"] >= 0) & (group["rel_slot"] <= min(6, args.post_slots))]
        late = group[(group["rel_slot"] >= 7) & (group["rel_slot"] <= args.post_slots)]
        impact_windows.append(
            {
                "event_type": event_type,
                "pre_delta_mean": float(pre["delta_incident_minus_control"].mean()),
                "post_0_30_delta_mean": float(early["delta_incident_minus_control"].mean()),
                "post_35_120_delta_mean": float(late["delta_incident_minus_control"].mean()) if not late.empty else np.nan,
            }
        )
    impact_summary = pd.DataFrame(impact_windows)

    long_df.to_csv(output_dir / "aligned_flow_windows_long.csv", index=False)
    cases.to_csv(output_dir / "aligned_incident_cases.csv", index=False)
    type_counts.to_csv(output_dir / "incident_type_counts_d5_official_all.csv", index=False)
    summary.to_csv(output_dir / "aligned_flow_summary.csv", index=False)
    pivot.to_csv(output_dir / "aligned_flow_delta_summary.csv", index=False)
    impact_summary.to_csv(output_dir / "impact_window_summary.csv", index=False)

    # Overall plot.
    all_summary = summary[summary["event_type"] == "ALL"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"incident": "#c43c39", "control": "#2f6f9f"}
    for kind, label in [("incident", "Incident-aligned flow"), ("control", "Matched no-event control")]:
        part = all_summary[all_summary["kind"] == kind].sort_values("rel_slot")
        x = part["rel_min"].to_numpy()
        y = part["mean"].to_numpy()
        lo = part["q25"].to_numpy()
        hi = part["q75"].to_numpy()
        axes[0].plot(x, y, label=label, color=colors[kind], linewidth=2.3)
        axes[0].fill_between(x, lo, hi, color=colors[kind], alpha=0.15, linewidth=0)
    all_delta = pivot[pivot["event_type"] == "ALL"].sort_values("rel_slot")
    axes[1].plot(
        all_delta["rel_min"],
        all_delta["delta_incident_minus_control"],
        color="#4a4a4a",
        linewidth=2.2,
        label="Incident - control",
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Flow")
    axes[1].set_ylabel("Delta flow")
    axes[1].set_xlabel("Minutes relative to matched incident time")
    axes[0].set_title("TraffiDent D5 Q1: event-aligned flow response")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="#e7e7e7", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "d5_incident_aligned_flow_overall.png", dpi=180)
    fig.savefig(output_dir / "d5_incident_aligned_flow_overall.pdf")
    plt.close(fig)

    # Type-specific deltas.
    eligible_types = type_counts[type_counts["used_incident_windows"] >= args.min_type_cases]["event_type"].tolist()
    eligible_types = [t for t in eligible_types if t != "ALL"]
    ncols = 2
    nrows = int(np.ceil(len(eligible_types) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.0 * nrows, 4.0)), sharex=True, sharey=False)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, event_type in zip(axes_arr, eligible_types):
        part = pivot[pivot["event_type"] == event_type].sort_values("rel_slot")
        n_cases = int(type_counts.loc[type_counts["event_type"] == event_type, "used_incident_windows"].iloc[0])
        ax.plot(part["rel_min"], part["delta_incident_minus_control"], color="#7952b3", linewidth=2.0)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1.0)
        ax.set_title(f"{event_type} (n={n_cases})")
        ax.grid(True, color="#ececec", linewidth=0.8)
        ax.set_ylabel("Delta flow")
    for ax in axes_arr[len(eligible_types) :]:
        ax.axis("off")
    axes_arr[-1].set_xlabel("Minutes relative to matched incident time")
    fig.suptitle("Incident minus matched-control flow by event type", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "d5_incident_aligned_flow_delta_by_type.png", dpi=180)
    fig.savefig(output_dir / "d5_incident_aligned_flow_delta_by_type.pdf")
    plt.close(fig)

    manifest = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "pre_slots": args.pre_slots,
        "post_slots": args.post_slots,
        "control_weeks": control_weeks,
        "matched_incidents": int(len(matched)),
        "used_incident_windows": int(len(cases)),
        "used_cases_with_controls": int((cases["n_controls"] > 0).sum()),
        "type_counts": type_counts.to_dict(orient="records"),
        "outputs": [
            "d5_incident_aligned_flow_overall.png",
            "d5_incident_aligned_flow_delta_by_type.png",
            "incident_type_counts_d5_official_all.csv",
            "impact_window_summary.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
