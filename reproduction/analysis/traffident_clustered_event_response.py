#!/usr/bin/env python3
"""Cluster sensors by normal flow pattern and visualize event response.

Sensors are clustered using their no-event average 5-minute time-of-day flow
profiles. Matched incidents are then aligned at t=0 and compared with
same-sensor same-time-of-week no-event controls. This is a descriptive
consistency diagnostic rather than a causal estimator.
"""

from __future__ import annotations

import argparse
import json
import zipfile
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
        default=Path("reproduction/analysis/traffident_d5_clustered_event_response"),
    )
    parser.add_argument("--raw-incident-zip", type=Path, default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"))
    parser.add_argument("--raw-incident-file", default=None)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pre-slots", type=int, default=12, help="12 slots = 60 minutes before incident.")
    parser.add_argument("--post-slots", type=int, default=24, help="24 slots = 120 minutes after incident.")
    parser.add_argument(
        "--control-weeks",
        default="-4,-3,-2,-1,1,2,3,4",
        help="Same-time-of-week offsets used as no-event controls.",
    )
    parser.add_argument("--min-controls", type=int, default=1)
    parser.add_argument(
        "--significant-delta-threshold",
        type=float,
        default=-20.0,
        help="Keep events whose post-0-60-min incident-control mean is at most this value.",
    )
    parser.add_argument("--min-cluster-cases", type=int, default=5)
    return parser.parse_args()


def build_time_index(year: int, months: list[int]) -> pd.DatetimeIndex:
    pieces = []
    for month in months:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        pieces.append(pd.date_range(start, end, freq="5min", inclusive="left"))
    return pieces[0].append(pieces[1:]) if len(pieces) > 1 else pieces[0]


def load_raw_incident_metadata(zip_path: Path, member_name: str) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame(columns=["incident_id_clean", "duration_minutes"])
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            raw = pd.read_csv(f, sep="\t", dtype={"incident_id": str}, low_memory=False)
    raw["incident_id_clean"] = raw["incident_id"].astype(str).str.strip()
    raw["duration_minutes"] = pd.to_numeric(raw.get("duration"), errors="coerce")
    return raw[["incident_id_clean", "duration_minutes"]].drop_duplicates("incident_id_clean")


def kmeans_numpy(x: np.ndarray, n_clusters: int, seed: int, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if x.shape[0] < n_clusters:
        raise ValueError(f"n_samples={x.shape[0]} < n_clusters={n_clusters}")
    centers = x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, x.shape[0])]
    return labels


def make_daily_profiles(flow: np.ndarray, event_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(flow.shape[0]) % SLOTS_PER_DAY
    profiles = np.empty((flow.shape[1], SLOTS_PER_DAY), dtype=np.float64)
    for slot in range(SLOTS_PER_DAY):
        mask = t == slot
        values = flow[mask].astype(np.float64)
        normal_values = np.where(event_channel[mask] == 0, values, np.nan)
        slot_mean = np.nanmean(normal_values, axis=0)
        fallback = np.nanmean(values, axis=0)
        profiles[:, slot] = np.where(np.isnan(slot_mean), fallback, slot_mean)
    node_mean = np.nanmean(profiles, axis=1, keepdims=True)
    node_std = np.nanstd(profiles, axis=1, keepdims=True)
    normalized = (profiles - node_mean) / np.maximum(node_std, 1e-6)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return profiles, normalized


def slot_of(dt: pd.Timestamp, start_time: pd.Timestamp) -> int:
    return int((dt - start_time) / pd.Timedelta(minutes=5))


def collect_controls(
    flow: np.ndarray,
    event_channel: np.ndarray,
    node_idx: int,
    slot: int,
    pre_slots: int,
    post_slots: int,
    control_weeks: list[int],
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    for week_offset in control_weeks:
        control_slot = slot + week_offset * SLOTS_PER_WEEK
        left = control_slot - pre_slots
        right = control_slot + post_slots
        if left < 0 or right >= flow.shape[0]:
            continue
        if np.nanmax(event_channel[left : right + 1, node_idx]) != 0:
            continue
        window = flow[left : right + 1, node_idx]
        if np.isnan(window).any():
            continue
        controls.append(window)
    return controls


def summarize_delta(delta_df: pd.DataFrame, pre_slots: int, post_slots: int) -> pd.DataFrame:
    rows = []
    for cluster_id, group in delta_df.groupby("cluster_id"):
        pre = group[(group["rel_slot"] >= -pre_slots) & (group["rel_slot"] <= -1)]
        early = group[(group["rel_slot"] >= 0) & (group["rel_slot"] <= min(12, post_slots))]
        late = group[(group["rel_slot"] >= 13) & (group["rel_slot"] <= post_slots)]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_events": int(group["incident_id"].nunique()),
                "n_nodes": int(group["node_idx"].nunique()),
                "median_duration_min": float(group["duration_minutes"].dropna().median()),
                "pre_delta_mean": float(pre["delta"].mean()),
                "post_0_60_delta_mean": float(early["delta"].mean()),
                "post_65_120_delta_mean": float(late["delta"].mean()) if not late.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_events", "cluster_id"], ascending=[False, True])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    months = [int(part) for part in args.months.split(",") if part.strip()]
    control_weeks = [int(part) for part in args.control_weeks.split(",") if part.strip()]
    times = build_time_index(args.year, months)

    data = np.load(args.dataset_dir / "data.npz")["data"]
    flow = data[:, :, 0].astype(np.float64, copy=False)
    event_channel = data[:, :, 3].astype(np.float32, copy=False)
    if len(times) != flow.shape[0]:
        raise ValueError(f"Time index length {len(times)} != data length {flow.shape[0]}")

    meta = pd.read_csv(args.dataset_dir / "sensor_meta_feature.csv")
    matched = pd.read_csv(args.dataset_dir / "matched_incidents.csv")
    matched["dt"] = pd.to_datetime(matched["dt"], errors="coerce")
    matched["incident_id_clean"] = matched["incident_id"].astype(str).str.strip()
    raw_member = args.raw_incident_file or f"incidents_y{args.year}.csv"
    raw_incidents = load_raw_incident_metadata(args.raw_incident_zip, raw_member)
    matched = matched.merge(raw_incidents, on="incident_id_clean", how="left").dropna(subset=["dt"])
    matched["duration_minutes"] = matched["duration_minutes"].clip(lower=0, upper=24 * 60)

    daily_profiles, normalized_profiles = make_daily_profiles(flow, event_channel)
    labels = kmeans_numpy(normalized_profiles, args.n_clusters, args.seed)
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    order = cluster_counts.sort_values(ascending=False).index.to_list()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(label)] for label in labels], dtype=np.int64)

    cluster_assignments = meta.copy()
    cluster_assignments["node_idx"] = np.arange(len(meta))
    cluster_assignments["cluster_id"] = labels
    cluster_assignments.to_csv(args.output_dir / "sensor_flow_pattern_clusters.csv", index=False)

    rel_slots = np.arange(-args.pre_slots, args.post_slots + 1)
    global_to_local = {int(row.global_index): int(idx) for idx, row in meta.iterrows()}
    case_rows = []
    delta_rows = []

    for incident in matched.itertuples(index=False):
        node_idx = global_to_local.get(int(incident.global_index))
        if node_idx is None:
            continue
        slot = slot_of(pd.Timestamp(incident.dt), times[0])
        left = slot - args.pre_slots
        right = slot + args.post_slots
        if left < 0 or right >= flow.shape[0]:
            continue
        incident_window = flow[left : right + 1, node_idx]
        if np.isnan(incident_window).any():
            continue
        controls = collect_controls(flow, event_channel, node_idx, slot, args.pre_slots, args.post_slots, control_weeks)
        if len(controls) < args.min_controls:
            continue
        control_mean = np.nanmean(np.vstack(controls), axis=0)
        delta = incident_window - control_mean
        early_mask = (rel_slots >= 0) & (rel_slots <= min(12, args.post_slots))
        post_delta = float(delta[early_mask].mean())
        if post_delta > args.significant_delta_threshold:
            continue
        duration = getattr(incident, "duration_minutes", np.nan)
        if pd.isna(duration):
            duration = 0.0
        cluster_id = int(labels[node_idx])
        case_rows.append(
            {
                "incident_id": int(incident.incident_id),
                "event_type": str(incident.Type),
                "station_id": int(incident.station_id),
                "node_idx": int(node_idx),
                "global_index": int(incident.global_index),
                "cluster_id": cluster_id,
                "dt": pd.Timestamp(incident.dt).isoformat(),
                "duration_minutes": float(duration),
                "distance": float(incident.distance),
                "n_controls": int(len(controls)),
                "post_0_60_delta_mean": post_delta,
            }
        )
        for rel, inc_value, ctrl_value, delta_value in zip(rel_slots, incident_window, control_mean, delta):
            delta_rows.append(
                {
                    "incident_id": int(incident.incident_id),
                    "event_type": str(incident.Type),
                    "node_idx": int(node_idx),
                    "cluster_id": cluster_id,
                    "rel_slot": int(rel),
                    "rel_min": int(rel * 5),
                    "incident_flow": float(inc_value),
                    "control_flow": float(ctrl_value),
                    "delta": float(delta_value),
                    "duration_minutes": float(duration),
                }
            )

    cases = pd.DataFrame(case_rows)
    delta_df = pd.DataFrame(delta_rows)
    cases.to_csv(args.output_dir / "significant_event_cases.csv", index=False)
    delta_df.to_csv(args.output_dir / "significant_event_delta_windows.csv", index=False)
    if cases.empty or delta_df.empty:
        raise RuntimeError("No significant event cases collected; relax --significant-delta-threshold.")

    cluster_summary = summarize_delta(delta_df, args.pre_slots, args.post_slots)
    cluster_summary = cluster_summary.merge(
        cases.groupby("cluster_id")["event_type"].agg(lambda x: ",".join(x.value_counts().head(3).index)),
        on="cluster_id",
        how="left",
    ).rename(columns={"event_type": "top_event_types"})
    cluster_summary.to_csv(args.output_dir / "cluster_event_response_summary.csv", index=False)

    profile_rows = []
    for cluster_id in range(args.n_clusters):
        members = labels == cluster_id
        if not members.any():
            continue
        profile = daily_profiles[members].mean(axis=0)
        for slot, value in enumerate(profile):
            profile_rows.append({"cluster_id": cluster_id, "tod_min": slot * 5, "flow": float(value)})
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(args.output_dir / "cluster_daily_profiles.csv", index=False)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
    for cluster_id, group in profile_df.groupby("cluster_id"):
        n_nodes = int((labels == cluster_id).sum())
        ax.plot(group["tod_min"] / 60, group["flow"], linewidth=1.8, label=f"C{cluster_id} ({n_nodes} nodes)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean no-event flow")
    ax.set_title("Sensor clusters by normal daily flow pattern")
    ax.grid(True, color="#e6e6e6")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cluster_daily_profiles.png", dpi=180)
    fig.savefig(args.output_dir / "cluster_daily_profiles.pdf")
    plt.close(fig)

    plot_clusters = cluster_summary[cluster_summary["n_events"] >= args.min_cluster_cases]["cluster_id"].astype(int).tolist()
    if not plot_clusters:
        plot_clusters = cluster_summary.head(min(args.n_clusters, 6))["cluster_id"].astype(int).tolist()
    n_cols = 2
    n_rows = int(np.ceil(len(plot_clusters) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 3.1 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cluster_id in zip(axes, plot_clusters):
        group = delta_df[delta_df["cluster_id"] == cluster_id]
        per_rel = group.groupby("rel_min")["delta"].agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
        ax.plot(per_rel["rel_min"], per_rel["mean"], color="#c43c39", linewidth=2.0)
        ax.fill_between(per_rel["rel_min"].to_numpy(), per_rel["q25"].to_numpy(), per_rel["q75"].to_numpy(), color="#c43c39", alpha=0.16)
        median_duration = float(group["duration_minutes"].replace(0, np.nan).median())
        if np.isfinite(median_duration):
            ax.axvspan(0, min(median_duration, args.post_slots * 5), color="#c43c39", alpha=0.10)
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
        row = cluster_summary[cluster_summary["cluster_id"] == cluster_id].iloc[0]
        ax.set_title(
            f"C{cluster_id}: {int(row['n_events'])} events, {int(row['n_nodes'])} nodes, "
            f"post={row['post_0_60_delta_mean']:.1f}"
        )
        ax.set_ylabel("Incident - control flow")
        ax.grid(True, color="#e9e9e9")
    for ax in axes[len(plot_clusters) :]:
        ax.axis("off")
    axes[min(len(plot_clusters), len(axes)) - 1].set_xlabel("Minutes relative to incident")
    fig.suptitle(
        f"Significant event response by normal-flow cluster "
        f"(post 0-60 min delta <= {args.significant_delta_threshold:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output_dir / "significant_event_response_by_cluster.png", dpi=180)
    fig.savefig(args.output_dir / "significant_event_response_by_cluster.pdf")
    plt.close(fig)

    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "n_clusters": args.n_clusters,
        "seed": args.seed,
        "significant_delta_threshold": args.significant_delta_threshold,
        "min_controls": args.min_controls,
        "matched_incidents": int(len(matched)),
        "significant_events": int(len(cases)),
        "significant_nodes": int(cases["node_idx"].nunique()),
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "outputs": [
            "sensor_flow_pattern_clusters.csv",
            "significant_event_cases.csv",
            "significant_event_delta_windows.csv",
            "cluster_event_response_summary.csv",
            "cluster_daily_profiles.csv",
            "cluster_daily_profiles.png",
            "significant_event_response_by_cluster.png",
        ],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
