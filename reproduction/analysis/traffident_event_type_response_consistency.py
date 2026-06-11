#!/usr/bin/env python3
"""Analyze whether TraffiDent event types induce consistent node responses.

For each official matched incident, this script compares the matched sensor's
raw flow window against same-sensor same-time-of-week no-event controls. It
then summarizes response consistency by incident type and by incident type x
normal-flow cluster.
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


TYPE_ORDER = ["Hazard", "NoInj", "Other", "UnknInj", "1141", "AHazard", "Fire", "CarFire"]


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
        default=Path("reproduction/analysis/traffident_d5_event_type_response_consistency"),
    )
    parser.add_argument("--raw-incident-zip", type=Path, default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"))
    parser.add_argument("--raw-incident-file", default=None)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pre-slots", type=int, default=12)
    parser.add_argument("--post-slots", type=int, default=24)
    parser.add_argument("--min-controls", type=int, default=1)
    parser.add_argument("--post-delta-threshold", type=float, default=-20.0)
    parser.add_argument("--post-minus-pre-threshold", type=float, default=-10.0)
    parser.add_argument("--control-weeks", default="-4,-3,-2,-1,1,2,3,4")
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


def slot_of(dt: pd.Timestamp, start_time: pd.Timestamp) -> int:
    return int((dt - start_time) / pd.Timedelta(minutes=5))


def kmeans_numpy(x: np.ndarray, n_clusters: int, seed: int, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if x.shape[0] < n_clusters:
        raise ValueError(f"n_samples={x.shape[0]} < n_clusters={n_clusters}")
    centers = x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()
    labels = np.full(x.shape[0], -1, dtype=np.int64)
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
    tod = np.arange(flow.shape[0]) % SLOTS_PER_DAY
    profiles = np.empty((flow.shape[1], SLOTS_PER_DAY), dtype=np.float64)
    for slot in range(SLOTS_PER_DAY):
        mask = tod == slot
        values = flow[mask].astype(np.float64)
        normal_values = np.where(event_channel[mask] == 0, values, np.nan)
        slot_mean = np.nanmean(normal_values, axis=0)
        fallback = np.nanmean(values, axis=0)
        profiles[:, slot] = np.where(np.isnan(slot_mean), fallback, slot_mean)
    node_mean = np.nanmean(profiles, axis=1, keepdims=True)
    node_std = np.nanstd(profiles, axis=1, keepdims=True)
    normalized = (profiles - node_mean) / np.maximum(node_std, 1e-6)
    return profiles, np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


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


def ordered_types(types: pd.Series) -> list[str]:
    present = set(types.astype(str).unique())
    ordered = [event_type for event_type in TYPE_ORDER if event_type in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def summarize_by_type(cases: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    node_case = cases.groupby(["event_type", "node_idx"], as_index=False).agg(
        node_post_delta_mean=("post_0_60_delta_mean", "mean"),
        node_post_minus_pre_mean=("post_minus_pre_delta", "mean"),
        n_events=("incident_id", "nunique"),
    )
    for event_type, group in cases.groupby("event_type"):
        node_group = node_case[node_case["event_type"] == event_type]
        rows.append(
            {
                "event_type": event_type,
                "n_events": int(group["incident_id"].nunique()),
                "n_nodes": int(group["node_idx"].nunique()),
                "median_duration_min": float(group["duration_minutes"].dropna().median()),
                "pre_delta_mean": float(group["pre_delta_mean"].mean()),
                "post_0_60_delta_mean": float(group["post_0_60_delta_mean"].mean()),
                "post_65_120_delta_mean": float(group["post_65_120_delta_mean"].mean()),
                "post_minus_pre_delta_mean": float(group["post_minus_pre_delta"].mean()),
                "post_delta_q25": float(group["post_0_60_delta_mean"].quantile(0.25)),
                "post_delta_q75": float(group["post_0_60_delta_mean"].quantile(0.75)),
                "event_frac_post_negative": float((group["post_0_60_delta_mean"] < 0).mean()),
                "event_frac_post_significant": float((group["post_0_60_delta_mean"] <= args.post_delta_threshold).mean()),
                "event_frac_post_minus_pre_negative": float((group["post_minus_pre_delta"] < 0).mean()),
                "event_frac_post_minus_pre_significant": float(
                    (group["post_minus_pre_delta"] <= args.post_minus_pre_threshold).mean()
                ),
                "node_frac_post_negative": float((node_group["node_post_delta_mean"] < 0).mean()),
                "node_frac_post_minus_pre_negative": float((node_group["node_post_minus_pre_mean"] < 0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["event_type"] = pd.Categorical(out["event_type"], categories=ordered_types(out["event_type"]), ordered=True)
    return out.sort_values("event_type").reset_index(drop=True)


def summarize_by_type_cluster(cases: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for (event_type, cluster_id), group in cases.groupby(["event_type", "cluster_id"]):
        rows.append(
            {
                "event_type": event_type,
                "cluster_id": int(cluster_id),
                "n_events": int(group["incident_id"].nunique()),
                "n_nodes": int(group["node_idx"].nunique()),
                "post_0_60_delta_mean": float(group["post_0_60_delta_mean"].mean()),
                "post_minus_pre_delta_mean": float(group["post_minus_pre_delta"].mean()),
                "frac_post_negative": float((group["post_0_60_delta_mean"] < 0).mean()),
                "frac_post_significant": float((group["post_0_60_delta_mean"] <= args.post_delta_threshold).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["event_type"] = pd.Categorical(out["event_type"], categories=ordered_types(out["event_type"]), ordered=True)
    return out.sort_values(["event_type", "cluster_id"]).reset_index(drop=True)


def plot_type_curves(delta_df: pd.DataFrame, cases: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    event_types = ordered_types(delta_df["event_type"])
    n_cols = 2
    n_rows = int(np.ceil(len(event_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, event_type in zip(axes, event_types):
        group = delta_df[delta_df["event_type"] == event_type]
        per_rel = group.groupby("rel_min")["delta"].agg(
            mean="mean",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).reset_index()
        ax.plot(per_rel["rel_min"], per_rel["mean"], color="#c43c39", linewidth=2.0)
        ax.fill_between(
            per_rel["rel_min"].to_numpy(),
            per_rel["q25"].to_numpy(),
            per_rel["q75"].to_numpy(),
            color="#c43c39",
            alpha=0.16,
        )
        median_duration = cases.loc[cases["event_type"] == event_type, "duration_minutes"].replace(0, np.nan).median()
        if np.isfinite(median_duration):
            ax.axvspan(0, min(float(median_duration), args.post_slots * 5), color="#c43c39", alpha=0.10)
        n_events = cases.loc[cases["event_type"] == event_type, "incident_id"].nunique()
        n_nodes = cases.loc[cases["event_type"] == event_type, "node_idx"].nunique()
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1.0)
        ax.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_title(f"{event_type}: {n_events} events, {n_nodes} nodes")
        ax.set_ylabel("Incident - control flow")
        ax.grid(True, color="#e9e9e9")
    for ax in axes[len(event_types) :]:
        ax.axis("off")
    axes[min(len(event_types), len(axes)) - 1].set_xlabel("Minutes relative to incident")
    fig.suptitle("Response curves by official incident type", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "event_type_response_curves.png", dpi=180)
    fig.savefig(output_dir / "event_type_response_curves.pdf")
    plt.close(fig)


def plot_effect_distribution(cases: pd.DataFrame, output_dir: Path) -> None:
    event_types = ordered_types(cases["event_type"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    values_post = [cases.loc[cases["event_type"] == event_type, "post_0_60_delta_mean"].to_numpy() for event_type in event_types]
    values_change = [cases.loc[cases["event_type"] == event_type, "post_minus_pre_delta"].to_numpy() for event_type in event_types]
    axes[0].boxplot(values_post, labels=event_types, showfliers=False)
    axes[0].axhline(0, color="#777777", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Post 0-60 delta")
    axes[0].set_title("Per-event post-window effect by incident type")
    axes[1].boxplot(values_change, labels=event_types, showfliers=False)
    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel("Post minus pre delta")
    axes[1].set_title("Pre-trend corrected effect by incident type")
    for ax in axes:
        ax.grid(True, axis="y", color="#e9e9e9")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "event_type_effect_distribution.png", dpi=180)
    fig.savefig(output_dir / "event_type_effect_distribution.pdf")
    plt.close(fig)


def plot_type_cluster_heatmap(type_cluster: pd.DataFrame, output_dir: Path, value_col: str, filename: str, title: str) -> None:
    event_types = ordered_types(type_cluster["event_type"])
    clusters = sorted(type_cluster["cluster_id"].unique())
    matrix = np.full((len(event_types), len(clusters)), np.nan, dtype=np.float64)
    counts = np.zeros((len(event_types), len(clusters)), dtype=np.int64)
    type_to_i = {event_type: idx for idx, event_type in enumerate(event_types)}
    cluster_to_j = {cluster_id: idx for idx, cluster_id in enumerate(clusters)}
    for row in type_cluster.itertuples(index=False):
        i = type_to_i[str(row.event_type)]
        j = cluster_to_j[int(row.cluster_id)]
        matrix[i, j] = float(getattr(row, value_col))
        counts[i, j] = int(row.n_events)

    vmax = np.nanpercentile(np.abs(matrix), 90)
    vmax = max(float(vmax), 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_xticklabels([f"C{cluster_id}" for cluster_id in clusters])
    ax.set_yticks(np.arange(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.set_xlabel("Normal-flow cluster")
    ax.set_ylabel("Incident type")
    ax.set_title(title)
    for i in range(len(event_types)):
        for j in range(len(clusters)):
            if counts[i, j] > 0:
                ax.text(j, i, f"{matrix[i, j]:.1f}\n(n={counts[i, j]})", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=180)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_node_consistency(type_summary: pd.DataFrame, output_dir: Path) -> None:
    event_types = type_summary["event_type"].astype(str).tolist()
    x = np.arange(len(event_types))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.2))
    ax.bar(x - 0.18, type_summary["event_frac_post_negative"], width=0.36, label="Event-level negative")
    ax.bar(x + 0.18, type_summary["node_frac_post_negative"], width=0.36, label="Node-level negative")
    ax.set_xticks(x)
    ax.set_xticklabels(event_types, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction with post 0-60 delta < 0")
    ax.set_title("Consistency of negative response by incident type")
    ax.grid(True, axis="y", color="#e9e9e9")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "event_type_negative_consistency.png", dpi=180)
    fig.savefig(output_dir / "event_type_negative_consistency.pdf")
    plt.close(fig)


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

    profiles, normalized_profiles = make_daily_profiles(flow, event_channel)
    labels = kmeans_numpy(normalized_profiles, args.n_clusters, args.seed)
    cluster_counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(cluster_counts.index.to_list())}
    labels = np.array([remap[int(label)] for label in labels], dtype=np.int64)

    cluster_assignments = meta.copy()
    cluster_assignments["node_idx"] = np.arange(len(meta))
    cluster_assignments["cluster_id"] = labels
    cluster_assignments.to_csv(args.output_dir / "sensor_flow_pattern_clusters.csv", index=False)

    rel_slots = np.arange(-args.pre_slots, args.post_slots + 1)
    pre_mask = (rel_slots >= -args.pre_slots) & (rel_slots <= -1)
    post_0_60_mask = (rel_slots >= 0) & (rel_slots <= min(12, args.post_slots))
    post_65_120_mask = (rel_slots >= 13) & (rel_slots <= args.post_slots)
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
        pre_delta = float(delta[pre_mask].mean())
        post_0_60_delta = float(delta[post_0_60_mask].mean())
        post_65_120_delta = float(delta[post_65_120_mask].mean()) if post_65_120_mask.any() else np.nan
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
                "pre_delta_mean": pre_delta,
                "post_0_60_delta_mean": post_0_60_delta,
                "post_65_120_delta_mean": post_65_120_delta,
                "post_minus_pre_delta": post_0_60_delta - pre_delta,
                "description": str(getattr(incident, "DESCRIPTION", "")),
                "location": str(getattr(incident, "LOCATION", "")),
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
    if cases.empty or delta_df.empty:
        raise RuntimeError("No event cases with controls were collected.")

    type_summary = summarize_by_type(cases, args)
    type_cluster = summarize_by_type_cluster(cases, args)
    cases.to_csv(args.output_dir / "event_type_response_cases.csv", index=False)
    delta_df.to_csv(args.output_dir / "event_type_response_delta_windows.csv", index=False)
    type_summary.to_csv(args.output_dir / "event_type_response_summary.csv", index=False)
    type_cluster.to_csv(args.output_dir / "event_type_cluster_response_summary.csv", index=False)

    plot_type_curves(delta_df, cases, args.output_dir, args)
    plot_effect_distribution(cases, args.output_dir)
    plot_type_cluster_heatmap(
        type_cluster,
        args.output_dir,
        "post_0_60_delta_mean",
        "event_type_cluster_post_delta_heatmap",
        "Mean post 0-60 min incident-control delta by type and flow cluster",
    )
    plot_type_cluster_heatmap(
        type_cluster,
        args.output_dir,
        "post_minus_pre_delta_mean",
        "event_type_cluster_post_minus_pre_heatmap",
        "Mean post-minus-pre delta by type and flow cluster",
    )
    plot_node_consistency(type_summary, args.output_dir)

    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "matched_incidents": int(len(matched)),
        "events_with_controls": int(cases["incident_id"].nunique()),
        "nodes_with_events": int(cases["node_idx"].nunique()),
        "n_clusters": args.n_clusters,
        "seed": args.seed,
        "post_delta_threshold": args.post_delta_threshold,
        "post_minus_pre_threshold": args.post_minus_pre_threshold,
        "type_summary": type_summary.to_dict(orient="records"),
        "outputs": [
            "event_type_response_cases.csv",
            "event_type_response_delta_windows.csv",
            "event_type_response_summary.csv",
            "event_type_cluster_response_summary.csv",
            "event_type_response_curves.png",
            "event_type_effect_distribution.png",
            "event_type_cluster_post_delta_heatmap.png",
            "event_type_cluster_post_minus_pre_heatmap.png",
            "event_type_negative_consistency.png",
        ],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
