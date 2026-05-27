#!/usr/bin/env python3
"""Retrieval-only pilot for incident response memory on TraffiDent county data.

This script is intentionally training-free. It asks whether historical incident
response residuals contain useful signal beyond a matched normal-traffic prior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


COUNTIES = ("LosAngeles", "Orange", "Alameda", "ContraCosta")


@dataclass
class EventSet:
    county: str
    records: pd.DataFrame
    pre: np.ndarray
    future: np.ndarray
    normal: np.ndarray
    residual: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/data/yuzhang_fei/TraffiDent/basicts",
        help="Root containing TraffiDent_<County>_2023Q1 directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/yuzhang_fei/code/event-traffic-prediction-git/"
            "reproduction/analysis/incident_memory_retrieval_pilot"
        ),
    )
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--flow-channel", type=int, default=0)
    parser.add_argument("--start-time", default="2023-01-01 00:00:00")
    parser.add_argument(
        "--normal-max-candidates",
        type=int,
        default=16,
        help="Maximum same weekday/time candidates used for the normal prior.",
    )
    return parser.parse_args()


def slot_from_dt(series: pd.Series, start_time: str) -> np.ndarray:
    start = pd.Timestamp(start_time)
    dt = pd.to_datetime(series)
    minutes = (dt - start).dt.total_seconds().to_numpy() / 60.0
    return np.floor(minutes / 5.0).astype(int)


def masked_mae(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def horizon_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    out = []
    for h in range(true.shape[1]):
        out.append(masked_mae(pred[:, h], true[:, h]))
    return np.array(out, dtype=float)


def build_incident_mask(events: pd.DataFrame, t_len: int, n_nodes: int, radius: int) -> np.ndarray:
    mask = np.zeros((t_len, n_nodes), dtype=bool)
    for slot, node in zip(events["target_start"].to_numpy(), events["node_idx"].to_numpy()):
        left = max(0, int(slot) - radius)
        right = min(t_len, int(slot) + radius + 1)
        mask[left:right, int(node)] = True
    return mask


def normal_prior_for_event(
    flow: np.ndarray,
    incident_mask: np.ndarray,
    node: int,
    target_start: int,
    train_target_start: int,
    train_target_end: int,
    horizon: int,
    max_candidates: int,
) -> np.ndarray:
    slots_per_day = 288
    slots_per_week = 7 * slots_per_day
    candidates = []

    # Same weekday and same time-of-day, using only training-period targets.
    base_mod = target_start % slots_per_week
    for s in range(train_target_start, train_target_end - horizon + 1):
        if s % slots_per_week != base_mod:
            continue
        if incident_mask[max(0, s - horizon) : min(len(flow), s + horizon + 1), node].any():
            continue
        candidates.append(s)

    # Relax to same time-of-day if the same weekday bucket is empty.
    if not candidates:
        base_tod = target_start % slots_per_day
        for s in range(train_target_start, train_target_end - horizon + 1):
            if s % slots_per_day != base_tod:
                continue
            if incident_mask[max(0, s - horizon) : min(len(flow), s + horizon + 1), node].any():
                continue
            candidates.append(s)

    if not candidates:
        last = flow[max(0, target_start - 1), node]
        return np.repeat(last, horizon).astype(float)

    candidates = candidates[-max_candidates:]
    seqs = np.stack([flow[s : s + horizon, node] for s in candidates], axis=0)
    return np.nanmedian(seqs, axis=0).astype(float)


def load_county(data_root: Path, county: str, args: argparse.Namespace) -> tuple[EventSet, EventSet]:
    data_dir = data_root / f"TraffiDent_{county}_2023Q1"
    data = np.load(data_dir / "data.npz")["data"]
    flow = data[:, :, args.flow_channel].astype(float)
    index = np.load(data_dir / "index.npz")
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv")
    incidents = pd.read_csv(data_dir / "matched_incidents.csv")

    gid_to_node = {int(gid): i for i, gid in enumerate(meta["global_index"].to_numpy())}
    incidents = incidents.copy()
    incidents["node_idx"] = incidents["global_index"].map(gid_to_node)
    incidents = incidents.dropna(subset=["node_idx", "dt", "Type", "Fwy", "incident_abs_pm"])
    incidents["node_idx"] = incidents["node_idx"].astype(int)
    incidents["event_slot"] = slot_from_dt(incidents["dt"], args.start_time)
    # Official post-incident forecasting starts from the next 5-minute slot.
    incidents["target_start"] = incidents["event_slot"] + 1

    train_target_start = int(index["train"][0, 1])
    train_target_end = int(index["train"][-1, 2])
    test_target_start = int(index["test"][0, 1])
    test_target_end = int(index["test"][-1, 2])

    valid = incidents[
        (incidents["target_start"] - args.history >= 0)
        & (incidents["target_start"] + args.horizon <= flow.shape[0])
    ].copy()
    valid = valid.sort_values(["target_start", "node_idx", "incident_id"])
    valid = valid.drop_duplicates(["target_start", "node_idx", "incident_id"])

    incident_mask = build_incident_mask(valid, flow.shape[0], flow.shape[1], args.horizon)

    def build_split(left: int, right: int) -> EventSet:
        rows = valid[
            (valid["target_start"] >= left)
            & (valid["target_start"] + args.horizon <= right)
        ].copy()
        pre, future, normal = [], [], []
        kept = []
        for row in rows.itertuples(index=False):
            node = int(row.node_idx)
            start = int(row.target_start)
            pre_seq = flow[start - args.history : start, node]
            fut_seq = flow[start : start + args.horizon, node]
            if not np.all(np.isfinite(pre_seq)) or not np.all(np.isfinite(fut_seq)):
                continue
            prior = normal_prior_for_event(
                flow,
                incident_mask,
                node,
                start,
                train_target_start,
                train_target_end,
                args.horizon,
                args.normal_max_candidates,
            )
            pre.append(pre_seq)
            future.append(fut_seq)
            normal.append(prior)
            kept.append(row._asdict())
        if not kept:
            empty = np.empty((0, args.horizon), dtype=float)
            return EventSet(county, pd.DataFrame(), empty, empty, empty, empty)
        pre_arr = np.asarray(pre, dtype=float)
        future_arr = np.asarray(future, dtype=float)
        normal_arr = np.asarray(normal, dtype=float)
        recs = pd.DataFrame(kept)
        return EventSet(county, recs, pre_arr, future_arr, normal_arr, future_arr - normal_arr)

    train = build_split(train_target_start, train_target_end)
    test = build_split(test_target_start, test_target_end)
    return train, test


def retrieval_predict(train: EventSet, test: EventSet, topk: int) -> dict[str, np.ndarray]:
    if len(train.records) == 0 or len(test.records) == 0:
        empty = np.full_like(test.future, np.nan)
        return {
            "normal_prior": test.normal,
            "global_residual": empty,
            "type_residual": empty,
            "retrieval_residual": empty,
            "persistence": empty,
        }

    train_pre = train.pre
    test_pre = test.pre
    scale = np.nanstd(train_pre)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0

    train_pm = train.records["incident_abs_pm"].to_numpy(dtype=float)
    test_pm = test.records["incident_abs_pm"].to_numpy(dtype=float)
    pm_scale = np.nanstd(train_pm)
    if not np.isfinite(pm_scale) or pm_scale <= 1e-6:
        pm_scale = 1.0

    train_tod = (train.records["target_start"].to_numpy(dtype=float) % 288) / 288.0
    test_tod = (test.records["target_start"].to_numpy(dtype=float) % 288) / 288.0

    global_resid = np.nanmean(train.residual, axis=0)
    by_type = {
        typ: np.nanmean(train.residual[train.records["Type"].to_numpy() == typ], axis=0)
        for typ in sorted(train.records["Type"].dropna().unique())
    }

    retrieval = []
    type_mean = []
    for i, row in enumerate(test.records.itertuples(index=False)):
        pre_dist = np.mean(np.abs(train_pre - test_pre[i][None, :]), axis=1) / scale
        pm_dist = np.abs(train_pm - test_pm[i]) / pm_scale
        type_dist = (train.records["Type"].to_numpy() != row.Type).astype(float)
        fwy_dist = (train.records["Fwy"].to_numpy() != row.Fwy).astype(float)
        tod_delta = np.abs(train_tod - test_tod[i])
        tod_dist = np.minimum(tod_delta, 1.0 - tod_delta)
        dist = 0.45 * pre_dist + 0.25 * type_dist + 0.15 * fwy_dist + 0.10 * pm_dist + 0.05 * tod_dist
        k = min(topk, len(dist))
        nn = np.argpartition(dist, k - 1)[:k]
        retrieval.append(np.nanmean(train.residual[nn], axis=0))
        type_mean.append(by_type.get(row.Type, global_resid))

    retrieval_arr = np.asarray(retrieval, dtype=float)
    type_arr = np.asarray(type_mean, dtype=float)
    persistence = np.repeat(test.pre[:, -1:], test.future.shape[1], axis=1)

    return {
        "normal_prior": test.normal,
        "global_residual": test.normal + global_resid[None, :],
        "type_residual": test.normal + type_arr,
        "retrieval_residual": test.normal + retrieval_arr,
        "persistence": persistence,
    }


def evaluate_county(train: EventSet, test: EventSet, topk: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = retrieval_predict(train, test, topk)
    rows = []
    horizon_rows = []
    for name, pred in preds.items():
        rows.append(
            {
                "county": test.county,
                "method": name,
                "n_train_events": len(train.records),
                "n_test_events": len(test.records),
                "mae": masked_mae(pred, test.future),
            }
        )
        h_mae = horizon_mae(pred, test.future)
        for h, value in enumerate(h_mae, start=1):
            horizon_rows.append(
                {
                    "county": test.county,
                    "method": name,
                    "horizon": h,
                    "mae": value,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(horizon_rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_horizon = []
    event_counts = []
    for county in args.counties:
        train, test = load_county(data_root, county, args)
        summary, by_horizon = evaluate_county(train, test, args.topk)
        all_summary.append(summary)
        all_horizon.append(by_horizon)
        event_counts.append(
            {
                "county": county,
                "train_events": int(len(train.records)),
                "test_events": int(len(test.records)),
                "train_types": train.records["Type"].value_counts().to_dict()
                if len(train.records)
                else {},
                "test_types": test.records["Type"].value_counts().to_dict()
                if len(test.records)
                else {},
            }
        )

    summary_df = pd.concat(all_summary, ignore_index=True)
    horizon_df = pd.concat(all_horizon, ignore_index=True)
    summary_df.to_csv(out_dir / "retrieval_pilot_summary.csv", index=False)
    horizon_df.to_csv(out_dir / "retrieval_pilot_by_horizon.csv", index=False)
    with (out_dir / "retrieval_pilot_event_counts.json").open("w") as f:
        json.dump(event_counts, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
