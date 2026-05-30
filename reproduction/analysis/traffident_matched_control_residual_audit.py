#!/usr/bin/env python3
"""Matched-control residual and risk audit for TraffiDent STID results.

This script does not train a model. It asks whether event windows differ from
same-node, same-weekday, same-time-of-day no-event controls in traffic change,
STID residual bias, MAE, and high-error risk. The goal is to decide whether the
next target should be mean residual correction, matched-control incident effect,
or risk/uncertainty prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reproduction.analysis.traffident_decay_kernel_pilot import (  # noqa: E402
    COUNTIES,
    event_masks,
    find_stid_result_dir,
    load_raw_or_npy,
    null_mask,
)


GROUPS = (
    "future_any",
    "future_onset",
    "history_any",
    "history_only",
    "ongoing",
    "post_last_slot",
)

METRICS = (
    "traffic_future_minus_history",
    "traffic_h1_minus_last",
    "traffic_future_min_minus_last",
    "traffic_max_drop_from_last",
    "stid_residual_mean",
    "stid_abs_error_mean",
    "stid_sq_error_mean",
    "stid_tail90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument(
        "--output-dir",
        default="reproduction/analysis/traffident_matched_control_residual_audit",
    )
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--start-weekday", type=int, default=6, help="2023-01-01 is Sunday=6")
    return parser.parse_args()


def make_prefix(x: np.ndarray) -> np.ndarray:
    prefix = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=np.float64)
    prefix[1:] = np.cumsum(x.astype(np.float64), axis=0, dtype=np.float64)
    return prefix


def time_keys(mid_slots: np.ndarray, steps_per_day: int, start_weekday: int) -> np.ndarray:
    day_idx = mid_slots // steps_per_day
    weekday = (start_weekday + day_idx) % 7
    tod = mid_slots % steps_per_day
    return weekday * steps_per_day + tod


def eval_scalar_metrics(
    data: np.ndarray,
    index: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    eval_rows: np.ndarray,
    null_val: float,
) -> Dict[str, np.ndarray]:
    flow = data[:, :, 0].astype(np.float32, copy=False)
    speed_prefix = make_prefix(flow)
    start = index[eval_rows, 0].astype(np.int64)
    mid = index[eval_rows, 1].astype(np.int64)
    end = index[eval_rows, 2].astype(np.int64)
    history_mean = (speed_prefix[mid] - speed_prefix[start]) / (mid - start)[:, None]
    future_mean = np.asarray(target[eval_rows, :, :, 0], dtype=np.float32).mean(axis=1)
    last_history = flow[mid - 1]
    h1 = np.asarray(target[eval_rows, 0, :, 0], dtype=np.float32)
    future_min = np.asarray(target[eval_rows, :, :, 0], dtype=np.float32).min(axis=1)
    pred_eval = np.asarray(pred[eval_rows, :, :, 0], dtype=np.float32)
    target_eval = np.asarray(target[eval_rows, :, :, 0], dtype=np.float32)
    valid = null_mask(target_eval, null_val)
    residual = target_eval - pred_eval
    abs_error = np.abs(pred_eval - target_eval)
    sq_error = (pred_eval - target_eval) ** 2
    valid_count = np.maximum(valid.sum(axis=1), 1)
    residual_mean = (residual * valid).sum(axis=1) / valid_count
    abs_error_mean = (abs_error * valid).sum(axis=1) / valid_count
    sq_error_mean = (sq_error * valid).sum(axis=1) / valid_count
    return {
        "traffic_future_minus_history": future_mean - history_mean,
        "traffic_h1_minus_last": h1 - last_history,
        "traffic_future_min_minus_last": future_min - last_history,
        "traffic_max_drop_from_last": last_history - future_min,
        "stid_residual_mean": residual_mean,
        "stid_abs_error_mean": abs_error_mean,
        "stid_sq_error_mean": sq_error_mean,
    }


def init_acc() -> Dict[str, float]:
    acc: Dict[str, float] = defaultdict(float)
    acc["event_count"] = 0
    acc["matched_count"] = 0
    acc["control_candidate_sum"] = 0
    return acc


def add_control_values(
    control_count: np.ndarray,
    control_sum: Dict[str, np.ndarray],
    metrics: Dict[str, np.ndarray],
    no_event: np.ndarray,
    keys: np.ndarray,
) -> None:
    for row_idx, key in enumerate(keys):
        nodes = np.flatnonzero(no_event[row_idx])
        if nodes.size == 0:
            continue
        control_count[nodes, key] += 1
        for name, arr in metrics.items():
            control_sum[name][nodes, key] += arr[row_idx, nodes]


def add_group_values(
    acc: Dict[str, float],
    mask: np.ndarray,
    metrics: Dict[str, np.ndarray],
    control_count: np.ndarray,
    control_sum: Dict[str, np.ndarray],
    keys: np.ndarray,
) -> None:
    count = int(mask.sum())
    acc["event_count"] += count
    if count == 0:
        return
    rows, nodes = np.where(mask)
    matched = control_count[nodes, keys[rows]]
    valid = matched > 0
    acc["matched_count"] += int(valid.sum())
    if not valid.any():
        return
    rows = rows[valid]
    nodes = nodes[valid]
    matched = matched[valid].astype(np.float64)
    key_values = keys[rows]
    acc["control_candidate_sum"] += int(matched.sum())
    for name, arr in metrics.items():
        event_values = arr[rows, nodes]
        control_values = control_sum[name][nodes, key_values] / matched
        diff = event_values - control_values
        acc[f"event_{name}_sum"] += float(event_values.sum())
        acc[f"control_{name}_sum"] += float(control_values.sum())
        acc[f"diff_{name}_sum"] += float(diff.sum())
        acc[f"diff_{name}_sq_sum"] += float((diff * diff).sum())
        acc[f"diff_{name}_pos_count"] += int((diff > 0).sum())
        acc[f"diff_{name}_neg_count"] += int((diff < 0).sum())


def finalize(acc: Dict[str, float], county: str, group: str) -> Dict[str, object]:
    matched = int(acc["matched_count"])
    event_count = int(acc["event_count"])
    row: Dict[str, object] = {
        "county": county,
        "group": group,
        "event_count": event_count,
        "matched_count": matched,
        "match_rate": float(matched / event_count) if event_count else np.nan,
        "avg_control_candidates": float(acc["control_candidate_sum"] / matched) if matched else np.nan,
    }
    for name in METRICS:
        if matched:
            diff_mean = acc[f"diff_{name}_sum"] / matched
            diff_var = max(acc[f"diff_{name}_sq_sum"] / matched - diff_mean * diff_mean, 0.0)
            row[f"event_{name}"] = float(acc[f"event_{name}_sum"] / matched)
            row[f"control_{name}"] = float(acc[f"control_{name}_sum"] / matched)
            row[f"diff_{name}"] = float(diff_mean)
            row[f"diff_{name}_std"] = float(np.sqrt(diff_var))
            row[f"diff_{name}_pos_ratio"] = float(acc[f"diff_{name}_pos_count"] / matched)
            row[f"diff_{name}_neg_ratio"] = float(acc[f"diff_{name}_neg_count"] / matched)
        else:
            row[f"event_{name}"] = np.nan
            row[f"control_{name}"] = np.nan
            row[f"diff_{name}"] = np.nan
            row[f"diff_{name}_std"] = np.nan
            row[f"diff_{name}_pos_ratio"] = np.nan
            row[f"diff_{name}_neg_ratio"] = np.nan
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for group, group_df in rows.groupby("group"):
        item = {
            "group": group,
            "counties": int(group_df["county"].nunique()),
            "matched_count": int(group_df["matched_count"].sum()),
            "mean_diff_traffic_future_minus_history": float(group_df["diff_traffic_future_minus_history"].mean()),
            "mean_diff_stid_residual_mean": float(group_df["diff_stid_residual_mean"].mean()),
            "mean_diff_stid_abs_error_mean": float(group_df["diff_stid_abs_error_mean"].mean()),
            "mean_diff_stid_tail90": float(group_df["diff_stid_tail90"].mean()),
            "tail90_positive_counties": int((group_df["diff_stid_tail90"] > 0).sum()),
            "abs_error_positive_counties": int((group_df["diff_stid_abs_error_mean"] > 0).sum()),
        }
        summary_rows.append(item)
    return pd.DataFrame(summary_rows).sort_values("group")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {"args": vars(args), "counties": {}}
    for county in args.counties:
        print(f"[audit] {county}", flush=True)
        data_dir = data_root / f"TraffiDent_{county}_2023Q1"
        data = np.load(data_dir / "data.npz")["data"]
        index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
        shape = (len(index), args.output_len, data.shape[1], 1)
        result_dir = find_stid_result_dir(repo_root, county)
        pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
        target = load_raw_or_npy(result_dir / "targets.npy", shape)
        eval_rows = np.arange(len(index) // 2, len(index), dtype=np.int64)
        event = data[:, :, args.event_channel] > 0
        masks_full = event_masks(event, index)
        masks = {name: masks_full[name][eval_rows] for name in GROUPS}
        no_event = masks_full["no_event"][eval_rows]
        keys = time_keys(index[eval_rows, 1], args.steps_per_day, args.start_weekday)
        num_keys = 7 * args.steps_per_day
        metrics = eval_scalar_metrics(data, index, pred, target, eval_rows, args.null_val)
        tail_threshold = float(np.quantile(metrics["stid_abs_error_mean"][no_event], 0.90)) if no_event.any() else np.nan
        metrics["stid_tail90"] = (metrics["stid_abs_error_mean"] > tail_threshold).astype(np.float32)
        control_count = np.zeros((data.shape[1], num_keys), dtype=np.int32)
        control_sum = {
            name: np.zeros((data.shape[1], num_keys), dtype=np.float64)
            for name in METRICS
        }
        add_control_values(control_count, control_sum, metrics, no_event, keys)
        for group, mask in masks.items():
            acc = init_acc()
            add_group_values(acc, mask, metrics, control_count, control_sum, keys)
            rows.append(finalize(acc, county, group))
        metadata["counties"][county] = {
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "eval_rows": int(len(eval_rows)),
            "tail90_threshold": tail_threshold,
            "result_dir": str(result_dir.relative_to(repo_root)),
        }

    rows_df = pd.DataFrame(rows)
    summary_df = summarize(rows_df)
    rows_path = output_dir / "matched_control_residual_audit.csv"
    summary_path = output_dir / "matched_control_residual_summary.csv"
    metadata_path = output_dir / "matched_control_residual_metadata.json"
    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metadata["files"] = {
        "audit_csv": str(rows_path),
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(json.dumps(metadata["files"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
