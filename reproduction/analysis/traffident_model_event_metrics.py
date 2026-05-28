#!/usr/bin/env python3
"""Event-window metrics for TraffiDent BasicTS county runs.

The script is read-only for checkpoints and datasets. It recomputes BasicTS-like
masked MAE/RMSE on saved test results, then slices errors by accident timing in
the history/future windows.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


COUNTIES = ("LosAngeles", "Orange", "Alameda", "ContraCosta")
MODELS = ("STID", "STIDAccident", "STIDGatedAccident")
GROUP_DEFINITIONS = {
    "all": "all test node-windows",
    "no_event": "no accident in history or future window",
    "future_onset": "history has no accident, future has accident",
    "history_only": "history has accident, future has no accident",
    "ongoing": "history has accident and future has accident",
    "post_last_slot": "last history slot has accident",
    "future_any": "future window has accident",
    "history_any": "history window has accident",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument("--output-dir", default="reproduction/analysis/traffident_model_event_metrics")
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--flow-channel", type=int, default=0)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def result_glob(repo_root: Path, model: str, county: str) -> str:
    return str(
        repo_root
        / "BasicTS"
        / "checkpoints"
        / model
        / f"TraffiDent_{county}_2023Q1_100_12_12_*"
        / "*"
        / "test_results"
    )


def find_result_dir(repo_root: Path, model: str, county: str) -> Path:
    matches = [Path(path) for path in glob.glob(result_glob(repo_root, model, county))]
    matches = [path for path in matches if (path / "predictions.npy").exists() and (path / "targets.npy").exists()]
    if not matches:
        raise FileNotFoundError(f"No test_results found for {model} {county}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_raw_or_npy(path: Path, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    try:
        arr = np.load(path, mmap_mode="r")
    except ValueError as exc:
        if "pickled data" not in str(exc):
            raise
        expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path} is not an npy file and has {actual_bytes} bytes; "
                f"expected {expected_bytes} for shape {shape}"
            ) from exc
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)
    if arr.shape != shape:
        raise ValueError(f"{path} shape {arr.shape} does not match expected {shape}")
    return arr


def make_prefix(mask: np.ndarray) -> np.ndarray:
    prefix = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=np.int32)
    prefix[1:] = np.cumsum(mask.astype(np.int32), axis=0, dtype=np.int32)
    return prefix


def group_masks(hist_any: np.ndarray, fut_any: np.ndarray, last_event: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones_like(hist_any, dtype=bool),
        "no_event": (~hist_any) & (~fut_any),
        "future_onset": (~hist_any) & fut_any,
        "history_only": hist_any & (~fut_any),
        "ongoing": hist_any & fut_any,
        "post_last_slot": last_event,
        "future_any": fut_any,
        "history_any": hist_any,
    }


def init_acc() -> Dict[str, Dict[str, float]]:
    acc = {}
    for group in GROUP_DEFINITIONS:
        item = defaultdict(float)
        item["node_windows"] = 0
        item["valid_values"] = 0
        for horizon in (1, 3, 6, 12):
            item[f"h{h}_valid_values"] = 0
        acc[group] = item
    return acc


def add_metrics(
    acc: Dict[str, Dict[str, float]],
    group: str,
    mask_2d: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> None:
    item = acc[group]
    item["node_windows"] += int(mask_2d.sum())
    if not mask_2d.any():
        return

    mask = mask_2d[:, None, :]
    selected = mask & valid
    count = int(selected.sum())
    item["valid_values"] += count
    if count == 0:
        return
    err = pred - target
    abs_err = np.abs(err)
    sq_err = err * err
    item["abs_error_sum"] += float(abs_err[selected].sum())
    item["sq_error_sum"] += float(sq_err[selected].sum())
    item["error_sum"] += float(err[selected].sum())

    for horizon in (1, 3, 6, 12):
        if horizon > pred.shape[1]:
            continue
        h_selected = mask_2d & valid[:, horizon - 1, :]
        h_count = int(h_selected.sum())
        item[f"h{horizon}_valid_values"] += h_count
        if h_count == 0:
            continue
        h_err = err[:, horizon - 1, :]
        item[f"h{horizon}_abs_error_sum"] += float(np.abs(h_err[h_selected]).sum())
        item[f"h{horizon}_sq_error_sum"] += float((h_err[h_selected] ** 2).sum())
        item[f"h{horizon}_error_sum"] += float(h_err[h_selected].sum())


def finalize(acc: Dict[str, Dict[str, float]], total_node_windows: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for group, item in acc.items():
        valid = int(item["valid_values"])
        row: Dict[str, object] = {
            "group": group,
            "group_definition": GROUP_DEFINITIONS[group],
            "node_windows": int(item["node_windows"]),
            "node_window_ratio": float(item["node_windows"] / total_node_windows) if total_node_windows else 0.0,
            "valid_values": valid,
            "MAE": float(item["abs_error_sum"] / valid) if valid else np.nan,
            "RMSE": float(np.sqrt(item["sq_error_sum"] / valid)) if valid else np.nan,
            "bias": float(item["error_sum"] / valid) if valid else np.nan,
        }
        for horizon in (1, 3, 6, 12):
            h_valid = int(item[f"h{horizon}_valid_values"])
            row[f"MAE@h{horizon}"] = (
                float(item[f"h{horizon}_abs_error_sum"] / h_valid) if h_valid else np.nan
            )
            row[f"RMSE@h{horizon}"] = (
                float(np.sqrt(item[f"h{horizon}_sq_error_sum"] / h_valid)) if h_valid else np.nan
            )
            row[f"bias@h{horizon}"] = (
                float(item[f"h{horizon}_error_sum"] / h_valid) if h_valid else np.nan
            )
            row[f"valid@h{horizon}"] = h_valid
        rows.append(row)
    return rows


def null_mask(target: np.ndarray, null_val: float) -> np.ndarray:
    if np.isnan(null_val):
        return np.isfinite(target)
    return np.isfinite(target) & (np.abs(target - null_val) > 5e-5)


def analyze_county_model(
    data_root: Path,
    repo_root: Path,
    county: str,
    model: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    data_dir = data_root / f"TraffiDent_{county}_2023Q1"
    data = np.load(data_dir / "data.npz")["data"]
    event = data[:, :, args.event_channel] > 0
    index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
    num_samples, num_nodes = len(index), data.shape[1]
    shape = (num_samples, args.output_len, num_nodes, 1)
    result_dir = find_result_dir(repo_root, model, county)
    pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
    target = load_raw_or_npy(result_dir / "targets.npy", shape)

    event_prefix = make_prefix(event)
    acc = init_acc()
    for offset in range(0, num_samples, args.chunk_size):
        chunk = index[offset : offset + args.chunk_size]
        start, mid, end = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        hist_count = event_prefix[mid] - event_prefix[start]
        fut_count = event_prefix[end] - event_prefix[mid]
        masks = group_masks(hist_count > 0, fut_count > 0, event[mid - 1])

        pred_chunk = np.asarray(pred[offset : offset + len(chunk), :, :, 0], dtype=np.float32)
        target_chunk = np.asarray(target[offset : offset + len(chunk), :, :, 0], dtype=np.float32)
        valid = null_mask(target_chunk, args.null_val)
        for group, mask in masks.items():
            add_metrics(acc, group, mask, pred_chunk, target_chunk, valid)

    total_node_windows = int(num_samples * num_nodes)
    rows = finalize(acc, total_node_windows)
    for row in rows:
        row.update(
            {
                "county": county,
                "model": model,
                "result_dir": str(result_dir.relative_to(repo_root)),
            }
        )
    meta = {
        "county": county,
        "model": model,
        "result_dir": str(result_dir.relative_to(repo_root)),
        "data_shape": list(data.shape),
        "test_index_shape": list(index.shape),
        "prediction_shape": list(shape),
    }
    return rows, meta


def add_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["county", "group"]
    metric_cols = ["MAE", "RMSE", "bias", "MAE@h1", "MAE@h3", "MAE@h6", "MAE@h12"]
    base = rows[rows["model"] == "STID"][key_cols + metric_cols].rename(
        columns={col: f"STID_{col}" for col in metric_cols}
    )
    out = rows.merge(base, on=key_cols, how="left")
    for col in metric_cols:
        out[f"delta_vs_STID_{col}"] = out[col] - out[f"STID_{col}"]
    return out


def compact_group_table(rows: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    subset = rows[rows["group"].isin(groups)].copy()
    columns = [
        "county",
        "group",
        "model",
        "node_windows",
        "node_window_ratio",
        "MAE",
        "delta_vs_STID_MAE",
        "RMSE",
        "bias",
        "MAE@h3",
        "MAE@h6",
        "MAE@h12",
    ]
    return subset[columns].sort_values(["county", "group", "model"])


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    metadata: List[Dict[str, object]] = []
    for county in args.counties:
        for model in args.models:
            model_rows, model_meta = analyze_county_model(data_root, repo_root, county, model, args)
            rows.extend(model_rows)
            metadata.append(model_meta)

    df = add_deltas(pd.DataFrame(rows))
    df.to_csv(output_dir / "traffident_model_event_metrics.csv", index=False)
    compact = compact_group_table(df, ["future_onset", "post_last_slot", "ongoing", "history_only"])
    compact.to_csv(output_dir / "traffident_model_event_metrics_compact.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "counties": args.counties,
        "models": args.models,
        "groups": GROUP_DEFINITIONS,
        "null_val": args.null_val,
        "metadata": metadata,
        "files": {
            "metrics_csv": str(output_dir / "traffident_model_event_metrics.csv"),
            "compact_csv": str(output_dir / "traffident_model_event_metrics_compact.csv"),
        },
    }
    with open(output_dir / "traffident_model_event_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary["files"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
