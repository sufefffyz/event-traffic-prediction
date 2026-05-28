#!/usr/bin/env python3
"""Fine-grained event-factor metrics for TraffiDent BasicTS runs.

This script slices saved STID/STIDAccident/STIDGatedAccident test results by:

1. incident type;
2. signed post-mile relation between incident and matched sensor;
3. matched-control traffic-change magnitude.

It is read-only for datasets and checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


COUNTIES = ("LosAngeles", "Orange", "Alameda", "ContraCosta")
MODELS = ("STID", "STIDAccident", "STIDGatedAccident")
SCOPES = ("future_any", "history_any", "post_last_slot", "ongoing")
RELATION_EPS_PM = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument("--output-dir", default="reproduction/analysis/traffident_event_factor_metrics")
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def dataset_dir(data_root: Path, county: str) -> Path:
    return data_root / f"TraffiDent_{county}_2023Q1"


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


def null_mask(target: np.ndarray, null_val: float) -> np.ndarray:
    if np.isnan(null_val):
        return np.isfinite(target)
    return np.isfinite(target) & (np.abs(target - null_val) > 5e-5)


def make_prefix(mask: np.ndarray) -> np.ndarray:
    prefix = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=np.int32)
    prefix[1:] = np.cumsum(mask.astype(np.int32), axis=0, dtype=np.int32)
    return prefix


def slot_from_dt(series: pd.Series, start_time: str = "2023-01-01 00:00:00") -> np.ndarray:
    start = pd.Timestamp(start_time)
    dt = pd.to_datetime(series)
    minutes = (dt - start).dt.total_seconds().to_numpy() / 60.0
    return np.floor(minutes / 5.0).astype(int)


def direction_sign(direction: object) -> int:
    text = str(direction).strip().upper()
    if text in {"N", "E"}:
        return 1
    if text in {"S", "W"}:
        return -1
    return 1


def relation_bucket(signed_downstream_pm: float) -> str:
    if signed_downstream_pm > RELATION_EPS_PM:
        return "downstream"
    if signed_downstream_pm < -RELATION_EPS_PM:
        return "upstream"
    return "at_source"


def load_incidents_with_local_meta(data_dir: Path) -> pd.DataFrame:
    incidents = pd.read_csv(data_dir / "matched_incidents.csv")
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv")
    local = meta[["global_index", "Direction", "Abs PM", "Fwy"]].copy()
    local["node_idx"] = np.arange(len(local), dtype=np.int64)
    incidents = incidents.merge(local, on=["global_index", "Fwy"], how="left", suffixes=("", "_meta"))
    incidents = incidents.dropna(subset=["node_idx", "dt", "Type", "incident_abs_pm", "sensor_abs_pm"]).copy()
    incidents["node_idx"] = incidents["node_idx"].astype(int)
    incidents["event_slot"] = slot_from_dt(incidents["dt"])
    gap = incidents["sensor_abs_pm"].astype(float) - incidents["incident_abs_pm"].astype(float)
    sign = incidents["Direction"].map(direction_sign).astype(float)
    incidents["signed_downstream_pm"] = gap * sign
    incidents["relation"] = incidents["signed_downstream_pm"].map(relation_bucket)
    incidents["type_label"] = incidents["Type"].astype(str)
    return incidents


def make_attribute_event_masks(
    incidents: pd.DataFrame,
    num_steps: int,
    num_nodes: int,
    attr: str,
    event_window_slots: int = 2,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for value, group in incidents.groupby(attr):
        if pd.isna(value):
            continue
        mask = np.zeros((num_steps, num_nodes), dtype=bool)
        for row in group.itertuples(index=False):
            slot = int(row.event_slot)
            node = int(row.node_idx)
            if slot < 0 or slot >= num_steps or node < 0 or node >= num_nodes:
                continue
            mask[slot : min(num_steps, slot + event_window_slots), node] = True
        if mask.any():
            masks[str(value)] = mask
    return masks


def scope_masks(event_mask: np.ndarray, index: np.ndarray) -> Dict[str, np.ndarray]:
    prefix = make_prefix(event_mask)
    start, mid, end = index[:, 0], index[:, 1], index[:, 2]
    hist_count = prefix[mid] - prefix[start]
    fut_count = prefix[end] - prefix[mid]
    hist_any = hist_count > 0
    fut_any = fut_count > 0
    last_event = event_mask[mid - 1]
    return {
        "future_any": fut_any,
        "history_any": hist_any,
        "post_last_slot": last_event,
        "ongoing": hist_any & fut_any,
    }


def init_acc() -> Dict[str, float]:
    return defaultdict(float, {"node_windows": 0, "valid_values": 0})


def add_metrics(acc: Dict[str, float], mask_2d: np.ndarray, pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
    acc["node_windows"] += int(mask_2d.sum())
    if not mask_2d.any():
        return
    selected = mask_2d[:, None, :] & valid
    count = int(selected.sum())
    acc["valid_values"] += count
    if count == 0:
        return
    err = pred - target
    acc["abs_error_sum"] += float(np.abs(err)[selected].sum())
    acc["sq_error_sum"] += float((err * err)[selected].sum())
    acc["error_sum"] += float(err[selected].sum())
    for horizon in (3, 6, 12):
        if horizon > pred.shape[1]:
            continue
        h_selected = mask_2d & valid[:, horizon - 1, :]
        h_count = int(h_selected.sum())
        acc[f"h{horizon}_valid_values"] += h_count
        if h_count == 0:
            continue
        h_err = err[:, horizon - 1, :]
        acc[f"h{horizon}_abs_error_sum"] += float(np.abs(h_err[h_selected]).sum())


def finalize_acc(acc: Dict[str, float]) -> Dict[str, float]:
    valid = int(acc["valid_values"])
    row = {
        "node_windows": int(acc["node_windows"]),
        "valid_values": valid,
        "MAE": float(acc["abs_error_sum"] / valid) if valid else np.nan,
        "RMSE": float(np.sqrt(acc["sq_error_sum"] / valid)) if valid else np.nan,
        "bias": float(acc["error_sum"] / valid) if valid else np.nan,
    }
    for horizon in (3, 6, 12):
        h_valid = int(acc[f"h{horizon}_valid_values"])
        row[f"MAE@h{horizon}"] = float(acc[f"h{horizon}_abs_error_sum"] / h_valid) if h_valid else np.nan
    return row


def compute_factor_metrics(
    repo_root: Path,
    county: str,
    model: str,
    index: np.ndarray,
    shape: Tuple[int, ...],
    factor_masks: Dict[Tuple[str, str, str], np.ndarray],
    null_val: float,
    chunk_size: int,
) -> List[Dict[str, object]]:
    result_dir = find_result_dir(repo_root, model, county)
    pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
    target = load_raw_or_npy(result_dir / "targets.npy", shape)
    acc = {key: init_acc() for key in factor_masks}

    for offset in range(0, len(index), chunk_size):
        pred_chunk = np.asarray(pred[offset : offset + chunk_size, :, :, 0], dtype=np.float32)
        target_chunk = np.asarray(target[offset : offset + chunk_size, :, :, 0], dtype=np.float32)
        valid = null_mask(target_chunk, null_val)
        for key, mask in factor_masks.items():
            add_metrics(acc[key], mask[offset : offset + len(pred_chunk)], pred_chunk, target_chunk, valid)

    rows = []
    for (factor, value, scope), item in acc.items():
        row = finalize_acc(item)
        row.update(
            {
                "county": county,
                "model": model,
                "factor": factor,
                "value": value,
                "scope": scope,
                "result_dir": str(result_dir.relative_to(repo_root)),
            }
        )
        rows.append(row)
    return rows


def add_deltas(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    metric_cols = ["MAE", "RMSE", "bias", "MAE@h3", "MAE@h6", "MAE@h12"]
    base = df[df["model"] == "STID"][key_cols + metric_cols].rename(
        columns={col: f"STID_{col}" for col in metric_cols}
    )
    out = df.merge(base, on=key_cols, how="left")
    for col in metric_cols:
        out[f"delta_vs_STID_{col}"] = out[col] - out[f"STID_{col}"]
    return out


def traffic_change_metrics(flow: np.ndarray, index: np.ndarray, input_len: int, output_len: int) -> np.ndarray:
    start, mid, end = index[:, 0], index[:, 1], index[:, 2]
    history_mean = np.stack([flow[start + lag] for lag in range(input_len)], axis=1).mean(axis=1)
    future_mean = np.stack([flow[mid + lag] for lag in range(output_len)], axis=1).mean(axis=1)
    return future_mean - history_mean


def timeweek_keys(index: np.ndarray, steps_per_day: int = 288) -> np.ndarray:
    mid = index[:, 1]
    return ((mid // steps_per_day) % 7) * steps_per_day + (mid % steps_per_day)


def matched_impact_masks(
    flow: np.ndarray,
    all_event_mask: np.ndarray,
    index: np.ndarray,
    input_len: int,
    output_len: int,
) -> Dict[Tuple[str, str, str], np.ndarray]:
    scopes = scope_masks(all_event_mask, index)
    change = traffic_change_metrics(flow, index, input_len, output_len)
    keys = timeweek_keys(index)
    num_keys = 7 * 288
    event_scopes = {name: scopes[name] for name in ("future_any", "history_any", "post_last_slot", "ongoing")}

    control_count = np.zeros((flow.shape[1], num_keys), dtype=np.int32)
    control_change_sum = np.zeros((flow.shape[1], num_keys), dtype=np.float64)
    no_event = ~(scopes["future_any"] | scopes["history_any"])
    for row, key in enumerate(keys):
        nodes = np.flatnonzero(no_event[row] & np.isfinite(change[row]))
        if nodes.size == 0:
            continue
        control_count[nodes, key] += 1
        control_change_sum[nodes, key] += change[row, nodes]

    factor_masks: Dict[Tuple[str, str, str], np.ndarray] = {}
    for scope, event_mask in event_scopes.items():
        matched_diff = np.full(event_mask.shape, np.nan, dtype=np.float32)
        for row, key in enumerate(keys):
            nodes = np.flatnonzero(event_mask[row])
            if nodes.size == 0:
                continue
            counts = control_count[nodes, key]
            valid = counts > 0
            if not valid.any():
                continue
            nodes = nodes[valid]
            control_mean = control_change_sum[nodes, key] / control_count[nodes, key]
            matched_diff[row, nodes] = change[row, nodes] - control_mean

        valid_values = np.abs(matched_diff[np.isfinite(matched_diff)])
        if valid_values.size == 0:
            continue
        q50, q75 = np.quantile(valid_values, [0.5, 0.75])
        abs_diff = np.abs(matched_diff)
        factor_masks[("impact_abs", "low", scope)] = np.isfinite(abs_diff) & (abs_diff <= q50)
        factor_masks[("impact_abs", "mid", scope)] = np.isfinite(abs_diff) & (abs_diff > q50) & (abs_diff <= q75)
        factor_masks[("impact_abs", "high", scope)] = np.isfinite(abs_diff) & (abs_diff > q75)
        factor_masks[("impact_direction", "drop", scope)] = np.isfinite(matched_diff) & (matched_diff < 0)
        factor_masks[("impact_direction", "rise", scope)] = np.isfinite(matched_diff) & (matched_diff > 0)
        factor_masks[("impact_threshold", f"q50={q50:.4f};q75={q75:.4f}", scope)] = np.zeros_like(event_mask, dtype=bool)
    return factor_masks


def build_factor_masks(data_dir: Path, data: np.ndarray, index: np.ndarray, args: argparse.Namespace) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], Dict[str, object]]:
    incidents = load_incidents_with_local_meta(data_dir)
    all_event_mask = data[:, :, args.event_channel] > 0
    factor_masks: Dict[Tuple[str, str, str], np.ndarray] = {}

    for factor, attr in (("incident_type", "type_label"), ("pm_relation", "relation")):
        for value, event_mask in make_attribute_event_masks(
            incidents, data.shape[0], data.shape[1], attr
        ).items():
            scoped = scope_masks(event_mask, index)
            for scope in SCOPES:
                factor_masks[(factor, value, scope)] = scoped[scope]

    factor_masks.update(
        matched_impact_masks(
            flow=data[:, :, 0].astype(np.float32, copy=False),
            all_event_mask=all_event_mask,
            index=index,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    )

    meta = {
        "incident_rows": int(len(incidents)),
        "incident_types": sorted(incidents["type_label"].dropna().astype(str).unique().tolist()),
        "relations": sorted(incidents["relation"].dropna().astype(str).unique().tolist()),
        "relation_definition": {
            "signed_downstream_pm": "(sensor_abs_pm - incident_abs_pm) * sign(Direction), sign=+1 for N/E and -1 for S/W",
            "downstream": f"signed_downstream_pm > {RELATION_EPS_PM}",
            "upstream": f"signed_downstream_pm < {-RELATION_EPS_PM}",
            "at_source": f"|signed_downstream_pm| <= {RELATION_EPS_PM}",
        },
    }
    return factor_masks, meta


def compact_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[
        (df["model"] == "STIDGatedAccident")
        & (df["factor"].isin(["incident_type", "pm_relation", "impact_abs", "impact_direction"]))
    ].copy()
    columns = [
        "county",
        "factor",
        "value",
        "scope",
        "node_windows",
        "MAE",
        "delta_vs_STID_MAE",
        "RMSE",
        "bias",
        "MAE@h3",
        "MAE@h6",
        "MAE@h12",
    ]
    return keep[columns].sort_values(["factor", "county", "value", "scope"])


def summarize_gated(df: pd.DataFrame) -> pd.DataFrame:
    gated = df[df["model"] == "STIDGatedAccident"].copy()
    rows = []
    for (factor, value, scope), group in gated.groupby(["factor", "value", "scope"]):
        if str(value).startswith("q50="):
            continue
        deltas = group["delta_vs_STID_MAE"].dropna()
        rows.append(
            {
                "factor": factor,
                "value": value,
                "scope": scope,
                "counties": int(group["county"].nunique()),
                "total_node_windows": int(group["node_windows"].sum()),
                "mean_delta_vs_STID_MAE": float(deltas.mean()) if len(deltas) else np.nan,
                "wins_vs_STID": int((deltas < 0).sum()),
                "mean_MAE": float(group["MAE"].mean()),
                "mean_bias": float(group["bias"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["factor", "scope", "mean_delta_vs_STID_MAE"])


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    metadata = {}
    for county in args.counties:
        data_dir = dataset_dir(data_root, county)
        data = np.load(data_dir / "data.npz")["data"]
        index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
        shape = (len(index), args.output_len, data.shape[1], 1)
        factor_masks, meta = build_factor_masks(data_dir, data, index, args)
        metadata[county] = {
            **meta,
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "factor_mask_count": len(factor_masks),
        }
        for model in args.models:
            rows.extend(
                compute_factor_metrics(
                    repo_root=repo_root,
                    county=county,
                    model=model,
                    index=index,
                    shape=shape,
                    factor_masks=factor_masks,
                    null_val=args.null_val,
                    chunk_size=args.chunk_size,
                )
            )

    key_cols = ["county", "factor", "value", "scope"]
    df = add_deltas(pd.DataFrame(rows), key_cols)
    metrics_path = output_dir / "event_factor_metrics.csv"
    compact_path = output_dir / "event_factor_metrics_compact.csv"
    summary_path = output_dir / "event_factor_gated_summary.csv"
    df.to_csv(metrics_path, index=False)
    compact_rows(df).to_csv(compact_path, index=False)
    summarize_gated(df).to_csv(summary_path, index=False)

    summary = {
        "output_dir": str(output_dir),
        "counties": args.counties,
        "models": args.models,
        "metadata": metadata,
        "files": {
            "metrics_csv": str(metrics_path),
            "compact_csv": str(compact_path),
            "gated_summary_csv": str(summary_path),
        },
    }
    with open(output_dir / "event_factor_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary["files"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
