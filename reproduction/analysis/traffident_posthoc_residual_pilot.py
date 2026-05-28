#!/usr/bin/env python3
"""Post-hoc residual pilot over saved STID test predictions.

This is an exploratory calibration/holdout pilot, not a final test result.
It uses the first half of each saved STID test_result as calibration data and
evaluates on the second half. The goal is to cheaply test whether V1-style
event features can correct pure STID predictions on observed-event windows.

No future target or matched-control impact is used as an inference feature.
Matched-control impact is used only for calibration weights and evaluation
groups.
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
TYPE_VALUES = ("1141", "NoInj", "UnknInj", "other")
RELATION_VALUES = ("downstream", "upstream", "at_source")
RELATION_EPS_PM = 0.025
HORIZONS = (1, 3, 6, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument("--output-dir", default="reproduction/analysis/traffident_posthoc_residual_pilot")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--high-impact-weight", type=float, default=4.0)
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    return parser.parse_args()


def result_glob(repo_root: Path, county: str) -> str:
    return str(
        repo_root
        / "BasicTS"
        / "checkpoints"
        / "STID"
        / f"TraffiDent_{county}_2023Q1_100_12_12_pure"
        / "*"
        / "test_results"
    )


def find_stid_result_dir(repo_root: Path, county: str) -> Path:
    matches = [Path(path) for path in glob.glob(result_glob(repo_root, county))]
    matches = [path for path in matches if (path / "predictions.npy").exists() and (path / "targets.npy").exists()]
    if not matches:
        raise FileNotFoundError(f"No STID test_results found for {county}")
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


def make_prefix(values: np.ndarray) -> np.ndarray:
    prefix = np.zeros((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    prefix[1:] = np.cumsum(values, axis=0, dtype=np.float64)
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


def load_incident_attributes(data_dir: Path, num_steps: int, num_nodes: int) -> Dict[str, np.ndarray]:
    incidents = pd.read_csv(data_dir / "matched_incidents.csv")
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv")
    local = meta[["global_index", "Direction", "Fwy"]].copy()
    local["node_idx"] = np.arange(len(local), dtype=np.int64)
    incidents = incidents.merge(local, on=["global_index", "Fwy"], how="left")
    incidents = incidents.dropna(subset=["node_idx", "dt", "Type", "incident_abs_pm", "sensor_abs_pm"]).copy()
    incidents["node_idx"] = incidents["node_idx"].astype(int)
    incidents["event_slot"] = slot_from_dt(incidents["dt"])
    gap = incidents["sensor_abs_pm"].astype(float) - incidents["incident_abs_pm"].astype(float)
    sign = incidents["Direction"].map(direction_sign).astype(float)
    incidents["signed_downstream_pm"] = gap * sign
    incidents["relation"] = incidents["signed_downstream_pm"].map(relation_bucket)

    type_id = np.full((num_steps, num_nodes), len(TYPE_VALUES) - 1, dtype=np.int16)
    relation_id = np.full((num_steps, num_nodes), RELATION_VALUES.index("at_source"), dtype=np.int16)
    signed_pm = np.zeros((num_steps, num_nodes), dtype=np.float32)
    distance = np.zeros((num_steps, num_nodes), dtype=np.float32)

    type_lookup = {name: idx for idx, name in enumerate(TYPE_VALUES)}
    relation_lookup = {name: idx for idx, name in enumerate(RELATION_VALUES)}
    for row in incidents.itertuples(index=False):
        slot = int(row.event_slot)
        node = int(row.node_idx)
        if slot < 0 or slot >= num_steps or node < 0 or node >= num_nodes:
            continue
        end = min(num_steps, slot + 2)
        type_id[slot:end, node] = type_lookup.get(str(row.Type), len(TYPE_VALUES) - 1)
        relation_id[slot:end, node] = relation_lookup[str(row.relation)]
        signed_pm[slot:end, node] = float(row.signed_downstream_pm)
        distance[slot:end, node] = float(row.distance)
    return {
        "type_id": type_id,
        "relation_id": relation_id,
        "signed_pm": signed_pm,
        "distance": distance,
    }


def event_masks(event: np.ndarray, index: np.ndarray) -> Dict[str, np.ndarray]:
    prefix = make_prefix(event.astype(np.float32))
    start, mid, end = index[:, 0], index[:, 1], index[:, 2]
    hist_any = (prefix[mid] - prefix[start]) > 0
    fut_any = (prefix[end] - prefix[mid]) > 0
    return {
        "no_event": (~hist_any) & (~fut_any),
        "history_any": hist_any,
        "future_any": fut_any,
        "post_last_slot": event[mid - 1],
        "ongoing": hist_any & fut_any,
    }


def traffic_change(flow: np.ndarray, index: np.ndarray, input_len: int, output_len: int) -> np.ndarray:
    prefix = make_prefix(flow.astype(np.float32))
    start, mid, end = index[:, 0], index[:, 1], index[:, 2]
    hist_mean = (prefix[mid] - prefix[start]) / input_len
    fut_mean = (prefix[end] - prefix[mid]) / output_len
    return fut_mean - hist_mean


def timeweek_keys(index: np.ndarray, steps_per_day: int = 288) -> np.ndarray:
    mid = index[:, 1]
    return ((mid // steps_per_day) % 7) * steps_per_day + (mid % steps_per_day)


def control_mean_change(change: np.ndarray, no_event: np.ndarray, rows: np.ndarray, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_nodes = change.shape[1]
    num_keys = 7 * 288
    count = np.zeros((num_nodes, num_keys), dtype=np.int32)
    total = np.zeros((num_nodes, num_keys), dtype=np.float64)
    for row in rows:
        key = int(keys[row])
        nodes = np.flatnonzero(no_event[row] & np.isfinite(change[row]))
        if nodes.size == 0:
            continue
        count[nodes, key] += 1
        total[nodes, key] += change[row, nodes]
    return count, total


def matched_impact(
    change: np.ndarray,
    masks: Dict[str, np.ndarray],
    calib_rows: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    count, total = control_mean_change(change, masks["no_event"], calib_rows, keys)
    impact = np.full(change.shape, np.nan, dtype=np.float32)
    event_mask = masks["history_any"] | masks["future_any"]
    for row in range(change.shape[0]):
        nodes = np.flatnonzero(event_mask[row])
        if nodes.size == 0:
            continue
        key = int(keys[row])
        c = count[nodes, key]
        valid = c > 0
        if not valid.any():
            continue
        nodes = nodes[valid]
        control = total[nodes, key] / count[nodes, key]
        impact[row, nodes] = change[row, nodes] - control
    return impact


def latest_event_features(
    row: int,
    node: int,
    index: np.ndarray,
    event: np.ndarray,
    attrs: Dict[str, np.ndarray],
) -> Tuple[int, int, float, float, int]:
    start, mid, _ = index[row]
    for slot in range(int(mid) - 1, int(start) - 1, -1):
        if event[slot, node]:
            return (
                int(attrs["type_id"][slot, node]),
                int(attrs["relation_id"][slot, node]),
                float(attrs["signed_pm"][slot, node]),
                float(attrs["distance"][slot, node]),
                int(mid - 1 - slot),
            )
    return len(TYPE_VALUES) - 1, RELATION_VALUES.index("at_source"), 0.0, 0.0, 99


def case_feature(
    county_id: int,
    type_id: int,
    relation_id: int,
    signed_pm: float,
    distance: float,
    age_slots: int,
    pre: np.ndarray,
    horizon: int,
) -> List[float]:
    values: List[float] = [1.0]
    values.extend(1.0 if county_id == idx else 0.0 for idx in range(len(COUNTIES)))
    values.extend(1.0 if type_id == idx else 0.0 for idx in range(len(TYPE_VALUES)))
    values.extend(1.0 if relation_id == idx else 0.0 for idx in range(len(RELATION_VALUES)))
    pre = pre.astype(float)
    values.extend(
        [
            float(signed_pm),
            float(abs(signed_pm)),
            float(distance),
            float(min(age_slots, 12) / 12.0),
            float(pre[-1]),
            float(pre.mean()),
            float(pre.std()),
            float((pre[-1] - pre[0]) / max(1, len(pre) - 1)),
            float(horizon / 12.0),
        ]
    )
    values.extend(1.0 if horizon == h else 0.0 for h in HORIZONS)
    return values


def build_design_for_county(
    county: str,
    county_id: int,
    data: np.ndarray,
    index: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    masks: Dict[str, np.ndarray],
    attrs: Dict[str, np.ndarray],
    impact: np.ndarray,
    rows: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    flow = data[:, :, 0]
    features: List[List[float]] = []
    labels: List[float] = []
    weights: List[float] = []
    locations: List[Tuple[int, int, int]] = []
    calib_impact_values = np.abs(impact[rows][np.isfinite(impact[rows])])
    high_thr = float(np.quantile(calib_impact_values, 0.75)) if calib_impact_values.size else np.inf

    for row in rows:
        nodes = np.flatnonzero(masks["history_any"][row])
        for node in nodes:
            type_id, relation_id, signed_pm, distance, age_slots = latest_event_features(row, int(node), index, data[:, :, args.event_channel] > 0, attrs)
            start, mid, _ = index[row]
            pre = flow[start:mid, node]
            if not np.all(np.isfinite(pre)):
                continue
            imp = impact[row, node]
            base_weight = 1.0 + args.high_impact_weight * float(np.isfinite(imp) and abs(float(imp)) >= high_thr)
            for h in range(args.output_len):
                y = float(target[row, h, node, 0])
                p = float(pred[row, h, node, 0])
                if not np.isfinite(y) or abs(y - args.null_val) <= 5e-5:
                    continue
                features.append(
                    case_feature(
                        county_id=county_id,
                        type_id=type_id,
                        relation_id=relation_id,
                        signed_pm=signed_pm,
                        distance=distance,
                        age_slots=age_slots,
                        pre=pre,
                        horizon=h + 1,
                    )
                )
                labels.append(y - p)
                weights.append(base_weight)
                locations.append((int(row), int(node), h))
    if not features:
        return np.empty((0, 1)), np.empty((0,)), np.empty((0,)), []
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64), np.asarray(weights, dtype=np.float64), locations


def weighted_ridge(x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float) -> np.ndarray:
    sqrt_w = np.sqrt(weight)[:, None]
    xw = x * sqrt_w
    yw = y * sqrt_w[:, 0]
    reg = alpha * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)


def metric(pred: np.ndarray, target: np.ndarray, mask_2d: np.ndarray, null_val: float) -> Dict[str, float]:
    valid = mask_2d[:, None, :] & null_mask(target, null_val)
    count = int(valid.sum())
    if count == 0:
        return {"valid_values": 0, "MAE": np.nan, "RMSE": np.nan, "bias": np.nan}
    err = pred - target
    out = {
        "valid_values": count,
        "MAE": float(np.abs(err)[valid].mean()),
        "RMSE": float(np.sqrt((err[valid] ** 2).mean())),
        "bias": float(err[valid].mean()),
    }
    for h in (3, 6, 12):
        h_valid = mask_2d & null_mask(target[:, h - 1, :], null_val)
        if h_valid.any():
            h_err = err[:, h - 1, :]
            out[f"MAE@h{h}"] = float(np.abs(h_err[h_valid]).mean())
        else:
            out[f"MAE@h{h}"] = np.nan
    return out


def evaluate_county(
    county: str,
    pred: np.ndarray,
    target: np.ndarray,
    correction: np.ndarray,
    masks: Dict[str, np.ndarray],
    impact: np.ndarray,
    eval_rows: np.ndarray,
    null_val: float,
) -> List[Dict[str, object]]:
    pred_eval = np.asarray(pred[eval_rows, :, :, 0], dtype=np.float32)
    target_eval = np.asarray(target[eval_rows, :, :, 0], dtype=np.float32)
    adj_eval = pred_eval + correction
    impact_eval = impact[eval_rows]
    high_valid = np.isfinite(impact_eval)
    high_abs = np.abs(impact_eval[high_valid])
    high_thr = float(np.quantile(high_abs, 0.75)) if high_abs.size else np.inf

    groups = {
        "all_eval": np.ones_like(masks["history_any"][eval_rows], dtype=bool),
        "no_event": masks["no_event"][eval_rows],
        "history_any": masks["history_any"][eval_rows],
        "post_last_slot": masks["post_last_slot"][eval_rows],
        "ongoing": masks["ongoing"][eval_rows],
        "high_impact_history_any": masks["history_any"][eval_rows] & high_valid & (np.abs(impact_eval) >= high_thr),
        "high_impact_post_last": masks["post_last_slot"][eval_rows] & high_valid & (np.abs(impact_eval) >= high_thr),
        "impact_drop_history_any": masks["history_any"][eval_rows] & np.isfinite(impact_eval) & (impact_eval < 0),
        "impact_rise_history_any": masks["history_any"][eval_rows] & np.isfinite(impact_eval) & (impact_eval > 0),
    }
    rows = []
    for group, group_mask in groups.items():
        stid_metrics = metric(pred_eval, target_eval, group_mask, null_val)
        adj_metrics = metric(adj_eval, target_eval, group_mask, null_val)
        rows.append(
            {
                "county": county,
                "group": group,
                "node_windows": int(group_mask.sum()),
                "STID_MAE": stid_metrics["MAE"],
                "Posthoc_MAE": adj_metrics["MAE"],
                "delta_vs_STID_MAE": adj_metrics["MAE"] - stid_metrics["MAE"],
                "STID_RMSE": stid_metrics["RMSE"],
                "Posthoc_RMSE": adj_metrics["RMSE"],
                "STID_bias": stid_metrics["bias"],
                "Posthoc_bias": adj_metrics["bias"],
                "Posthoc_MAE@h3": adj_metrics.get("MAE@h3", np.nan),
                "Posthoc_MAE@h6": adj_metrics.get("MAE@h6", np.nan),
                "Posthoc_MAE@h12": adj_metrics.get("MAE@h12", np.nan),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    county_cache = {}
    all_x, all_y, all_w = [], [], []
    metadata = {"mode": "test_result calibration split; exploratory only", "counties": {}}

    for county_id, county in enumerate(args.counties):
        data_dir = data_root / f"TraffiDent_{county}_2023Q1"
        data = np.load(data_dir / "data.npz")["data"]
        index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
        split = len(index) // 2
        calib_rows = np.arange(0, split, dtype=np.int64)
        eval_rows = np.arange(split, len(index), dtype=np.int64)
        shape = (len(index), args.output_len, data.shape[1], 1)
        result_dir = find_stid_result_dir(repo_root, county)
        pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
        target = load_raw_or_npy(result_dir / "targets.npy", shape)
        event = data[:, :, args.event_channel] > 0
        masks = event_masks(event, index)
        attrs = load_incident_attributes(data_dir, data.shape[0], data.shape[1])
        change = traffic_change(data[:, :, 0], index, args.input_len, args.output_len)
        keys = timeweek_keys(index)
        impact = matched_impact(change, masks, calib_rows, keys)

        x, y, w, _ = build_design_for_county(
            county=county,
            county_id=county_id,
            data=data,
            index=index,
            pred=pred,
            target=target,
            masks=masks,
            attrs=attrs,
            impact=impact,
            rows=calib_rows,
            args=args,
        )
        if len(x):
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
        county_cache[county] = {
            "data": data,
            "index": index,
            "pred": pred,
            "target": target,
            "masks": masks,
            "attrs": attrs,
            "impact": impact,
            "calib_rows": calib_rows,
            "eval_rows": eval_rows,
        }
        metadata["counties"][county] = {
            "result_dir": str(result_dir.relative_to(repo_root)),
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "calibration_rows": int(len(calib_rows)),
            "evaluation_rows": int(len(eval_rows)),
            "calibration_records": int(len(x)),
        }

    x_train = np.concatenate(all_x, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    w_train = np.concatenate(all_w, axis=0)
    coef = weighted_ridge(x_train, y_train, w_train, args.ridge_alpha)
    metadata["train_records_total"] = int(len(x_train))
    metadata["feature_dim"] = int(x_train.shape[1])

    result_rows = []
    for county_id, county in enumerate(args.counties):
        cache = county_cache[county]
        eval_rows = cache["eval_rows"]
        x_eval, _, _, locations = build_design_for_county(
            county=county,
            county_id=county_id,
            data=cache["data"],
            index=cache["index"],
            pred=cache["pred"],
            target=cache["target"],
            masks=cache["masks"],
            attrs=cache["attrs"],
            impact=cache["impact"],
            rows=eval_rows,
            args=args,
        )
        correction = np.zeros((len(eval_rows), args.output_len, cache["data"].shape[1]), dtype=np.float32)
        row_to_eval = {int(row): idx for idx, row in enumerate(eval_rows)}
        if len(x_eval):
            pred_resid = x_eval @ coef
            for value, (row, node, horizon) in zip(pred_resid, locations):
                correction[row_to_eval[int(row)], horizon, int(node)] = float(value)
        result_rows.extend(
            evaluate_county(
                county=county,
                pred=cache["pred"],
                target=cache["target"],
                correction=correction,
                masks=cache["masks"],
                impact=cache["impact"],
                eval_rows=eval_rows,
                null_val=args.null_val,
            )
        )
        metadata["counties"][county]["evaluation_records"] = int(len(x_eval))

    df = pd.DataFrame(result_rows)
    df.to_csv(output_dir / "posthoc_residual_pilot_metrics.csv", index=False)
    summary_rows = []
    for group, group_df in df.groupby("group"):
        deltas = group_df["delta_vs_STID_MAE"].dropna()
        summary_rows.append(
            {
                "group": group,
                "counties": int(group_df["county"].nunique()),
                "total_node_windows": int(group_df["node_windows"].sum()),
                "mean_delta_vs_STID_MAE": float(deltas.mean()) if len(deltas) else np.nan,
                "wins_vs_STID": int((deltas < 0).sum()),
                "mean_STID_MAE": float(group_df["STID_MAE"].mean()),
                "mean_Posthoc_MAE": float(group_df["Posthoc_MAE"].mean()),
            }
        )
    pd.DataFrame(summary_rows).sort_values("group").to_csv(
        output_dir / "posthoc_residual_pilot_summary.csv", index=False
    )
    metadata["files"] = {
        "metrics_csv": str(output_dir / "posthoc_residual_pilot_metrics.csv"),
        "summary_csv": str(output_dir / "posthoc_residual_pilot_summary.csv"),
    }
    with open(output_dir / "posthoc_residual_pilot_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(json.dumps(metadata["files"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
