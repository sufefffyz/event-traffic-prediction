#!/usr/bin/env python3
"""Exponential-decay incident residual pilot over saved STID test predictions.

This is a low-cost exploratory pilot, not a final test protocol. It uses the
first half of each saved STID test_result for calibration and evaluates on the
second half. The goal is to test whether an explicit incident kernel,

    K = type/relation features * exp(-space / lambda_s) * exp(-time / lambda_t),

is a better use of future/near-future incident information than a binary future
accident flag.

No future target flow is used as an inference feature. The only oracle signal is
the incident record time/type/location used to construct incident-node-horizon
triples.
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
TIME_BUCKETS = ("pre", "onset", "post")
HORIZONS = (1, 3, 6, 12)
RELATION_EPS_PM = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument(
        "--output-dir",
        default="reproduction/analysis/traffident_decay_kernel_pilot",
    )
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--lambda-space", type=float, default=1.0)
    parser.add_argument("--lambda-time-post", type=float, default=6.0)
    parser.add_argument("--lambda-time-pre", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=3.0)
    parser.add_argument("--max-pre-slots", type=int, default=12)
    parser.add_argument("--max-post-slots", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--clip-residual", type=float, default=20.0)
    parser.add_argument("--same-direction-only", action="store_true")
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
    matches = [
        path
        for path in matches
        if (path / "predictions.npy").exists() and (path / "targets.npy").exists()
    ]
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


def type_bucket(value: object) -> str:
    text = str(value)
    return text if text in TYPE_VALUES[:-1] else "other"


def time_bucket(delta_slots: int) -> str:
    if delta_slots < 0:
        return "pre"
    if delta_slots == 0:
        return "onset"
    return "post"


def make_prefix(mask: np.ndarray) -> np.ndarray:
    prefix = np.zeros((mask.shape[0] + 1, mask.shape[1]), dtype=np.int32)
    prefix[1:] = np.cumsum(mask.astype(np.int32), axis=0, dtype=np.int32)
    return prefix


def event_masks(event: np.ndarray, index: np.ndarray) -> Dict[str, np.ndarray]:
    prefix = make_prefix(event)
    start, mid, end = index[:, 0], index[:, 1], index[:, 2]
    hist_any = (prefix[mid] - prefix[start]) > 0
    fut_any = (prefix[end] - prefix[mid]) > 0
    return {
        "no_event": (~hist_any) & (~fut_any),
        "future_any": fut_any,
        "history_any": hist_any,
        "future_onset": (~hist_any) & fut_any,
        "history_only": hist_any & (~fut_any),
        "ongoing": hist_any & fut_any,
        "post_last_slot": event[mid - 1],
    }


def load_incidents_and_meta(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    incidents = pd.read_csv(data_dir / "matched_incidents.csv")
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv").copy()
    meta["node_idx"] = np.arange(len(meta), dtype=np.int64)

    local = meta[["global_index", "Fwy", "Direction", "Abs PM", "node_idx"]].copy()
    incidents = incidents.merge(
        local,
        on=["global_index", "Fwy"],
        how="left",
        suffixes=("", "_matched_sensor"),
    )
    incidents = incidents.dropna(
        subset=["dt", "Type", "incident_abs_pm", "Fwy", "Direction"]
    ).copy()
    incidents["event_slot"] = slot_from_dt(incidents["dt"])
    incidents["type_bucket"] = incidents["Type"].map(type_bucket)
    incidents["direction_sign"] = incidents["Direction"].map(direction_sign).astype(float)
    # Multiple matched sensors can point to the same incident. Keep one incident
    # record because spatial expansion is recomputed from all sensors below.
    incidents = incidents.drop_duplicates(
        subset=["incident_id", "dt", "Fwy", "incident_abs_pm", "Type", "Direction"]
    ).reset_index(drop=True)
    return incidents, meta


def feature_names(counties: Iterable[str]) -> List[str]:
    names = ["bias"]
    names += [f"county={county}" for county in counties]
    names += ["kernel_sum", "kernel_sq_sum", "kernel_max"]
    names += [f"type={value}" for value in TYPE_VALUES]
    names += [f"relation={value}" for value in RELATION_VALUES]
    names += [f"time={value}" for value in TIME_BUCKETS]
    names += [f"type_relation={t}:{r}" for t in TYPE_VALUES for r in RELATION_VALUES]
    names += ["signed_pm", "abs_pm", "delta_slots", "abs_delta_slots"]
    names += ["hist_last", "hist_mean", "hist_std", "hist_trend"]
    names += [f"horizon={h}" for h in HORIZONS]
    return names


def empty_feature(county: str, counties: List[str]) -> np.ndarray:
    names = feature_names(counties)
    x = np.zeros(len(names), dtype=np.float64)
    x[0] = 1.0
    offset = 1
    x[offset + counties.index(county)] = 1.0
    return x


def add_kernel_features(
    x: np.ndarray,
    k: float,
    signed_pm: float,
    delta_slots: int,
    type_value: str,
    relation_value: str,
    names: List[str],
) -> None:
    x[names.index("kernel_sum")] += k
    x[names.index("kernel_sq_sum")] += k * k
    x[names.index("kernel_max")] = max(x[names.index("kernel_max")], k)
    x[names.index(f"type={type_value}")] += k
    x[names.index(f"relation={relation_value}")] += k
    tb = time_bucket(delta_slots)
    x[names.index(f"time={tb}")] += k
    x[names.index(f"type_relation={type_value}:{relation_value}")] += k
    x[names.index("signed_pm")] += k * signed_pm
    x[names.index("abs_pm")] += k * abs(signed_pm)
    x[names.index("delta_slots")] += k * float(delta_slots)
    x[names.index("abs_delta_slots")] += k * abs(float(delta_slots))


def add_history_features(
    x: np.ndarray,
    names: List[str],
    flow: np.ndarray,
    index: np.ndarray,
    row: int,
    node: int,
    horizon: int,
) -> bool:
    start, mid, _ = index[row]
    pre = flow[start:mid, node].astype(np.float64)
    if not np.all(np.isfinite(pre)):
        return False
    scale = max(float(x[names.index("kernel_sum")]), 1e-6)
    x[names.index("signed_pm")] /= scale
    x[names.index("abs_pm")] /= scale
    x[names.index("delta_slots")] /= scale
    x[names.index("abs_delta_slots")] /= scale
    x[names.index("hist_last")] = float(pre[-1])
    x[names.index("hist_mean")] = float(pre.mean())
    x[names.index("hist_std")] = float(pre.std())
    x[names.index("hist_trend")] = float((pre[-1] - pre[0]) / max(1, len(pre) - 1))
    if horizon in HORIZONS:
        x[names.index(f"horizon={horizon}")] = 1.0
    return True


def iter_affected_nodes(
    meta: pd.DataFrame,
    incident: object,
    args: argparse.Namespace,
) -> Iterable[Tuple[int, float, str]]:
    same_fwy = meta["Fwy"].to_numpy() == int(incident.Fwy)
    if args.same_direction_only:
        same_fwy &= meta["Direction"].astype(str).to_numpy() == str(incident.Direction)
    idxs = np.flatnonzero(same_fwy)
    if idxs.size == 0:
        return
    abs_pm = meta["Abs PM"].to_numpy(dtype=np.float64)[idxs]
    direction = meta["Direction"].to_numpy()[idxs]
    signs = np.asarray([direction_sign(x) for x in direction], dtype=np.float64)
    signed_pm = (abs_pm - float(incident.incident_abs_pm)) * signs
    keep = np.abs(signed_pm) <= args.max_distance
    for node, signed in zip(idxs[keep], signed_pm[keep]):
        yield int(node), float(signed), relation_bucket(float(signed))


def temporal_kernel(delta_slots: int, args: argparse.Namespace) -> float:
    if delta_slots >= 0:
        return float(np.exp(-float(delta_slots) / args.lambda_time_post))
    return float(np.exp(float(delta_slots) / args.lambda_time_pre))


def spatial_kernel(signed_pm: float, args: argparse.Namespace) -> float:
    return float(np.exp(-abs(float(signed_pm)) / args.lambda_space))


def build_design(
    county: str,
    counties: List[str],
    data: np.ndarray,
    index: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    incidents: pd.DataFrame,
    meta: pd.DataFrame,
    rows: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    names = feature_names(counties)
    row_set = set(int(r) for r in rows)
    mid_to_rows: Dict[int, List[int]] = defaultdict(list)
    for row in rows:
        mid_to_rows[int(index[row, 1])].append(int(row))
    mid_values = np.asarray(list(mid_to_rows.keys()), dtype=np.int64)
    min_mid = int(mid_values.min()) if mid_values.size else 0
    max_mid = int(mid_values.max()) if mid_values.size else -1

    features: Dict[Tuple[int, int, int], np.ndarray] = {}
    kernel_any = np.zeros((len(index), data.shape[1]), dtype=bool)
    kernel_post_any = np.zeros((len(index), data.shape[1]), dtype=bool)

    min_delta = -args.max_pre_slots
    max_delta = args.max_post_slots
    for incident in incidents.itertuples(index=False):
        event_slot = int(incident.event_slot)
        if event_slot < 0:
            continue
        # Union over horizons:
        # mid in [event_slot + min_delta - (H - 1), event_slot + max_delta].
        if event_slot + min_delta - (args.output_len - 1) > max_mid:
            continue
        if event_slot + max_delta < min_mid:
            continue
        type_value = type_bucket(incident.Type)
        affected = list(iter_affected_nodes(meta, incident, args))
        if not affected:
            continue
        for h0 in range(args.output_len):
            horizon = h0 + 1
            # delta = target_slot - event_slot = mid + h0 - event_slot.
            lo_mid = event_slot + min_delta - h0
            hi_mid = event_slot + max_delta - h0
            for mid in range(lo_mid, hi_mid + 1):
                for row in mid_to_rows.get(mid, []):
                    if row not in row_set:
                        continue
                    delta = int(mid + h0 - event_slot)
                    kt = temporal_kernel(delta, args)
                    if kt <= 0:
                        continue
                    for node, signed_pm, relation in affected:
                        ks = spatial_kernel(signed_pm, args)
                        k = ks * kt
                        if k <= 1e-8:
                            continue
                        key = (row, node, h0)
                        if key not in features:
                            features[key] = empty_feature(county, counties)
                        add_kernel_features(
                            features[key],
                            k=k,
                            signed_pm=signed_pm,
                            delta_slots=delta,
                            type_value=type_value,
                            relation_value=relation,
                            names=names,
                        )
                        kernel_any[row, node] = True
                        if delta >= 0:
                            kernel_post_any[row, node] = True

    flow = data[:, :, 0]
    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    weights: List[float] = []
    locations: List[Tuple[int, int, int]] = []
    for (row, node, h0), x in features.items():
        horizon = h0 + 1
        if not add_history_features(x, names, flow, index, row, node, horizon):
            continue
        y = float(target[row, h0, node, 0])
        p = float(pred[row, h0, node, 0])
        if not np.isfinite(y) or abs(y - args.null_val) <= 5e-5:
            continue
        x_rows.append(x)
        y_rows.append(y - p)
        weights.append(1.0 + float(x[names.index("kernel_sum")]))
        locations.append((int(row), int(node), int(h0)))

    if not x_rows:
        return (
            np.empty((0, len(names)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
            kernel_any,
        )
    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        locations,
        kernel_any | kernel_post_any,
    )


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
    for h in HORIZONS:
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
    kernel_mask: np.ndarray,
    eval_rows: np.ndarray,
    null_val: float,
) -> List[Dict[str, object]]:
    pred_eval = np.asarray(pred[eval_rows, :, :, 0], dtype=np.float32)
    target_eval = np.asarray(target[eval_rows, :, :, 0], dtype=np.float32)
    adj_eval = pred_eval + correction
    kernel_eval = kernel_mask[eval_rows]
    groups = {
        "all_eval": np.ones_like(kernel_eval, dtype=bool),
        "kernel_candidate": kernel_eval,
        "no_event": masks["no_event"][eval_rows],
        "future_any": masks["future_any"][eval_rows],
        "future_onset": masks["future_onset"][eval_rows],
        "history_any": masks["history_any"][eval_rows],
        "history_only": masks["history_only"][eval_rows],
        "ongoing": masks["ongoing"][eval_rows],
        "post_last_slot": masks["post_last_slot"][eval_rows],
        "kernel_and_future_any": kernel_eval & masks["future_any"][eval_rows],
        "kernel_and_post_last": kernel_eval & masks["post_last_slot"][eval_rows],
    }
    rows: List[Dict[str, object]] = []
    for group, group_mask in groups.items():
        stid_metrics = metric(pred_eval, target_eval, group_mask, null_val)
        adj_metrics = metric(adj_eval, target_eval, group_mask, null_val)
        rows.append(
            {
                "county": county,
                "group": group,
                "node_windows": int(group_mask.sum()),
                "STID_MAE": stid_metrics["MAE"],
                "DecayKernel_MAE": adj_metrics["MAE"],
                "delta_vs_STID_MAE": adj_metrics["MAE"] - stid_metrics["MAE"],
                "STID_RMSE": stid_metrics["RMSE"],
                "DecayKernel_RMSE": adj_metrics["RMSE"],
                "STID_bias": stid_metrics["bias"],
                "DecayKernel_bias": adj_metrics["bias"],
                "DecayKernel_MAE@h3": adj_metrics.get("MAE@h3", np.nan),
                "DecayKernel_MAE@h6": adj_metrics.get("MAE@h6", np.nan),
                "DecayKernel_MAE@h12": adj_metrics.get("MAE@h12", np.nan),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counties = list(args.counties)

    county_cache = {}
    all_x, all_y, all_w = [], [], []
    metadata = {
        "mode": "STID test_result calibration split; exploratory only",
        "lambda_space": args.lambda_space,
        "lambda_time_post": args.lambda_time_post,
        "lambda_time_pre": args.lambda_time_pre,
        "max_distance": args.max_distance,
        "max_pre_slots": args.max_pre_slots,
        "max_post_slots": args.max_post_slots,
        "ridge_alpha": args.ridge_alpha,
        "clip_residual": args.clip_residual,
        "same_direction_only": bool(args.same_direction_only),
        "feature_names": feature_names(counties),
        "counties": {},
    }

    for county in counties:
        print(f"[build] {county}", flush=True)
        data_dir = data_root / f"TraffiDent_{county}_2023Q1"
        data = np.load(data_dir / "data.npz")["data"]
        index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
        shape = (len(index), args.output_len, data.shape[1], 1)
        result_dir = find_stid_result_dir(repo_root, county)
        pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
        target = load_raw_or_npy(result_dir / "targets.npy", shape)
        event = data[:, :, args.event_channel] > 0
        masks = event_masks(event, index)
        incidents, meta = load_incidents_and_meta(data_dir)
        split = len(index) // 2
        calib_rows = np.arange(0, split, dtype=np.int64)
        eval_rows = np.arange(split, len(index), dtype=np.int64)
        x, y, w, locations, kernel_mask = build_design(
            county=county,
            counties=counties,
            data=data,
            index=index,
            pred=pred,
            target=target,
            incidents=incidents,
            meta=meta,
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
            "incidents": incidents,
            "meta": meta,
            "eval_rows": eval_rows,
            "kernel_mask_calib": kernel_mask,
        }
        metadata["counties"][county] = {
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "result_dir": str(result_dir.relative_to(repo_root)),
            "calibration_rows": int(len(calib_rows)),
            "evaluation_rows": int(len(eval_rows)),
            "calibration_records": int(len(x)),
            "calibration_locations": int(len(locations)),
            "incident_records": int(len(incidents)),
        }
        print(f"[build] {county} calibration_records={len(x)}", flush=True)

    if not all_x:
        raise RuntimeError("No calibration records were built.")
    x_train = np.concatenate(all_x, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    w_train = np.concatenate(all_w, axis=0)
    print(f"[fit] records={len(x_train)} features={x_train.shape[1]}", flush=True)
    coef = weighted_ridge(x_train, y_train, w_train, args.ridge_alpha)
    metadata["train_records_total"] = int(len(x_train))
    metadata["feature_dim"] = int(x_train.shape[1])
    np.save(output_dir / "decay_kernel_ridge_coef.npy", coef)

    result_rows = []
    for county in counties:
        print(f"[eval] {county}", flush=True)
        cache = county_cache[county]
        eval_rows = cache["eval_rows"]
        x_eval, _, _, locations, kernel_mask_eval = build_design(
            county=county,
            counties=counties,
            data=cache["data"],
            index=cache["index"],
            pred=cache["pred"],
            target=cache["target"],
            incidents=cache["incidents"],
            meta=cache["meta"],
            rows=eval_rows,
            args=args,
        )
        correction = np.zeros(
            (len(eval_rows), args.output_len, cache["data"].shape[1]),
            dtype=np.float32,
        )
        row_to_eval = {int(row): idx for idx, row in enumerate(eval_rows)}
        if len(x_eval):
            pred_resid = x_eval @ coef
            pred_resid = np.clip(pred_resid, -args.clip_residual, args.clip_residual)
            for value, (row, node, horizon) in zip(pred_resid, locations):
                correction[row_to_eval[int(row)], horizon, int(node)] = float(value)
        result_rows.extend(
            evaluate_county(
                county=county,
                pred=cache["pred"],
                target=cache["target"],
                correction=correction,
                masks=cache["masks"],
                kernel_mask=kernel_mask_eval,
                eval_rows=eval_rows,
                null_val=args.null_val,
            )
        )
        metadata["counties"][county]["evaluation_records"] = int(len(x_eval))
        metadata["counties"][county]["evaluation_locations"] = int(len(locations))
        print(f"[eval] {county} evaluation_records={len(x_eval)}", flush=True)

    metrics_path = output_dir / "decay_kernel_pilot_metrics.csv"
    summary_path = output_dir / "decay_kernel_pilot_summary.csv"
    metadata_path = output_dir / "decay_kernel_pilot_metadata.json"
    df = pd.DataFrame(result_rows)
    df.to_csv(metrics_path, index=False)

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
                "mean_DecayKernel_MAE": float(group_df["DecayKernel_MAE"].mean()),
            }
        )
    pd.DataFrame(summary_rows).sort_values("group").to_csv(summary_path, index=False)
    metadata["files"] = {
        "metrics_csv": str(metrics_path),
        "summary_csv": str(summary_path),
        "coef_npy": str(output_dir / "decay_kernel_ridge_coef.npy"),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(json.dumps(metadata["files"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
