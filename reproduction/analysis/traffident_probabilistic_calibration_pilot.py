#!/usr/bin/env python3
"""STID-fixed probabilistic calibration pilot for TraffiDent.

The script keeps the saved STID mean prediction fixed and learns only a
heteroscedastic Gaussian scale. It compares traffic/time uncertainty features
against incident type-space-time fields to test whether incident information
improves probabilistic forecasting beyond a strong traffic-state baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reproduction.analysis.traffident_decay_kernel_pilot import (  # noqa: E402
    COUNTIES,
    RELATION_VALUES,
    TIME_BUCKETS,
    TYPE_VALUES,
    event_masks,
    find_stid_result_dir,
    load_incidents_and_meta,
    load_raw_or_npy,
    null_mask,
)
from reproduction.analysis.traffident_matched_control_residual_audit import (  # noqa: E402
    load_type_event_masks,
    time_keys,
)
from reproduction.analysis.traffident_type_risk_pilot import (  # noqa: E402
    build_incident_field,
    incident_feature_names,
    make_flow_stats,
    sample_flat,
)


Z_VALUES = {
    0.05: -1.6448536269514729,
    0.10: -1.2815515655446004,
    0.25: -0.6744897501960817,
    0.50: 0.0,
    0.75: 0.6744897501960817,
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
}

GROUPS = (
    "all_eval_sample",
    "no_event_sample",
    "future_any",
    "future_onset",
    "history_any",
    "history_only",
    "ongoing",
    "post_last_slot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument(
        "--output-dir",
        default="reproduction/analysis/traffident_probabilistic_calibration_pilot",
    )
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--start-weekday", type=int, default=6, help="2023-01-01 is Sunday=6")
    parser.add_argument("--type-event-window-slots", type=int, default=2)
    parser.add_argument("--incident-scope", choices=("history", "history_future"), default="history_future")
    parser.add_argument("--lambda-space", type=float, default=1.0)
    parser.add_argument("--lambda-time-post", type=float, default=6.0)
    parser.add_argument("--lambda-time-pre", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--max-pre-slots", type=int, default=6)
    parser.add_argument("--max-post-slots", type=int, default=12)
    parser.add_argument("--same-direction-only", action="store_true")
    parser.add_argument("--max-train-records-per-county", type=int, default=300_000)
    parser.add_argument("--max-eval-control-records-per-county", type=int, default=120_000)
    parser.add_argument("--max-eval-event-records-per-county", type=int, default=220_000)
    parser.add_argument("--sigma-l2", type=float, default=1e-4)
    parser.add_argument("--sigma-lr", type=float, default=0.03)
    parser.add_argument("--sigma-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def feature_names(counties: Iterable[str], output_len: int) -> List[str]:
    names = ["bias"]
    names += [f"county={county}" for county in counties]
    names += [f"horizon={h}" for h in range(1, output_len + 1)]
    names += [
        "hist_last",
        "hist_mean",
        "hist_std",
        "hist_trend",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
    ]
    names += incident_feature_names()
    return names


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def model_columns(names: List[str], model: str) -> np.ndarray:
    base = ["bias"]
    base += [name for name in names if name.startswith("county=")]
    base += [name for name in names if name.startswith("horizon=")]
    traffic = [
        "hist_last",
        "hist_mean",
        "hist_std",
        "hist_trend",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
    ]
    incident = ["kernel_sum", "kernel_sq_sum", "kernel_max"]
    incident += [name for name in names if name.startswith("type=")]
    incident += [name for name in names if name.startswith("relation=")]
    incident += [name for name in names if name.startswith("time=")]
    if model == "constant_sigma":
        keep = base
    elif model == "traffic_time_sigma":
        keep = base + traffic
    elif model == "incident_field_sigma":
        keep = base + incident
    elif model == "full_sigma":
        keep = base + traffic + incident
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.asarray([names.index(name) for name in keep], dtype=np.int64)


def sample_window_records(
    mask_2d: np.ndarray,
    valid_3d: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_2d, nodes_2d = sample_flat(mask_2d & valid_3d.any(axis=1), max_count, rng)
    if len(rows_2d) == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    rows = np.repeat(rows_2d, valid_3d.shape[1])
    nodes = np.repeat(nodes_2d, valid_3d.shape[1])
    h0 = np.tile(np.arange(valid_3d.shape[1], dtype=np.int64), len(rows_2d))
    keep = valid_3d[rows, h0, nodes]
    rows, nodes, h0 = rows[keep], nodes[keep], h0[keep]
    if max_count > 0 and len(rows) > max_count:
        idx = rng.choice(len(rows), size=max_count, replace=False)
        rows, nodes, h0 = rows[idx], nodes[idx], h0[idx]
    return rows.astype(np.int64), nodes.astype(np.int64), h0.astype(np.int64)


def build_features(
    county: str,
    counties: List[str],
    rows_global: np.ndarray,
    nodes: np.ndarray,
    h0: np.ndarray,
    all_rows: np.ndarray,
    index: np.ndarray,
    flow_stats: Dict[str, np.ndarray],
    incident_field: np.ndarray,
    incident_names: List[str],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[str]]:
    names = feature_names(counties, args.output_len)
    col = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(rows_global), len(names)), dtype=np.float32)
    x[:, col["bias"]] = 1.0
    x[:, col[f"county={county}"]] = 1.0
    for h in range(1, args.output_len + 1):
        x[h0 == h - 1, col[f"horizon={h}"]] = 1.0

    for name in ("hist_last", "hist_mean", "hist_std", "hist_trend"):
        x[:, col[name]] = flow_stats[name][rows_global, nodes].astype(np.float32)

    keys = time_keys(index[rows_global, 1], args.steps_per_day, args.start_weekday)
    tod = (keys % args.steps_per_day).astype(np.float32)
    dow = (keys // args.steps_per_day).astype(np.float32)
    x[:, col["tod_sin"]] = np.sin(2.0 * np.pi * tod / float(args.steps_per_day))
    x[:, col["tod_cos"]] = np.cos(2.0 * np.pi * tod / float(args.steps_per_day))
    x[:, col["dow_sin"]] = np.sin(2.0 * np.pi * dow / 7.0)
    x[:, col["dow_cos"]] = np.cos(2.0 * np.pi * dow / 7.0)

    row_lookup = {int(row): local for local, row in enumerate(all_rows)}
    local_rows = np.asarray([row_lookup[int(row)] for row in rows_global], dtype=np.int64)
    incident_values = incident_field[local_rows, nodes]
    for src_idx, name in enumerate(incident_names):
        x[:, col[name]] = incident_values[:, src_idx]
    return x, names


def fit_log_sigma(
    x: np.ndarray,
    residual: np.ndarray,
    l2: float,
    lr: float,
    epochs: int,
) -> np.ndarray:
    coef = np.zeros(x.shape[1], dtype=np.float64)
    init = np.log(max(float(np.sqrt(np.mean(residual**2))), 1e-3))
    coef[0] = init
    residual_sq = residual.astype(np.float64) ** 2
    for _ in range(epochs):
        log_sigma = np.clip(x @ coef, -5.0, 6.0)
        inv_var_err = residual_sq * np.exp(-2.0 * log_sigma)
        grad = x.T @ (1.0 - inv_var_err) / len(residual)
        reg = l2 * coef
        reg[0] = 0.0
        coef -= lr * (grad + reg)
    return coef


def predict_sigma(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x @ coef, -5.0, 6.0))


def pinball_loss(residual: np.ndarray, sigma: np.ndarray) -> float:
    losses = []
    for tau, z in Z_VALUES.items():
        q_res = z * sigma
        diff = residual - q_res
        losses.append(np.maximum(tau * diff, (tau - 1.0) * diff))
    return float(np.mean(np.stack(losses, axis=0)))


def gaussian_metrics(residual: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    sigma = np.maximum(sigma.astype(np.float64), 1e-6)
    residual = residual.astype(np.float64)
    nll = 0.5 * (residual / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2.0 * np.pi)
    abs_res = np.abs(residual)
    cov80 = abs_res <= Z_VALUES[0.90] * sigma
    cov90 = abs_res <= Z_VALUES[0.95] * sigma
    return {
        "records": int(len(residual)),
        "mean_abs_residual": float(abs_res.mean()) if len(residual) else np.nan,
        "sigma_mean": float(sigma.mean()) if len(residual) else np.nan,
        "nll": float(nll.mean()) if len(residual) else np.nan,
        "pinball": pinball_loss(residual, sigma) if len(residual) else np.nan,
        "coverage80": float(cov80.mean()) if len(residual) else np.nan,
        "coverage90": float(cov90.mean()) if len(residual) else np.nan,
        "coverage80_error": float(abs(cov80.mean() - 0.80)) if len(residual) else np.nan,
        "coverage90_error": float(abs(cov90.mean() - 0.90)) if len(residual) else np.nan,
        "width80": float(2.0 * Z_VALUES[0.90] * sigma.mean()) if len(residual) else np.nan,
        "width90": float(2.0 * Z_VALUES[0.95] * sigma.mean()) if len(residual) else np.nan,
    }


def summarize_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for (factor, value, group, model), df in rows.groupby(["factor", "value", "group", "model"]):
        weights = df["records"].to_numpy(dtype=np.float64)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(df)) / max(len(df), 1)
        item = {
            "factor": factor,
            "value": value,
            "group": group,
            "model": model,
            "counties": int(df["county"].nunique()),
            "records": int(df["records"].sum()),
        }
        for name in (
            "mean_abs_residual",
            "sigma_mean",
            "nll",
            "pinball",
            "coverage80",
            "coverage90",
            "coverage80_error",
            "coverage90_error",
            "width80",
            "width90",
        ):
            values = df[name].to_numpy(dtype=np.float64)
            valid = np.isfinite(values)
            item[name] = float(np.sum(values[valid] * weights[valid]) / weights[valid].sum()) if valid.any() else np.nan
        summary.append(item)
    return pd.DataFrame(summary).sort_values(["factor", "value", "group", "model"])


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    counties = list(args.counties)

    train_x_parts, train_res_parts = [], []
    payloads = {}
    metadata = {"mode": "STID-fixed probabilistic calibration pilot", "args": vars(args), "counties": {}}

    for county in counties:
        print(f"[build-train] {county}", flush=True)
        data_dir = data_root / f"TraffiDent_{county}_2023Q1"
        data = np.load(data_dir / "data.npz")["data"]
        index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
        shape = (len(index), args.output_len, data.shape[1], 1)
        result_dir = find_stid_result_dir(repo_root, county)
        pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
        target = load_raw_or_npy(result_dir / "targets.npy", shape)
        pred_flow = np.asarray(pred[:, :, :, 0], dtype=np.float32)
        target_flow = np.asarray(target[:, :, :, 0], dtype=np.float32)
        residual = target_flow - pred_flow
        valid = null_mask(target_flow, args.null_val)
        event = data[:, :, args.event_channel] > 0
        masks = event_masks(event, index)
        split = len(index) // 2
        train_rows = np.arange(0, split, dtype=np.int64)
        eval_rows = np.arange(split, len(index), dtype=np.int64)
        train_event = ~masks["no_event"][train_rows]
        event_r, event_n, event_h = sample_window_records(
            train_event,
            valid[train_rows],
            args.max_train_records_per_county // 2,
            rng,
        )
        no_event_r, no_event_n, no_event_h = sample_window_records(
            masks["no_event"][train_rows],
            valid[train_rows],
            args.max_train_records_per_county // 2,
            rng,
        )
        local_r = np.concatenate([event_r, no_event_r])
        nodes = np.concatenate([event_n, no_event_n])
        h0 = np.concatenate([event_h, no_event_h])
        rows_global = train_rows[local_r]

        incidents, meta = load_incidents_and_meta(data_dir)
        incident_field, incident_names = build_incident_field(train_rows, index, data.shape[1], incidents, meta, args)
        flow_stats = make_flow_stats(data[:, :, 0].astype(np.float32), index)
        x, names = build_features(
            county,
            counties,
            rows_global,
            nodes,
            h0,
            train_rows,
            index,
            flow_stats,
            incident_field,
            incident_names,
            args,
        )
        train_x_parts.append(x)
        train_res_parts.append(residual[rows_global, h0, nodes].astype(np.float64))
        payloads[county] = {
            "data_dir": data_dir,
            "data": data,
            "index": index,
            "pred": pred_flow,
            "target": target_flow,
            "residual": residual,
            "valid": valid,
            "masks": masks,
            "eval_rows": eval_rows,
            "incidents": incidents,
            "meta": meta,
            "result_dir": result_dir,
        }
        metadata["counties"][county] = {
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "train_records": int(len(rows_global)),
            "result_dir": str(result_dir.relative_to(repo_root)),
        }

    x_train = np.concatenate(train_x_parts, axis=0)
    residual_train = np.concatenate(train_res_parts, axis=0)
    print(f"[fit] train_records={len(residual_train)}", flush=True)

    models = {}
    for model_name in ("constant_sigma", "traffic_time_sigma", "incident_field_sigma", "full_sigma"):
        cols = model_columns(names, model_name)
        x_part = x_train[:, cols]
        mean, std = standardize_fit(x_part)
        coef = fit_log_sigma(
            standardize_apply(x_part, mean, std),
            residual_train,
            args.sigma_l2,
            args.sigma_lr,
            args.sigma_epochs,
        )
        models[model_name] = {"columns": cols, "mean": mean, "std": std, "coef": coef}

    rows_out: List[Dict[str, object]] = []
    for county, payload in payloads.items():
        print(f"[eval] {county}", flush=True)
        data = payload["data"]
        index = payload["index"]
        eval_rows = payload["eval_rows"]
        valid = payload["valid"][eval_rows]
        masks = payload["masks"]
        residual = payload["residual"]
        incident_field, incident_names = build_incident_field(
            eval_rows,
            index,
            data.shape[1],
            payload["incidents"],
            payload["meta"],
            args,
        )
        flow_stats = make_flow_stats(data[:, :, 0].astype(np.float32), index)

        no_event_r, no_event_n, no_event_h = sample_window_records(
            masks["no_event"][eval_rows],
            valid,
            args.max_eval_control_records_per_county,
            rng,
        )
        event_r, event_n, event_h = sample_window_records(
            ~masks["no_event"][eval_rows],
            valid,
            args.max_eval_event_records_per_county,
            rng,
        )
        positions = [
            ("all", "all", "no_event_sample", no_event_r, no_event_n, no_event_h),
            ("all", "all", "all_eval_sample", np.concatenate([no_event_r, event_r]), np.concatenate([no_event_n, event_n]), np.concatenate([no_event_h, event_h])),
        ]
        for group in GROUPS[2:]:
            r, n, h = sample_window_records(masks[group][eval_rows], valid, 0, rng)
            positions.append(("all", "all", group, r, n, h))

        type_masks = load_type_event_masks(
            data_dir=payload["data_dir"],
            num_steps=data.shape[0],
            num_nodes=data.shape[1],
            event_window_slots=args.type_event_window_slots,
        )
        for type_value, type_event in sorted(type_masks.items()):
            type_scope = event_masks(type_event, index)
            for group in ("future_any", "future_onset", "ongoing", "post_last_slot"):
                r, n, h = sample_window_records(type_scope[group][eval_rows], valid, 0, rng)
                positions.append(("type", type_value, group, r, n, h))

        for factor, value, group, local_r, nodes, h0 in positions:
            if len(local_r) == 0:
                continue
            rows_global = eval_rows[local_r]
            x_eval, _ = build_features(
                county,
                counties,
                rows_global,
                nodes,
                h0,
                eval_rows,
                index,
                flow_stats,
                incident_field,
                incident_names,
                args,
            )
            res_eval = residual[rows_global, h0, nodes].astype(np.float64)
            for model_name, model in models.items():
                cols = model["columns"]
                sigma = predict_sigma(standardize_apply(x_eval[:, cols], model["mean"], model["std"]), model["coef"])
                row = {
                    "county": county,
                    "factor": factor,
                    "value": value,
                    "group": group,
                    "model": model_name,
                }
                row.update(gaussian_metrics(res_eval, sigma))
                rows_out.append(row)

    metrics = pd.DataFrame(rows_out)
    summary = summarize_metrics(metrics)
    metrics_path = output_dir / "probabilistic_calibration_metrics.csv"
    summary_path = output_dir / "probabilistic_calibration_summary.csv"
    metadata_path = output_dir / "probabilistic_calibration_metadata.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata["files"] = {
        "metrics_csv": str(metrics_path),
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(json.dumps(metadata["files"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
