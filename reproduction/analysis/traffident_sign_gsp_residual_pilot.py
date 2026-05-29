#!/usr/bin/env python3
"""Loss-aware sign residual and graph-signal audit over saved STID results.

This is a post-hoc pilot for the G3TRC direction. It keeps STID frozen and
tests whether incident candidate records contain a reliable residual direction
signal before building a trainable BasicTS module.
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

from reproduction.analysis.traffident_decay_kernel_pilot import (
    COUNTIES,
    HORIZONS,
    build_design,
    event_masks,
    find_stid_result_dir,
    load_incidents_and_meta,
    load_raw_or_npy,
    metric,
    null_mask,
    weighted_ridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument(
        "--output-dir",
        default="reproduction/analysis/traffident_sign_gsp_residual_pilot",
    )
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--event-channel", type=int, default=3)
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--lambda-space", type=float, default=1.0)
    parser.add_argument("--lambda-time-post", type=float, default=6.0)
    parser.add_argument("--lambda-time-pre", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--max-pre-slots", type=int, default=6)
    parser.add_argument("--max-post-slots", type=int, default=12)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--sign-alpha", type=float, default=1000.0)
    parser.add_argument("--sign-margin", type=float, default=1.0)
    parser.add_argument("--max-correction", type=float, default=5.0)
    parser.add_argument("--same-direction-only", action="store_true")
    parser.add_argument(
        "--scale-grid",
        default="0,0.1,0.25,0.5,1.0",
        help="comma-separated correction magnitude scales tuned on calibration validation",
    )
    return parser.parse_args()


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.average(values, weights=weights))


def fit_county_magnitudes(
    y: np.ndarray,
    w: np.ndarray,
    county_ids: np.ndarray,
    counties: List[str],
    margin: float,
    max_correction: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    global_pos = weighted_mean(np.abs(y[y > margin]), w[y > margin]) if np.any(y > margin) else 0.0
    global_neg = weighted_mean(np.abs(y[y < -margin]), w[y < -margin]) if np.any(y < -margin) else 0.0
    for idx, county in enumerate(counties):
        mask = county_ids == idx
        pos_mask = mask & (y > margin)
        neg_mask = mask & (y < -margin)
        pos = weighted_mean(np.abs(y[pos_mask]), w[pos_mask]) if np.any(pos_mask) else global_pos
        neg = weighted_mean(np.abs(y[neg_mask]), w[neg_mask]) if np.any(neg_mask) else global_neg
        out[county] = {
            "rise": float(np.clip(pos, 0.0, max_correction)),
            "drop": float(np.clip(neg, 0.0, max_correction)),
        }
    return out


def correction_from_scores(
    scores: np.ndarray,
    county_ids: np.ndarray,
    counties: List[str],
    magnitudes: Dict[str, Dict[str, float]],
    threshold: float,
    scale: float,
) -> np.ndarray:
    corr = np.zeros(len(scores), dtype=np.float32)
    active_pos = scores > threshold
    active_neg = scores < -threshold
    for idx, county in enumerate(counties):
        county_mask = county_ids == idx
        corr[county_mask & active_pos] = scale * magnitudes[county]["rise"]
        corr[county_mask & active_neg] = -scale * magnitudes[county]["drop"]
    return corr


def choose_threshold_and_scale(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    county_val: np.ndarray,
    counties: List[str],
    magnitudes: Dict[str, Dict[str, float]],
    scale_grid: Iterable[float],
) -> Dict[str, float]:
    abs_scores = np.abs(scores_val)
    candidates = [0.0]
    if len(abs_scores):
        candidates.extend(float(x) for x in np.quantile(abs_scores, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]))
    best = {"threshold": 0.0, "scale": 0.0, "mae": float(np.mean(np.abs(y_val)))}
    for threshold in sorted(set(candidates)):
        for scale in scale_grid:
            corr = correction_from_scores(
                scores=scores_val,
                county_ids=county_val,
                counties=counties,
                magnitudes=magnitudes,
                threshold=threshold,
                scale=scale,
            )
            mae = float(np.mean(np.abs(y_val - corr)))
            if mae < best["mae"]:
                best = {"threshold": float(threshold), "scale": float(scale), "mae": mae}
    return best


def build_neighbor_edges(meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    edges = []
    for _, group in meta.groupby(["Fwy", "Direction"], dropna=True):
        group = group.sort_values("Abs PM")
        nodes = group["node_idx"].to_numpy(dtype=np.int64)
        if len(nodes) < 2:
            continue
        edges.extend((int(a), int(b)) for a, b in zip(nodes[:-1], nodes[1:]))
    if not edges:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src, dst = np.asarray(edges, dtype=np.int64).T
    return src, dst


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12:
        return np.nan
    return float(np.sum(a * b) / denom)


def residual_structure_audit(
    county: str,
    pred_eval: np.ndarray,
    target_eval: np.ndarray,
    group_masks: Dict[str, np.ndarray],
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    null_val: float,
    margin: float,
) -> List[Dict[str, object]]:
    residual = target_eval - pred_eval
    valid_base = null_mask(target_eval, null_val)
    rows = []
    for group, mask_2d in group_masks.items():
        valid = mask_2d[:, None, :] & valid_base
        count = int(valid.sum())
        if count == 0:
            continue
        values = residual[valid]
        pos = float(np.mean(values > margin))
        neg = float(np.mean(values < -margin))
        zero = float(np.mean(np.abs(values) <= margin))
        if edge_src.size:
            edge_valid = valid[:, :, edge_src] & valid[:, :, edge_dst]
            left = residual[:, :, edge_src][edge_valid]
            right = residual[:, :, edge_dst][edge_valid]
            graph_corr = safe_corr(left, right)
            graph_tv = float(np.mean(np.abs(left - right))) if len(left) else np.nan
        else:
            graph_corr = np.nan
            graph_tv = np.nan
        t_valid = valid[:, :-1, :] & valid[:, 1:, :]
        t_left = residual[:, :-1, :][t_valid]
        t_right = residual[:, 1:, :][t_valid]
        rows.append(
            {
                "county": county,
                "group": group,
                "valid_values": count,
                "mean_residual": float(values.mean()),
                "mean_abs_residual": float(np.abs(values).mean()),
                "rise_ratio": pos,
                "drop_ratio": neg,
                "small_ratio": zero,
                "graph_autocorr": graph_corr,
                "graph_tv": graph_tv,
                "temporal_autocorr": safe_corr(t_left, t_right),
            }
        )
    return rows


def group_masks_for_eval(masks: Dict[str, np.ndarray], kernel_mask: np.ndarray, eval_rows: np.ndarray) -> Dict[str, np.ndarray]:
    kernel_eval = kernel_mask[eval_rows]
    return {
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


def evaluate_methods(
    county: str,
    pred_eval: np.ndarray,
    target_eval: np.ndarray,
    corrections: Dict[str, np.ndarray],
    group_masks: Dict[str, np.ndarray],
    null_val: float,
) -> List[Dict[str, object]]:
    rows = []
    base_metrics = {group: metric(pred_eval, target_eval, mask, null_val) for group, mask in group_masks.items()}
    for group, mask in group_masks.items():
        base = base_metrics[group]
        row = {
            "county": county,
            "group": group,
            "node_windows": int(mask.sum()),
            "STID_MAE": base["MAE"],
            "STID_RMSE": base["RMSE"],
            "STID_bias": base["bias"],
        }
        for name, corr in corrections.items():
            metrics = metric(pred_eval + corr, target_eval, mask, null_val)
            row[f"{name}_MAE"] = metrics["MAE"]
            row[f"{name}_RMSE"] = metrics["RMSE"]
            row[f"{name}_bias"] = metrics["bias"]
            row[f"{name}_delta_vs_STID_MAE"] = metrics["MAE"] - base["MAE"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counties = list(args.counties)
    scale_grid = [float(x) for x in args.scale_grid.split(",") if x.strip()]

    county_cache = {}
    train_x, train_y, train_w, train_county = [], [], [], []
    val_x, val_y, val_w, val_county = [], [], [], []
    all_x, all_y, all_w = [], [], []
    metadata = {
        "mode": "post-hoc G3TRC sign/GSP pilot over saved STID test_results",
        "counties": {},
        "args": vars(args),
    }

    for cidx, county in enumerate(counties):
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
        x, y, w, locations, kernel_mask_calib = build_design(
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
        if len(x) == 0:
            continue
        row_values = np.asarray([loc[0] for loc in locations], dtype=np.int64)
        val_boundary = int(split * 0.7)
        train_mask = row_values < val_boundary
        val_mask = ~train_mask
        if not np.any(val_mask):
            val_mask = np.ones(len(x), dtype=bool)
            train_mask = np.ones(len(x), dtype=bool)
        train_x.append(x[train_mask])
        train_y.append(y[train_mask])
        train_w.append(w[train_mask])
        train_county.append(np.full(int(train_mask.sum()), cidx, dtype=np.int64))
        val_x.append(x[val_mask])
        val_y.append(y[val_mask])
        val_w.append(w[val_mask])
        val_county.append(np.full(int(val_mask.sum()), cidx, dtype=np.int64))
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
        }
        metadata["counties"][county] = {
            "result_dir": str(result_dir.relative_to(repo_root)),
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "calibration_records": int(len(x)),
            "sign_train_records": int(train_mask.sum()),
            "sign_val_records": int(val_mask.sum()),
        }

    x_train = np.concatenate(train_x, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    w_train = np.concatenate(train_w, axis=0)
    c_train = np.concatenate(train_county, axis=0)
    x_val = np.concatenate(val_x, axis=0)
    y_val = np.concatenate(val_y, axis=0)
    c_val = np.concatenate(val_county, axis=0)
    x_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    w_all = np.concatenate(all_w, axis=0)

    print(f"[fit] sign_records={len(x_train)} residual_records={len(x_all)}", flush=True)
    x_mean, x_std = standardize_fit(x_train)
    sign_target = np.zeros_like(y_train)
    sign_target[y_train > args.sign_margin] = 1.0
    sign_target[y_train < -args.sign_margin] = -1.0
    sign_coef = weighted_ridge(
        standardize_apply(x_train, x_mean, x_std),
        sign_target,
        w_train,
        args.sign_alpha,
    )
    decay_coef = weighted_ridge(x_all, y_all, w_all, args.ridge_alpha)
    bias_by_county = {
        county: weighted_mean(y_all[np.argmax(x_all[:, 1 : 1 + len(counties)], axis=1) == idx], w_all[np.argmax(x_all[:, 1 : 1 + len(counties)], axis=1) == idx])
        for idx, county in enumerate(counties)
    }
    magnitudes = fit_county_magnitudes(
        y=y_train,
        w=w_train,
        county_ids=c_train,
        counties=counties,
        margin=args.sign_margin,
        max_correction=args.max_correction,
    )
    val_scores = standardize_apply(x_val, x_mean, x_std) @ sign_coef
    gate = choose_threshold_and_scale(
        y_val=y_val,
        scores_val=val_scores,
        county_val=c_val,
        counties=counties,
        magnitudes=magnitudes,
        scale_grid=scale_grid,
    )
    metadata["sign_gate"] = gate
    metadata["magnitudes"] = magnitudes
    metadata["bias_by_county"] = bias_by_county
    np.save(output_dir / "sign_ridge_coef.npy", sign_coef)
    np.save(output_dir / "decay_kernel_ridge_coef.npy", decay_coef)
    np.savez(output_dir / "feature_standardizer.npz", mean=x_mean, std=x_std)

    metric_rows = []
    audit_rows = []
    for cidx, county in enumerate(counties):
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
        pred_eval = np.asarray(cache["pred"][eval_rows, :, :, 0], dtype=np.float32)
        target_eval = np.asarray(cache["target"][eval_rows, :, :, 0], dtype=np.float32)
        shape = (len(eval_rows), args.output_len, cache["data"].shape[1])
        corrections = {
            "BiasOnly": np.zeros(shape, dtype=np.float32),
            "DecayKernel": np.zeros(shape, dtype=np.float32),
            "SignReliability": np.zeros(shape, dtype=np.float32),
        }
        row_to_eval = {int(row): idx for idx, row in enumerate(eval_rows)}
        if len(x_eval):
            decay_values = np.clip(x_eval @ decay_coef, -args.max_correction, args.max_correction)
            sign_scores = standardize_apply(x_eval, x_mean, x_std) @ sign_coef
            sign_values = correction_from_scores(
                scores=sign_scores,
                county_ids=np.full(len(x_eval), cidx, dtype=np.int64),
                counties=counties,
                magnitudes=magnitudes,
                threshold=gate["threshold"],
                scale=gate["scale"],
            )
            for idx, (row, node, horizon) in enumerate(locations):
                erow = row_to_eval[int(row)]
                corrections["BiasOnly"][erow, horizon, int(node)] = float(bias_by_county[county])
                corrections["DecayKernel"][erow, horizon, int(node)] = float(decay_values[idx])
                corrections["SignReliability"][erow, horizon, int(node)] = float(sign_values[idx])
        group_masks = group_masks_for_eval(cache["masks"], kernel_mask_eval, eval_rows)
        metric_rows.extend(
            evaluate_methods(
                county=county,
                pred_eval=pred_eval,
                target_eval=target_eval,
                corrections=corrections,
                group_masks=group_masks,
                null_val=args.null_val,
            )
        )
        src, dst = build_neighbor_edges(cache["meta"])
        audit_rows.extend(
            residual_structure_audit(
                county=county,
                pred_eval=pred_eval,
                target_eval=target_eval,
                group_masks=group_masks,
                edge_src=src,
                edge_dst=dst,
                null_val=args.null_val,
                margin=args.sign_margin,
            )
        )
        metadata["counties"][county]["evaluation_records"] = int(len(x_eval))
        metadata["counties"][county]["edge_count"] = int(len(src))

    metrics_df = pd.DataFrame(metric_rows)
    audit_df = pd.DataFrame(audit_rows)
    metrics_path = output_dir / "sign_gsp_metrics.csv"
    audit_path = output_dir / "residual_structure_audit.csv"
    summary_path = output_dir / "sign_gsp_summary.csv"
    metadata_path = output_dir / "sign_gsp_metadata.json"
    metrics_df.to_csv(metrics_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    summary_rows = []
    method_names = ["BiasOnly", "DecayKernel", "SignReliability"]
    for group, group_df in metrics_df.groupby("group"):
        row = {
            "group": group,
            "counties": int(group_df["county"].nunique()),
            "total_node_windows": int(group_df["node_windows"].sum()),
            "mean_STID_MAE": float(group_df["STID_MAE"].mean()),
        }
        for method in method_names:
            delta = group_df[f"{method}_delta_vs_STID_MAE"].dropna()
            row[f"mean_delta_{method}_vs_STID_MAE"] = float(delta.mean()) if len(delta) else np.nan
            row[f"{method}_wins_vs_STID"] = int((delta < 0).sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values("group").to_csv(summary_path, index=False)

    metadata["files"] = {
        "metrics_csv": str(metrics_path),
        "summary_csv": str(summary_path),
        "audit_csv": str(audit_path),
        "sign_coef_npy": str(output_dir / "sign_ridge_coef.npy"),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(json.dumps(metadata["files"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
