#!/usr/bin/env python3
"""Type-conditioned incident risk pilot over saved TraffiDent STID results.

This is a post-hoc analysis script, not a BasicTS training entrypoint. It uses
saved pure-STID test_results to build a tail90 label and asks whether incident
type / spatial / temporal fields can predict when STID enters its no-event
high-error tail.

Two incident scopes are supported:

- history: deployable-style, only incidents visible at the forecast origin.
- history_future: oracle-style, incidents in the future horizon may contribute.
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
    iter_affected_nodes,
    load_incidents_and_meta,
    load_raw_or_npy,
    null_mask,
    relation_bucket,
    spatial_kernel,
    temporal_kernel,
    time_bucket,
    type_bucket,
)
from reproduction.analysis.traffident_matched_control_residual_audit import (  # noqa: E402
    load_type_event_masks,
    time_keys,
)


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
        default="reproduction/analysis/traffident_type_risk_pilot",
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
    parser.add_argument("--max-train-records-per-county", type=int, default=200_000)
    parser.add_argument("--max-eval-control-records-per-county", type=int, default=120_000)
    parser.add_argument("--max-eval-all-records-per-county", type=int, default=220_000)
    parser.add_argument("--logistic-l2", type=float, default=1e-3)
    parser.add_argument("--logistic-lr", type=float, default=0.05)
    parser.add_argument("--logistic-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def feature_names(counties: Iterable[str]) -> List[str]:
    names = ["bias"]
    names += [f"county={county}" for county in counties]
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
    names += ["kernel_sum", "kernel_sq_sum", "kernel_max"]
    names += [f"type={value}" for value in TYPE_VALUES]
    names += [f"relation={value}" for value in RELATION_VALUES]
    names += [f"time={value}" for value in TIME_BUCKETS]
    return names


def incident_feature_names() -> List[str]:
    names = ["kernel_sum", "kernel_sq_sum", "kernel_max"]
    names += [f"type={value}" for value in TYPE_VALUES]
    names += [f"relation={value}" for value in RELATION_VALUES]
    names += [f"time={value}" for value in TIME_BUCKETS]
    return names


def make_flow_stats(flow: np.ndarray, index: np.ndarray) -> Dict[str, np.ndarray]:
    prefix = np.zeros((flow.shape[0] + 1, flow.shape[1]), dtype=np.float64)
    prefix2 = np.zeros_like(prefix)
    prefix[1:] = np.cumsum(flow.astype(np.float64), axis=0)
    prefix2[1:] = np.cumsum((flow.astype(np.float64) ** 2), axis=0)
    start = index[:, 0].astype(np.int64)
    mid = index[:, 1].astype(np.int64)
    length = np.maximum(mid - start, 1)[:, None]
    mean = (prefix[mid] - prefix[start]) / length
    mean2 = (prefix2[mid] - prefix2[start]) / length
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    last = flow[mid - 1].astype(np.float64)
    first = flow[start].astype(np.float64)
    trend = (last - first) / np.maximum((mid - start - 1), 1)[:, None]
    return {
        "hist_last": last,
        "hist_mean": mean,
        "hist_std": std,
        "hist_trend": trend,
    }


def error_mean(pred: np.ndarray, target: np.ndarray, null_val: float) -> np.ndarray:
    pred_eval = np.asarray(pred[:, :, :, 0], dtype=np.float32)
    target_eval = np.asarray(target[:, :, :, 0], dtype=np.float32)
    valid = null_mask(target_eval, null_val)
    abs_error = np.abs(pred_eval - target_eval)
    valid_count = np.maximum(valid.sum(axis=1), 1)
    return (abs_error * valid).sum(axis=1) / valid_count


def add_incident_feature(
    field: np.ndarray,
    row_local: int,
    node: int,
    k: float,
    type_value: str,
    relation_value: str,
    delta_slots: int,
    idx: Dict[str, int],
) -> None:
    field[row_local, node, idx["kernel_sum"]] += k
    field[row_local, node, idx["kernel_sq_sum"]] += k * k
    field[row_local, node, idx["kernel_max"]] = max(field[row_local, node, idx["kernel_max"]], k)
    field[row_local, node, idx[f"type={type_value}"]] += k
    field[row_local, node, idx[f"relation={relation_value}"]] += k
    field[row_local, node, idx[f"time={time_bucket(delta_slots)}"]] += k


def build_incident_field(
    rows: np.ndarray,
    index: np.ndarray,
    num_nodes: int,
    incidents: pd.DataFrame,
    meta: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[str]]:
    names = incident_feature_names()
    idx = {name: i for i, name in enumerate(names)}
    field = np.zeros((len(rows), num_nodes, len(names)), dtype=np.float32)
    row_lookup = {int(row): local for local, row in enumerate(rows)}
    mid_to_rows: Dict[int, List[int]] = {}
    for row in rows:
        mid_to_rows.setdefault(int(index[row, 1]), []).append(int(row))
    if not mid_to_rows:
        return field, names
    min_mid = min(mid_to_rows)
    max_mid = max(mid_to_rows)

    for incident in incidents.itertuples(index=False):
        event_slot = int(incident.event_slot)
        if event_slot < 0:
            continue
        affected = list(iter_affected_nodes(meta, incident, args))
        if not affected:
            continue
        type_value = type_bucket(incident.Type)

        if args.incident_scope == "history":
            if event_slot > max_mid - 1 or event_slot < min_mid - args.max_post_slots:
                continue
            for mid in range(event_slot + 1, event_slot + args.max_post_slots + 2):
                for row in mid_to_rows.get(mid, []):
                    delta = int(mid - 1 - event_slot)
                    kt = temporal_kernel(delta, args)
                    row_local = row_lookup[row]
                    for node, signed_pm, relation in affected:
                        k = spatial_kernel(signed_pm, args) * kt
                        if k > 1e-8:
                            add_incident_feature(field, row_local, node, k, type_value, relation, delta, idx)
            continue

        if event_slot - args.max_pre_slots - (args.output_len - 1) > max_mid:
            continue
        if event_slot + args.max_post_slots < min_mid:
            continue
        for h0 in range(args.output_len):
            lo_mid = event_slot - args.max_pre_slots - h0
            hi_mid = event_slot + args.max_post_slots - h0
            for mid in range(lo_mid, hi_mid + 1):
                for row in mid_to_rows.get(mid, []):
                    delta = int(mid + h0 - event_slot)
                    kt = temporal_kernel(delta, args)
                    row_local = row_lookup[row]
                    for node, signed_pm, relation in affected:
                        k = spatial_kernel(signed_pm, args) * kt
                        if k > 1e-8:
                            add_incident_feature(field, row_local, node, k, type_value, relation, delta, idx)

    return field, names


def assemble_features(
    county: str,
    counties: List[str],
    rows: np.ndarray,
    nodes: np.ndarray,
    all_rows: np.ndarray,
    index: np.ndarray,
    flow_stats: Dict[str, np.ndarray],
    incident_field: np.ndarray,
    incident_names: List[str],
    steps_per_day: int,
    start_weekday: int,
) -> Tuple[np.ndarray, List[str]]:
    names = feature_names(counties)
    x = np.zeros((len(rows), len(names)), dtype=np.float32)
    col = {name: i for i, name in enumerate(names)}
    x[:, col["bias"]] = 1.0
    x[:, col[f"county={county}"]] = 1.0

    row_lookup = {int(row): local for local, row in enumerate(all_rows)}
    local_rows = np.asarray([row_lookup[int(row)] for row in rows], dtype=np.int64)
    for name in ("hist_last", "hist_mean", "hist_std", "hist_trend"):
        x[:, col[name]] = flow_stats[name][rows, nodes].astype(np.float32)

    keys = time_keys(index[rows, 1], steps_per_day, start_weekday)
    tod = (keys % steps_per_day).astype(np.float32)
    dow = (keys // steps_per_day).astype(np.float32)
    x[:, col["tod_sin"]] = np.sin(2.0 * np.pi * tod / float(steps_per_day))
    x[:, col["tod_cos"]] = np.cos(2.0 * np.pi * tod / float(steps_per_day))
    x[:, col["dow_sin"]] = np.sin(2.0 * np.pi * dow / 7.0)
    x[:, col["dow_cos"]] = np.cos(2.0 * np.pi * dow / 7.0)

    field_values = incident_field[local_rows, nodes]
    for src_idx, name in enumerate(incident_names):
        x[:, col[name]] = field_values[:, src_idx]
    return x, names


def sample_flat(mask: np.ndarray, max_count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.flatnonzero(mask.reshape(-1))
    if max_count > 0 and len(flat) > max_count:
        flat = rng.choice(flat, size=max_count, replace=False)
    rows, nodes = np.unravel_index(flat, mask.shape)
    return rows.astype(np.int64), nodes.astype(np.int64)


def sample_training_positions(
    label: np.ndarray,
    event_any: np.ndarray,
    valid: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    event_rows, event_nodes = sample_flat(event_any & valid, max_count, rng)
    remaining = max(max_count - len(event_rows), max_count // 2)
    pos_rows, pos_nodes = sample_flat(label & valid, remaining // 2, rng)
    neg_rows, neg_nodes = sample_flat((~label) & valid & (~event_any), remaining, rng)
    rows = np.concatenate([event_rows, pos_rows, neg_rows])
    nodes = np.concatenate([event_nodes, pos_nodes, neg_nodes])
    if len(rows) == 0:
        return rows, nodes
    key = rows.astype(np.int64) * label.shape[1] + nodes.astype(np.int64)
    _, keep = np.unique(key, return_index=True)
    return rows[keep], nodes[keep]


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    lr: float,
    epochs: int,
) -> np.ndarray:
    coef = np.zeros(x.shape[1], dtype=np.float64)
    y = y.astype(np.float64)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weight = np.where(y > 0, 0.5 / pos, 0.5 / neg)
    for _ in range(epochs):
        pred = sigmoid(x @ coef)
        grad = x.T @ ((pred - y) * weight)
        reg = l2 * coef
        reg[0] = 0.0
        coef -= lr * (grad + reg)
    return coef


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum = float(ranks[y].sum())
    return (rank_sum - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)


def ap_score(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def ece_score(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    y = y.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if not mask.any():
            continue
        out += float(mask.mean()) * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return out


def summarize_scores(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = y.astype(np.float64)
    score = np.clip(score.astype(np.float64), 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        return {
            "records": 0,
            "positive_rate": np.nan,
            "mean_score": np.nan,
            "brier": np.nan,
            "auc": np.nan,
            "ap": np.nan,
            "ece": np.nan,
            "top10_lift": np.nan,
        }
    top_k = max(int(np.ceil(0.1 * len(y))), 1)
    top = np.argsort(-score)[:top_k]
    return {
        "records": int(len(y)),
        "positive_rate": float(y.mean()),
        "mean_score": float(score.mean()),
        "brier": float(((score - y) ** 2).mean()),
        "auc": float(auc_score(y, score)),
        "ap": float(ap_score(y, score)),
        "ece": float(ece_score(y, score)),
        "top10_lift": float(y[top].mean() - y.mean()),
    }


def type_only_columns(names: List[str]) -> List[int]:
    keep = ["bias"] + [name for name in names if name.startswith("county=")]
    keep += [name for name in names if name.startswith("type=")]
    return [names.index(name) for name in keep]


def evaluate_group(
    county: str,
    factor: str,
    value: str,
    group: str,
    rows_global: np.ndarray,
    nodes: np.ndarray,
    label_full: np.ndarray,
    x_full: np.ndarray,
    names: List[str],
    models: Dict[str, Dict[str, np.ndarray]],
    county_base_rate: float,
    global_base_rate: float,
) -> List[Dict[str, object]]:
    y = label_full[rows_global, nodes].astype(np.float32)
    output = []
    scores = {
        "global_base": np.full(len(y), global_base_rate, dtype=np.float64),
        "county_base": np.full(len(y), county_base_rate, dtype=np.float64),
    }
    for model_name, model in models.items():
        cols = model["columns"]
        x = standardize_apply(x_full[:, cols], model["mean"], model["std"])
        scores[model_name] = sigmoid(x @ model["coef"])
    for model_name, score in scores.items():
        row = {
            "county": county,
            "factor": factor,
            "value": value,
            "group": group,
            "model": model_name,
        }
        row.update(summarize_scores(y, score))
        output.append(row)
    return output


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
        for name in ("positive_rate", "mean_score", "brier", "auc", "ap", "ece", "top10_lift"):
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

    train_x_parts, train_y_parts = [], []
    county_payload = {}
    metadata = {"mode": "type-conditioned tail90 risk pilot over saved STID results", "args": vars(args), "counties": {}}

    for county in counties:
        print(f"[build-train] {county}", flush=True)
        data_dir = data_root / f"TraffiDent_{county}_2023Q1"
        data = np.load(data_dir / "data.npz")["data"]
        index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
        shape = (len(index), args.output_len, data.shape[1], 1)
        result_dir = find_stid_result_dir(repo_root, county)
        pred = load_raw_or_npy(result_dir / "predictions.npy", shape)
        target = load_raw_or_npy(result_dir / "targets.npy", shape)
        event = data[:, :, args.event_channel] > 0
        masks = event_masks(event, index)
        err = error_mean(pred, target, args.null_val)
        split = len(index) // 2
        train_rows = np.arange(0, split, dtype=np.int64)
        eval_rows = np.arange(split, len(index), dtype=np.int64)
        no_event_train = masks["no_event"][train_rows]
        threshold = float(np.quantile(err[train_rows][no_event_train], 0.90))
        label = err > threshold
        valid = np.isfinite(err)
        train_event = ~masks["no_event"][train_rows]
        local_rows, nodes = sample_training_positions(
            label[train_rows],
            train_event,
            valid[train_rows],
            args.max_train_records_per_county,
            rng,
        )
        rows = train_rows[local_rows]
        incidents, meta = load_incidents_and_meta(data_dir)
        field, incident_names = build_incident_field(train_rows, index, data.shape[1], incidents, meta, args)
        flow_stats = make_flow_stats(data[:, :, 0].astype(np.float32), index)
        x, names = assemble_features(
            county,
            counties,
            rows,
            nodes,
            train_rows,
            index,
            flow_stats,
            field,
            incident_names,
            args.steps_per_day,
            args.start_weekday,
        )
        train_x_parts.append(x)
        train_y_parts.append(label[rows, nodes].astype(np.float32))
        county_payload[county] = {
            "data_dir": data_dir,
            "data": data,
            "index": index,
            "pred": pred,
            "target": target,
            "masks": masks,
            "err": err,
            "label": label,
            "valid": valid,
            "eval_rows": eval_rows,
            "incidents": incidents,
            "meta": meta,
            "result_dir": result_dir,
            "tail90_threshold": threshold,
            "train_records": int(len(rows)),
            "train_positive_rate": float(label[rows, nodes].mean()) if len(rows) else np.nan,
        }
        metadata["counties"][county] = {
            "data_shape": list(data.shape),
            "test_index_shape": list(index.shape),
            "tail90_threshold": threshold,
            "train_records": int(len(rows)),
            "train_positive_rate": county_payload[county]["train_positive_rate"],
            "result_dir": str(result_dir.relative_to(repo_root)),
        }

    x_train = np.concatenate(train_x_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)
    global_base_rate = float(y_train.mean())
    print(f"[fit] train_records={len(y_train)} positive_rate={global_base_rate:.4f}", flush=True)

    full_columns = np.arange(x_train.shape[1], dtype=np.int64)
    type_columns = np.asarray(type_only_columns(names), dtype=np.int64)
    models = {}
    for model_name, columns in {
        "type_only": type_columns,
        "v3_type_risk": full_columns,
    }.items():
        x_part = x_train[:, columns]
        mean, std = standardize_fit(x_part)
        coef = fit_logistic(
            standardize_apply(x_part, mean, std),
            y_train,
            args.logistic_l2,
            args.logistic_lr,
            args.logistic_epochs,
        )
        models[model_name] = {"columns": columns, "mean": mean, "std": std, "coef": coef}

    rows_out: List[Dict[str, object]] = []
    for county, payload in county_payload.items():
        print(f"[eval] {county}", flush=True)
        data = payload["data"]
        index = payload["index"]
        eval_rows = payload["eval_rows"]
        masks = payload["masks"]
        label = payload["label"]
        valid = payload["valid"]
        incidents = payload["incidents"]
        meta = payload["meta"]
        flow_stats = make_flow_stats(data[:, :, 0].astype(np.float32), index)
        field, incident_names = build_incident_field(eval_rows, index, data.shape[1], incidents, meta, args)
        no_event_eval = masks["no_event"][eval_rows] & valid[eval_rows]
        no_event_rows, no_event_nodes = sample_flat(no_event_eval, args.max_eval_control_records_per_county, rng)
        event_eval = (~masks["no_event"][eval_rows]) & valid[eval_rows]
        event_rows, event_nodes = sample_flat(event_eval, args.max_eval_all_records_per_county, rng)
        all_rows = np.concatenate([no_event_rows, event_rows])
        all_nodes = np.concatenate([no_event_nodes, event_nodes])
        if len(all_rows):
            key = all_rows * data.shape[1] + all_nodes
            _, keep = np.unique(key, return_index=True)
            all_rows, all_nodes = all_rows[keep], all_nodes[keep]

        group_positions: List[Tuple[str, str, str, np.ndarray, np.ndarray]] = [
            ("all", "all", "all_eval_sample", all_rows, all_nodes),
            ("all", "all", "no_event_sample", no_event_rows, no_event_nodes),
        ]
        for group in GROUPS[2:]:
            local_r, n = sample_flat(masks[group][eval_rows] & valid[eval_rows], 0, rng)
            group_positions.append(("all", "all", group, local_r, n))

        type_masks = load_type_event_masks(
            data_dir=payload["data_dir"],
            num_steps=data.shape[0],
            num_nodes=data.shape[1],
            event_window_slots=args.type_event_window_slots,
        )
        for type_value, type_event in sorted(type_masks.items()):
            type_scope = event_masks(type_event, index)
            for group in ("future_any", "future_onset", "ongoing", "post_last_slot"):
                local_r, n = sample_flat(type_scope[group][eval_rows] & valid[eval_rows], 0, rng)
                group_positions.append(("type", type_value, group, local_r, n))

        county_base_rate = float(payload["train_positive_rate"])
        for factor, value, group, local_r, nodes in group_positions:
            if len(local_r) == 0:
                continue
            rows_global = eval_rows[local_r]
            x_eval, _ = assemble_features(
                county,
                counties,
                rows_global,
                nodes,
                eval_rows,
                index,
                flow_stats,
                field,
                incident_names,
                args.steps_per_day,
                args.start_weekday,
            )
            rows_out.extend(
                evaluate_group(
                    county,
                    factor,
                    value,
                    group,
                    rows_global,
                    nodes,
                    label,
                    x_eval,
                    names,
                    models,
                    county_base_rate,
                    global_base_rate,
                )
            )

    metrics = pd.DataFrame(rows_out)
    summary = summarize_metrics(metrics)
    metrics_path = output_dir / "type_risk_pilot_metrics.csv"
    summary_path = output_dir / "type_risk_pilot_summary.csv"
    metadata_path = output_dir / "type_risk_pilot_metadata.json"
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
