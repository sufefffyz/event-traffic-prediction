#!/usr/bin/env python3
"""Recompute TraffiDent post-incident forecasting table metrics.

The paper's Appendix A.8 defines a post-incident case as follows: match each
incident to the nearest sensor on the same freeway by absolute postmile, map
the incident timestamp to a 5-minute slot, and use the next slot as the
forecasting origin. With 1 hour of history and 30 minutes of future, horizons
t=1/3/6 correspond to 5/15/30 minutes after that origin.
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


DEFAULT_HORIZONS = (1, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument("--dataset", default="TraffiDent_D5_2023Q1")
    parser.add_argument("--models", nargs="+", default=["AGCRN"])
    parser.add_argument(
        "--result-glob",
        default="BasicTS/checkpoints/{model}/{dataset}_*_12_12*/**/test_results",
        help="Glob relative to repo root. It may contain {model} and {dataset}.",
    )
    parser.add_argument("--output-dir", default="reproduction/analysis/traffident_post_incident_table")
    parser.add_argument("--start-time", default=None, help="Optional override, e.g. 2023-01-01 00:00:00")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument(
        "--incident-origin-offset",
        type=int,
        default=1,
        help="Appendix A.8 uses the next 5-minute slot after the incident slot as origin.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def parse_horizons(value: str) -> Tuple[int, ...]:
    horizons = tuple(int(part) for part in value.split(",") if part.strip())
    if not horizons:
        raise ValueError("At least one horizon is required")
    return horizons


def find_result_dir(repo_root: Path, model: str, dataset: str, pattern: str) -> Path:
    expanded = pattern.format(model=model, dataset=dataset)
    matches = [Path(path) for path in glob.glob(str(repo_root / expanded), recursive=True)]
    matches = [
        path
        for path in matches
        if (path / "predictions.npy").exists() and (path / "targets.npy").exists()
    ]
    if not matches:
        raise FileNotFoundError(f"No test_results found for model={model}, dataset={dataset}, glob={expanded}")
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


def valid_mask(target: np.ndarray, null_val: float) -> np.ndarray:
    if np.isnan(null_val):
        return np.isfinite(target)
    return np.isfinite(target) & (np.abs(target - null_val) > 5e-5)


def infer_start_time(data_dir: Path, override: str | None) -> pd.Timestamp:
    if override:
        return pd.Timestamp(override)
    summary_path = data_dir / "preprocess_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        year = int(summary.get("year", 2023))
        months = summary.get("months", [1])
        return pd.Timestamp(f"{year}-{int(months[0]):02d}-01 00:00:00")
    return pd.Timestamp("2023-01-01 00:00:00")


def post_incident_pairs(
    data_dir: Path,
    index: np.ndarray,
    start_time: pd.Timestamp,
    origin_offset: int,
    num_samples: int,
    num_nodes: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv")
    matched = pd.read_csv(data_dir / "matched_incidents.csv")
    if matched.empty:
        return np.zeros((num_samples, num_nodes), dtype=bool), matched

    local_lookup = {
        int(global_idx): local_idx for local_idx, global_idx in enumerate(meta["global_index"].astype(int))
    }
    origin_to_sample = {int(mid): sample_idx for sample_idx, mid in enumerate(index[:, 1].astype(int))}
    matched = matched.copy()
    matched["dt"] = pd.to_datetime(matched["dt"], errors="coerce")
    matched = matched.dropna(subset=["dt", "global_index"])
    matched["incident_slot"] = ((matched["dt"] - start_time) / pd.Timedelta(minutes=5)).astype(int)
    matched["forecast_origin_slot"] = matched["incident_slot"] + origin_offset
    rows = []
    mask = np.zeros((num_samples, num_nodes), dtype=bool)
    for row in matched.itertuples(index=False):
        sample_idx = origin_to_sample.get(int(row.forecast_origin_slot))
        local_idx = local_lookup.get(int(row.global_index))
        if sample_idx is None or local_idx is None:
            continue
        mask[sample_idx, local_idx] = True
        rows.append(
            {
                "incident_id": row.incident_id,
                "incident_type": getattr(row, "Type"),
                "global_index": int(row.global_index),
                "local_index": local_idx,
                "incident_slot": int(row.incident_slot),
                "forecast_origin_slot": int(row.forecast_origin_slot),
                "sample_index": sample_idx,
                "dt": row.dt,
                "distance": float(row.distance),
            }
        )
    return mask, pd.DataFrame(rows)


def init_acc(horizons: Iterable[int]) -> Dict[str, float]:
    acc: Dict[str, float] = defaultdict(float)
    for horizon in horizons:
        acc[f"h{horizon}_valid"] = 0
    return acc


def add_slice_metrics(
    acc: Dict[str, float],
    pred: np.ndarray,
    target: np.ndarray,
    selected_2d: np.ndarray,
    horizons: Iterable[int],
    null_val: float,
) -> None:
    acc["node_windows"] += int(selected_2d.sum())
    if not selected_2d.any():
        return
    valid = valid_mask(target, null_val)
    err = pred - target
    for horizon in horizons:
        h = horizon - 1
        if h < 0 or h >= pred.shape[1]:
            continue
        selected = selected_2d & valid[:, h, :]
        count = int(selected.sum())
        acc[f"h{horizon}_valid"] += count
        if count == 0:
            continue
        h_err = err[:, h, :]
        h_abs = np.abs(h_err[selected])
        h_sq = h_err[selected] ** 2
        h_target = target[:, h, :][selected]
        acc[f"h{horizon}_abs_sum"] += float(h_abs.sum())
        acc[f"h{horizon}_sq_sum"] += float(h_sq.sum())
        acc[f"h{horizon}_mape_sum"] += float((h_abs / np.abs(h_target)).sum())


def finalize_rows(
    model: str,
    dataset: str,
    result_dir: Path,
    acc_by_split: Dict[str, Dict[str, float]],
    horizons: Iterable[int],
    repo_root: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split_name, acc in acc_by_split.items():
        row: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "split": split_name,
            "node_windows": int(acc["node_windows"]),
            "result_dir": str(result_dir.relative_to(repo_root)),
        }
        for horizon in horizons:
            valid = int(acc[f"h{horizon}_valid"])
            row[f"valid@t{horizon}"] = valid
            row[f"MAE@t{horizon}"] = float(acc[f"h{horizon}_abs_sum"] / valid) if valid else np.nan
            row[f"RMSE@t{horizon}"] = (
                float(np.sqrt(acc[f"h{horizon}_sq_sum"] / valid)) if valid else np.nan
            )
            row[f"MAPE@t{horizon}"] = (
                float(100.0 * acc[f"h{horizon}_mape_sum"] / valid) if valid else np.nan
            )
        rows.append(row)
    return rows


def analyze_model(
    repo_root: Path,
    data_dir: Path,
    dataset: str,
    model: str,
    args: argparse.Namespace,
    horizons: Tuple[int, ...],
) -> Tuple[List[Dict[str, object]], Dict[str, object], pd.DataFrame]:
    index = np.load(data_dir / "index.npz")["test"].astype(np.int64)
    num_samples = len(index)
    data_shape = np.load(data_dir / "data.npz")["data"].shape
    num_nodes = int(data_shape[1])
    shape = (num_samples, args.output_len, num_nodes, 1)
    result_dir = find_result_dir(repo_root, model, dataset, args.result_glob)
    predictions = load_raw_or_npy(result_dir / "predictions.npy", shape)
    targets = load_raw_or_npy(result_dir / "targets.npy", shape)

    incident_mask, incident_cases = post_incident_pairs(
        data_dir,
        index,
        infer_start_time(data_dir, args.start_time),
        args.incident_origin_offset,
        num_samples,
        num_nodes,
    )
    split_masks = {
        "General": np.ones((num_samples, num_nodes), dtype=bool),
        "Incident": incident_mask,
    }
    acc_by_split = {name: init_acc(horizons) for name in split_masks}
    for offset in range(0, num_samples, args.chunk_size):
        stop = min(offset + args.chunk_size, num_samples)
        pred = np.asarray(predictions[offset:stop, :, :, 0], dtype=np.float32)
        target = np.asarray(targets[offset:stop, :, :, 0], dtype=np.float32)
        for split_name, split_mask in split_masks.items():
            add_slice_metrics(
                acc_by_split[split_name],
                pred,
                target,
                split_mask[offset:stop],
                horizons,
                args.null_val,
            )
    rows = finalize_rows(model, dataset, result_dir, acc_by_split, horizons, repo_root)
    meta = {
        "dataset": dataset,
        "model": model,
        "result_dir": str(result_dir.relative_to(repo_root)),
        "data_shape": list(data_shape),
        "test_index_shape": list(index.shape),
        "prediction_shape": list(shape),
        "incident_node_windows": int(incident_mask.sum()),
        "matched_incident_cases": int(len(incident_cases)),
    }
    return rows, meta, incident_cases


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_root) / args.dataset
    output_dir = (repo_root / args.output_dir / args.dataset).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)

    rows: List[Dict[str, object]] = []
    metadata: List[Dict[str, object]] = []
    case_tables: List[pd.DataFrame] = []
    for model in args.models:
        model_rows, model_meta, incident_cases = analyze_model(
            repo_root, data_dir, args.dataset, model, args, horizons
        )
        rows.extend(model_rows)
        metadata.append(model_meta)
        if not incident_cases.empty:
            incident_cases = incident_cases.copy()
            incident_cases["model"] = model
            case_tables.append(incident_cases)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "post_incident_forecasting_table.csv", index=False)
    if case_tables:
        pd.concat(case_tables, ignore_index=True).to_csv(output_dir / "post_incident_cases.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "models": args.models,
        "horizons": list(horizons),
        "post_incident_definition": (
            "incident slot from reported dt; forecasting origin is the next 5-minute slot; "
            "Incident metrics use only the matched sensor/node at that origin"
        ),
        "metadata": metadata,
        "files": {
            "table_csv": str(output_dir / "post_incident_forecasting_table.csv"),
            "cases_csv": str(output_dir / "post_incident_cases.csv"),
        },
    }
    with (output_dir / "post_incident_forecasting_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary["files"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
