"""Audit event distributions and event-conditioned history/future windows.

The script is intentionally read-only with respect to datasets. It reports:
1. event sparsity and temporal/node concentration;
2. node-window categories defined by whether an event appears in the history
   window and/or future window;
3. speed-change statistics for each category.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    in_steps: int
    out_steps: int
    event_channels: Tuple[Tuple[str, Optional[int]], ...]


DATASET_SPECS = {
    "SD": DatasetSpec("SD", in_steps=12, out_steps=12, event_channels=(("accident", 3),)),
    "BA": DatasetSpec("BA", in_steps=12, out_steps=12, event_channels=(("accident", 3),)),
    "TKY": DatasetSpec(
        "TKY",
        in_steps=6,
        out_steps=6,
        event_channels=(("accident", 3), ("regulation", 4)),
    ),
}


BUCKETS = {
    "hist0_future0": (False, False),
    "hist1_future0": (True, False),
    "hist0_future1": (False, True),
    "hist1_future1": (True, True),
}


def read_hdf_dataframe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_hdf(path).fillna(0)
    except ValueError as exc:
        if "unrecognized index type datetime64" not in str(exc):
            raise
        import tables

        with tables.open_file(path) as h5_file:
            group = getattr(h5_file.root, "t")
            values = group.block0_values.read()
            index = pd.to_datetime(group.axis1.read())
            columns = group.axis0.read()
            if columns.dtype.kind == "S":
                columns = columns.astype(str)
        return pd.DataFrame(values, index=index, columns=columns).fillna(0)


def quantiles(values: np.ndarray, qs: Iterable[float] = (0, 0.5, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def load_dataset(data_root: Path, spec: DatasetSpec) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Optional[pd.DatetimeIndex]]:
    data_dir = data_root / spec.name
    data = np.load(data_dir / "data.npz")["data"].astype(np.float32, copy=False)
    speed = data[..., 0].astype(np.float32, copy=False)

    events: Dict[str, np.ndarray] = {}
    for event_name, channel_idx in spec.event_channels:
        event = None
        if channel_idx is not None and data.shape[-1] > channel_idx:
            event = data[..., channel_idx].astype(np.float32, copy=False)
        elif event_name == "accident" and (data_dir / "accident.h5").exists():
            event_df = read_hdf_dataframe(data_dir / "accident.h5")
            event = event_df.values.astype(np.float32, copy=False)
        if event is None:
            continue
        common_t = min(speed.shape[0], event.shape[0])
        common_n = min(speed.shape[1], event.shape[1])
        events[event_name] = event[:common_t, :common_n]
        speed = speed[:common_t, :common_n]

    index = np.load(data_dir / "index.npz")
    data_index = None
    h5_path = data_dir / "data.h5"
    if h5_path.exists():
        data_index = read_hdf_dataframe(h5_path).index[: speed.shape[0]]
    return speed, events, index, data_index


def value_counts(values: np.ndarray, limit: int = 30) -> Dict[str, int]:
    nz = values[values > 0]
    if nz.size == 0:
        return {}
    vals, counts = np.unique(nz, return_counts=True)
    pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0]))[:limit]
    return {str(float(v) if isinstance(v, np.floating) else v): int(c) for v, c in pairs}


def event_distribution(event: np.ndarray, split_indices: Dict[str, np.ndarray]) -> Dict:
    event_bool = event > 0
    active_per_node = event_bool.sum(axis=0)
    active_per_time = event_bool.sum(axis=1)
    onset = event_bool & np.concatenate([np.zeros((1, event_bool.shape[1]), dtype=bool), ~event_bool[:-1]], axis=0)
    runs_per_node = onset.sum(axis=0)
    active_times = active_per_time[active_per_time > 0]
    active_nodes = active_per_node[active_per_node > 0]

    split_stats = {}
    for split, idx in split_indices.items():
        start = int(idx[0, 0])
        end = int(idx[-1, 2])
        chunk = event_bool[start:end]
        active_per_time_split = chunk.sum(axis=1)
        split_stats[split] = {
            "time_range": [start, end],
            "nonzero_ratio": float(chunk.mean()),
            "nonzero_count": int(chunk.sum()),
            "timesteps_with_any_event": int(np.count_nonzero(active_per_time_split)),
            "timesteps_with_any_event_ratio": float(np.count_nonzero(active_per_time_split) / len(chunk)),
            "active_nodes_when_any_quantiles": quantiles(active_per_time_split[active_per_time_split > 0]),
        }

    top_node_ids = np.argsort(active_per_node)[-10:][::-1]
    return {
        "shape": list(event.shape),
        "nonzero_count": int(event_bool.sum()),
        "nonzero_ratio": float(event_bool.mean()),
        "value_counts_nonzero_top": value_counts(event),
        "nodes_with_event": int(np.count_nonzero(active_per_node)),
        "nodes_with_event_ratio": float(np.count_nonzero(active_per_node) / event.shape[1]),
        "events_per_node_quantiles": quantiles(active_nodes),
        "top_nodes_by_event_slots": [
            {"node": int(node), "event_slots": int(active_per_node[node]), "onsets": int(runs_per_node[node])}
            for node in top_node_ids
            if active_per_node[node] > 0
        ],
        "timesteps_with_any_event": int(np.count_nonzero(active_per_time)),
        "timesteps_with_any_event_ratio": float(np.count_nonzero(active_per_time) / event.shape[0]),
        "active_nodes_when_any_quantiles": quantiles(active_times),
        "onset_count": int(onset.sum()),
        "split_stats": split_stats,
    }


def make_prefix(x: np.ndarray) -> np.ndarray:
    prefix = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=np.float64)
    prefix[1:] = np.cumsum(x, axis=0, dtype=np.float64)
    return prefix


def add_masked(stats: Dict[str, Dict[str, float]], key: str, mask: np.ndarray, values: Dict[str, np.ndarray]) -> None:
    count = int(mask.sum())
    item = stats[key]
    item["count"] += count
    if count == 0:
        return
    for name, arr in values.items():
        item[f"{name}_sum"] += float(arr[mask].sum())
        item[f"{name}_sq_sum"] += float((arr[mask] ** 2).sum())


def finalize_bucket_stats(stats: Dict[str, Dict[str, float]], metric_names: List[str], total_pairs: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for bucket, item in stats.items():
        count = int(item["count"])
        row = {
            "count": count,
            "ratio": float(count / total_pairs) if total_pairs else 0.0,
        }
        for name in metric_names:
            if count == 0:
                row[name] = None
                row[f"{name}_std"] = None
                continue
            mean = item[f"{name}_sum"] / count
            var = max(item[f"{name}_sq_sum"] / count - mean * mean, 0.0)
            row[name] = float(mean)
            row[f"{name}_std"] = float(np.sqrt(var))
        out[bucket] = row
    return out


def event_window_stats(speed: np.ndarray, event: np.ndarray, indices: Dict[str, np.ndarray], in_steps: int, out_steps: int) -> Dict:
    event_bool = event > 0
    event_prefix = make_prefix(event_bool.astype(np.float32))
    speed_prefix = make_prefix(speed)

    horizon_steps = [h for h in (1, 3, 6, 12) if h <= out_steps]
    metric_names = [
        "history_mean",
        "future_mean",
        "future_minus_history_mean",
        "h1_minus_last_history",
        "future_min_minus_last_history",
        "future_max_drop_from_last_history",
        "history_event_slot_fraction",
        "future_event_slot_fraction",
        *[f"h{h}_minus_last_history" for h in horizon_steps],
    ]

    all_results = {}
    chunk_size = 512

    for split, idx in indices.items():
        stats = {
            bucket: defaultdict(float)
            for bucket in [*BUCKETS.keys(), "post_last_slot", "future_any", "history_any"]
        }
        for item in stats.values():
            item["count"] = 0

        for offset in range(0, len(idx), chunk_size):
            chunk = idx[offset : offset + chunk_size].astype(np.int64)
            start = chunk[:, 0]
            mid = chunk[:, 1]
            end = chunk[:, 2]

            hist_event_count = event_prefix[mid] - event_prefix[start]
            fut_event_count = event_prefix[end] - event_prefix[mid]
            hist_any = hist_event_count > 0
            fut_any = fut_event_count > 0
            last_event = event_bool[mid - 1]

            hist_mean = (speed_prefix[mid] - speed_prefix[start]) / in_steps
            fut_mean = (speed_prefix[end] - speed_prefix[mid]) / out_steps
            last_hist = speed[mid - 1]
            future_values = np.stack([speed[mid + k] for k in range(out_steps)], axis=1)
            future_min = future_values.min(axis=1)
            h1_minus_last = future_values[:, 0, :] - last_hist

            values = {
                "history_mean": hist_mean,
                "future_mean": fut_mean,
                "future_minus_history_mean": fut_mean - hist_mean,
                "h1_minus_last_history": h1_minus_last,
                "future_min_minus_last_history": future_min - last_hist,
                "future_max_drop_from_last_history": last_hist - future_min,
                "history_event_slot_fraction": hist_event_count / in_steps,
                "future_event_slot_fraction": fut_event_count / out_steps,
            }
            for h in horizon_steps:
                values[f"h{h}_minus_last_history"] = future_values[:, h - 1, :] - last_hist

            for bucket, (hist_flag, fut_flag) in BUCKETS.items():
                add_masked(stats, bucket, (hist_any == hist_flag) & (fut_any == fut_flag), values)
            add_masked(stats, "post_last_slot", last_event, values)
            add_masked(stats, "future_any", fut_any, values)
            add_masked(stats, "history_any", hist_any, values)

        total_pairs = int(len(idx) * speed.shape[1])
        all_results[split] = finalize_bucket_stats(stats, metric_names, total_pairs)

    return all_results


def flatten_rows(summary: Dict) -> Tuple[List[Dict], List[Dict]]:
    dist_rows = []
    bucket_rows = []
    for dataset, dataset_info in summary.items():
        for event_name, event_info in dataset_info["events"].items():
            dist = event_info["distribution"]
            base = {
                "dataset": dataset,
                "event": event_name,
                "nonzero_ratio": dist["nonzero_ratio"],
                "nonzero_count": dist["nonzero_count"],
                "nodes_with_event_ratio": dist["nodes_with_event_ratio"],
                "timesteps_with_any_event_ratio": dist["timesteps_with_any_event_ratio"],
                "onset_count": dist["onset_count"],
            }
            dist_rows.append(base)
            for split, buckets in event_info["window_stats"].items():
                for bucket, row in buckets.items():
                    bucket_rows.append({"dataset": dataset, "event": event_name, "split": split, "bucket": bucket, **row})
    return dist_rows, bucket_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="reproduction/ConFormer/data")
    parser.add_argument("--datasets", nargs="+", default=["SD", "BA", "TKY"])
    parser.add_argument("--output-dir", default="reproduction/metric_recalculation/event_window_audit")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        speed, events, index_obj, data_index = load_dataset(data_root, spec)
        indices = {split: index_obj[split] for split in ("train", "val", "test")}
        dataset_summary = {
            "shape": {"speed": list(speed.shape)},
            "time_range": [str(data_index[0]), str(data_index[-1])] if data_index is not None and len(data_index) else None,
            "in_steps": spec.in_steps,
            "out_steps": spec.out_steps,
            "splits": {split: list(indices[split].shape) for split in indices},
            "events": {},
        }
        for event_name, event in events.items():
            event = event[: speed.shape[0], : speed.shape[1]]
            dataset_summary["events"][event_name] = {
                "distribution": event_distribution(event, indices),
                "window_stats": event_window_stats(speed, event, indices, spec.in_steps, spec.out_steps),
            }
        summary[dataset] = dataset_summary

    with open(output_dir / "event_window_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    dist_rows, bucket_rows = flatten_rows(summary)
    pd.DataFrame(dist_rows).to_csv(output_dir / "event_distribution_summary.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(output_dir / "event_window_bucket_stats.csv", index=False)

    print(json.dumps({"output_dir": str(output_dir), "datasets": args.datasets}, indent=2))


if __name__ == "__main__":
    main()
