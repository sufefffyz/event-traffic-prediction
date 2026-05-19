"""Matched-control audit for event-conditioned traffic windows.

For each event node-window, this script finds non-event controls from the same
split, same node, same day-of-week, and same time-of-day. Controls must have no
event in either the history or future window for the same event channel.
"""

from __future__ import annotations

import argparse
import json
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
    steps_per_day: int
    event_channels: Tuple[str, ...]


DATASET_SPECS = {
    "SD": DatasetSpec("SD", in_steps=12, out_steps=12, steps_per_day=96, event_channels=("accident",)),
    "BA": DatasetSpec("BA", in_steps=12, out_steps=12, steps_per_day=96, event_channels=("accident",)),
    "TKY": DatasetSpec("TKY", in_steps=6, out_steps=6, steps_per_day=144, event_channels=("accident", "regulation")),
}


GROUPS = {
    "future_onset": "history has no event, future has event",
    "history_only": "history has event, future has no event",
    "ongoing": "history has event, future has event",
    "post_last_slot": "last history slot has event",
    "future_any": "future has event",
    "history_any": "history has event",
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


def load_speed_and_events(data_root: Path, spec: DatasetSpec) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DatetimeIndex]:
    data_dir = data_root / spec.name
    speed_df = read_hdf_dataframe(data_dir / "data.h5")
    speed = speed_df.values.astype(np.float32, copy=False)
    events: Dict[str, np.ndarray] = {}

    if "accident" in spec.event_channels:
        accident_path = data_dir / "accident.h5"
        if accident_path.exists():
            events["accident"] = read_hdf_dataframe(accident_path).values.astype(np.float32, copy=False)
        elif (data_dir / "data.npz").exists():
            data = np.load(data_dir / "data.npz")["data"]
            if data.shape[-1] > 3:
                events["accident"] = data[..., 3].astype(np.float32, copy=False)

    if spec.name == "TKY":
        external = np.load(data_dir / "external.npz")["data"].astype(np.float32, copy=False)
        events["accident"] = external[..., 0]
        events["regulation"] = external[..., 1]

    common_t = speed.shape[0]
    common_n = speed.shape[1]
    for event in events.values():
        common_t = min(common_t, event.shape[0])
        common_n = min(common_n, event.shape[1])

    speed = speed[:common_t, :common_n]
    events = {name: event[:common_t, :common_n] for name, event in events.items()}
    return speed, events, speed_df.index[:common_t]


def make_prefix(x: np.ndarray) -> np.ndarray:
    prefix = np.zeros((x.shape[0] + 1, x.shape[1]), dtype=np.float64)
    prefix[1:] = np.cumsum(x, axis=0, dtype=np.float64)
    return prefix


def make_timeweek_keys(index: pd.DatetimeIndex, mids: np.ndarray, steps_per_day: int) -> np.ndarray:
    timestamps = index[mids]
    tod_slot = (timestamps.hour * 60 + timestamps.minute) // (24 * 60 // steps_per_day)
    return timestamps.dayofweek.to_numpy(dtype=np.int64) * steps_per_day + tod_slot.to_numpy(dtype=np.int64)


def quantiles(values: np.ndarray, qs: Iterable[float] = (0, 0.5, 0.9, 0.99, 1.0)) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def metric_names(out_steps: int) -> List[str]:
    horizons = [h for h in (1, 3, 6, 12) if h <= out_steps]
    return [
        "history_mean",
        "future_mean",
        "future_minus_history_mean",
        "h1_minus_last_history",
        "future_min_minus_last_history",
        "future_max_drop_from_last_history",
        *[f"h{h}_minus_last_history" for h in horizons],
    ]


def compute_metrics(speed: np.ndarray, speed_prefix: np.ndarray, start: np.ndarray, mid: np.ndarray, end: np.ndarray, in_steps: int, out_steps: int) -> Dict[str, np.ndarray]:
    history_mean = (speed_prefix[mid] - speed_prefix[start]) / in_steps
    future_mean = (speed_prefix[end] - speed_prefix[mid]) / out_steps
    last_history = speed[mid - 1]
    future_values = np.stack([speed[mid + k] for k in range(out_steps)], axis=1)
    future_min = future_values.min(axis=1)

    metrics = {
        "history_mean": history_mean,
        "future_mean": future_mean,
        "future_minus_history_mean": future_mean - history_mean,
        "h1_minus_last_history": future_values[:, 0, :] - last_history,
        "future_min_minus_last_history": future_min - last_history,
        "future_max_drop_from_last_history": last_history - future_min,
    }
    for h in (1, 3, 6, 12):
        if h <= out_steps:
            metrics[f"h{h}_minus_last_history"] = future_values[:, h - 1, :] - last_history
    return metrics


def group_masks(hist_any: np.ndarray, fut_any: np.ndarray, last_event: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "future_onset": (~hist_any) & fut_any,
        "history_only": hist_any & (~fut_any),
        "ongoing": hist_any & fut_any,
        "post_last_slot": last_event,
        "future_any": fut_any,
        "history_any": hist_any,
    }


def add_event_values(acc: Dict[str, Dict[str, float]], group: str, mask: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    item = acc[group]
    count = int(mask.sum())
    item["event_count"] += count
    if count == 0:
        return
    for name, arr in metrics.items():
        item[f"event_{name}_sum"] += float(arr[mask].sum())
        item[f"event_{name}_sq_sum"] += float((arr[mask] ** 2).sum())


def add_matched_values(
    acc: Dict[str, Dict[str, float]],
    group: str,
    mask: np.ndarray,
    metrics: Dict[str, np.ndarray],
    keys: np.ndarray,
    control_count: np.ndarray,
    control_sums: Dict[str, np.ndarray],
) -> None:
    item = acc[group]
    if not mask.any():
        return
    rows, nodes = np.where(mask)
    matched = control_count[nodes, keys[rows]]
    valid = matched > 0
    item["matched_event_count"] += int(valid.sum())
    item["control_candidate_count_sum"] += int(matched[valid].sum()) if valid.any() else 0
    if not valid.any():
        return
    rows = rows[valid]
    nodes = nodes[valid]
    matched = matched[valid].astype(np.float64)
    key_values = keys[rows]
    for name, arr in metrics.items():
        event_values = arr[rows, nodes]
        control_mean = control_sums[name][nodes, key_values] / matched
        diff = event_values - control_mean
        item[f"matched_event_{name}_sum"] += float(event_values.sum())
        item[f"matched_control_{name}_sum"] += float(control_mean.sum())
        item[f"matched_diff_{name}_sum"] += float(diff.sum())
        item[f"matched_diff_{name}_sq_sum"] += float((diff ** 2).sum())


def init_acc(metric_names_: List[str]) -> Dict[str, Dict[str, float]]:
    acc = {}
    for group in GROUPS:
        item = defaultdict(float)
        item["event_count"] = 0
        item["matched_event_count"] = 0
        item["control_candidate_count_sum"] = 0
        for name in metric_names_:
            item[f"event_{name}_sum"] = 0.0
            item[f"event_{name}_sq_sum"] = 0.0
            item[f"matched_event_{name}_sum"] = 0.0
            item[f"matched_control_{name}_sum"] = 0.0
            item[f"matched_diff_{name}_sum"] = 0.0
            item[f"matched_diff_{name}_sq_sum"] = 0.0
        acc[group] = item
    return acc


def finalize(acc: Dict[str, Dict[str, float]], metric_names_: List[str], total_pairs: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for group, item in acc.items():
        event_count = int(item["event_count"])
        matched_count = int(item["matched_event_count"])
        row = {
            "event_count": event_count,
            "event_ratio": float(event_count / total_pairs) if total_pairs else 0.0,
            "matched_event_count": matched_count,
            "match_rate": float(matched_count / event_count) if event_count else 0.0,
            "avg_control_candidates": float(item["control_candidate_count_sum"] / matched_count) if matched_count else 0.0,
        }
        for name in metric_names_:
            if event_count:
                row[f"event_{name}"] = float(item[f"event_{name}_sum"] / event_count)
            else:
                row[f"event_{name}"] = None
            if matched_count:
                diff_mean = item[f"matched_diff_{name}_sum"] / matched_count
                diff_var = max(item[f"matched_diff_{name}_sq_sum"] / matched_count - diff_mean * diff_mean, 0.0)
                row[f"matched_event_{name}"] = float(item[f"matched_event_{name}_sum"] / matched_count)
                row[f"matched_control_{name}"] = float(item[f"matched_control_{name}_sum"] / matched_count)
                row[f"matched_diff_{name}"] = float(diff_mean)
                row[f"matched_diff_{name}_std"] = float(np.sqrt(diff_var))
            else:
                row[f"matched_event_{name}"] = None
                row[f"matched_control_{name}"] = None
                row[f"matched_diff_{name}"] = None
                row[f"matched_diff_{name}_std"] = None
        out[group] = row
    return out


def analyze_split(speed: np.ndarray, event: np.ndarray, index: np.ndarray, time_index: pd.DatetimeIndex, spec: DatasetSpec, chunk_size: int) -> Dict[str, Dict[str, float]]:
    event_bool = event > 0
    event_prefix = make_prefix(event_bool.astype(np.float32))
    speed_prefix = make_prefix(speed)
    keys_all = make_timeweek_keys(time_index, index[:, 1].astype(np.int64), spec.steps_per_day)
    num_keys = 7 * spec.steps_per_day
    names = metric_names(spec.out_steps)

    control_count = np.zeros((speed.shape[1], num_keys), dtype=np.int32)
    control_sums = {name: np.zeros((speed.shape[1], num_keys), dtype=np.float64) for name in names}

    for offset in range(0, len(index), chunk_size):
        chunk = index[offset : offset + chunk_size].astype(np.int64)
        keys = keys_all[offset : offset + len(chunk)]
        start, mid, end = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        hist_count = event_prefix[mid] - event_prefix[start]
        fut_count = event_prefix[end] - event_prefix[mid]
        control_mask = (hist_count == 0) & (fut_count == 0)
        metrics = compute_metrics(speed, speed_prefix, start, mid, end, spec.in_steps, spec.out_steps)
        for row, key in enumerate(keys):
            nodes = np.flatnonzero(control_mask[row])
            if nodes.size == 0:
                continue
            control_count[nodes, key] += 1
            for name, arr in metrics.items():
                control_sums[name][nodes, key] += arr[row, nodes]

    acc = init_acc(names)
    for offset in range(0, len(index), chunk_size):
        chunk = index[offset : offset + chunk_size].astype(np.int64)
        keys = keys_all[offset : offset + len(chunk)]
        start, mid, end = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        hist_count = event_prefix[mid] - event_prefix[start]
        fut_count = event_prefix[end] - event_prefix[mid]
        masks = group_masks(hist_count > 0, fut_count > 0, event_bool[mid - 1])
        metrics = compute_metrics(speed, speed_prefix, start, mid, end, spec.in_steps, spec.out_steps)
        for group, mask in masks.items():
            add_event_values(acc, group, mask, metrics)
            add_matched_values(acc, group, mask, metrics, keys, control_count, control_sums)

    total_pairs = int(len(index) * speed.shape[1])
    result = finalize(acc, names, total_pairs)
    controls_per_key_node = control_count[control_count > 0]
    result["_control_pool"] = {
        "control_units": int(control_count.sum()),
        "node_timeweek_cells_with_control": int(np.count_nonzero(control_count)),
        "controls_per_nonempty_cell_quantiles": quantiles(controls_per_key_node),
    }
    return result


def flatten_rows(summary: Dict) -> List[Dict]:
    rows = []
    for dataset, dataset_info in summary.items():
        for event_name, event_info in dataset_info["events"].items():
            for split, split_info in event_info["splits"].items():
                for group, row in split_info.items():
                    if group == "_control_pool":
                        continue
                    rows.append({
                        "dataset": dataset,
                        "event": event_name,
                        "split": split,
                        "group": group,
                        "group_definition": GROUPS[group],
                        **row,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="reproduction/ConFormer/data")
    parser.add_argument("--datasets", nargs="+", default=["SD", "BA", "TKY"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-dir", default="reproduction/metric_recalculation/event_matched_control")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        speed, events, time_index = load_speed_and_events(data_root, spec)
        index_obj = np.load(data_root / dataset / "index.npz")
        dataset_summary = {
            "speed_shape": list(speed.shape),
            "time_range": [str(time_index[0]), str(time_index[-1])],
            "in_steps": spec.in_steps,
            "out_steps": spec.out_steps,
            "events": {},
        }
        for event_name in spec.event_channels:
            event = events[event_name]
            event_summary = {"event_shape": list(event.shape), "splits": {}}
            for split in args.splits:
                event_summary["splits"][split] = analyze_split(
                    speed=speed,
                    event=event,
                    index=index_obj[split],
                    time_index=time_index,
                    spec=spec,
                    chunk_size=args.chunk_size,
                )
            dataset_summary["events"][event_name] = event_summary
        summary[dataset] = dataset_summary

    with open(output_dir / "event_matched_control_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    rows = flatten_rows(summary)
    pd.DataFrame(rows).to_csv(output_dir / "event_matched_control_stats.csv", index=False)
    print(json.dumps({"output_dir": str(output_dir), "datasets": args.datasets, "splits": args.splits}, indent=2))


if __name__ == "__main__":
    main()
