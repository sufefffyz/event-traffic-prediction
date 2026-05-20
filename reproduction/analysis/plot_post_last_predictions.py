"""Visualize post-last incident prediction curves.

The post-last slice follows the TraffiDent-style audit used in this project:
an event is active at the last history slot, and the model forecasts the next
future window. The script plots raw history/future speeds from the dataset and
overlays predictions saved by ConFormer or BasicTS.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    in_steps: int
    out_steps: int
    event_channel: Optional[int]


DATASET_SPECS = {
    "SD": DatasetSpec("SD", in_steps=12, out_steps=12, event_channel=3),
    "BA": DatasetSpec("BA", in_steps=12, out_steps=12, event_channel=None),
    "TKY_ACCIDENT": DatasetSpec("TKY", in_steps=6, out_steps=6, event_channel=3),
    "TKY_REGULATION": DatasetSpec("TKY", in_steps=6, out_steps=6, event_channel=4),
}


DEFAULT_RESULTS = {
    "SD": [
        (
            "ConFormer-3L-acc",
            "reproduction/ConFormer/test_results/ConFormer-SD-3layer-2026-05-18-12-16-25-test_result.npz",
        ),
        (
            "ConFormer-3L-noacc",
            "reproduction/ConFormer/test_results/ConFormer-SD-noacc-3layer-noacc3layer-2026-05-19-10-54-06-test_result.npz",
        ),
        (
            "STID",
            "BasicTS/checkpoints/STID/ConFormer_SD_100_12_12_pure/f80e28235b0b079f8b067790a50309d4/test_results",
        ),
        (
            "STID+Acc",
            "BasicTS/checkpoints/STIDAccident/ConFormer_SD_100_12_12_accident/0238117603ded453c92d0197c39ac881/test_results",
        ),
    ]
}


COLORS = {
    "truth": "#111111",
    "history": "#6f6f6f",
    "event": "#d62728",
    "boundary": "#444444",
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


def resolve_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def parse_result_arg(values: Optional[List[str]]) -> Optional[List[Tuple[str, str]]]:
    if not values:
        return None
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Result spec must be LABEL=PATH, got: {value}")
        label, path = value.split("=", 1)
        parsed.append((label.strip(), path.strip()))
    return parsed


def ensure_4d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array[..., None]
    if array.ndim == 4:
        return array
    raise ValueError(f"Expected 3D or 4D array, got shape {array.shape}")


def load_array(path: Path, expected_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    try:
        return np.load(path)
    except ValueError as exc:
        if "pickled data" not in str(exc) or expected_shape is None:
            raise
        expected_bytes = int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path} looks like a raw memmap, but size {actual_bytes} "
                f"does not match expected shape {expected_shape} ({expected_bytes} bytes)"
            ) from exc
        return np.memmap(path, dtype=np.float32, mode="r", shape=expected_shape)


def load_prediction_result(path: Path, expected_shape: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if path.is_dir():
        prediction_path = path / "prediction.npy"
        if not prediction_path.exists():
            prediction_path = path / "predictions.npy"
        target_path = path / "target.npy"
        if not target_path.exists():
            target_path = path / "targets.npy"
        prediction = load_array(prediction_path, expected_shape)
        target = load_array(target_path, expected_shape) if target_path.exists() else None
    elif path.suffix == ".npz":
        data = np.load(path)
        pred_key = "prediction" if "prediction" in data.files else "predictions"
        prediction = data[pred_key]
        target = data["target"] if "target" in data.files else None
    else:
        raise ValueError(f"Unsupported result path: {path}")
    return ensure_4d(prediction).astype(np.float32, copy=False), ensure_4d(target).astype(np.float32, copy=False) if target is not None else None


def load_dataset(data_root: Path, spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DatetimeIndex]]:
    data_dir = data_root / spec.name
    data = np.load(data_dir / "data.npz")["data"].astype(np.float32, copy=False)
    speed = data[..., 0].astype(np.float32, copy=False)
    event = None
    if spec.event_channel is not None and data.shape[-1] > spec.event_channel:
        event = data[..., spec.event_channel].astype(np.float32, copy=False)
    accident_h5 = data_dir / "accident.h5"
    if event is None and accident_h5.exists():
        event = read_hdf_dataframe(accident_h5).values.astype(np.float32, copy=False)
        common_t = min(speed.shape[0], event.shape[0])
        common_n = min(speed.shape[1], event.shape[1])
        speed = speed[:common_t, :common_n]
        event = event[:common_t, :common_n]
    if event is None:
        raise ValueError(f"{data_dir} has no usable event channel or accident.h5")
    index = np.load(data_dir / "index.npz")["test"].astype(np.int64)

    time_index = None
    h5_path = data_dir / "data.h5"
    if h5_path.exists():
        time_index = read_hdf_dataframe(h5_path).index[: speed.shape[0]]
    return speed, event, index, time_index


def relative_x_positions(in_steps: int, out_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist_x = np.arange(-in_steps + 1, 1)
    fut_x = np.arange(1, out_steps + 1)
    full_x = np.concatenate([hist_x, fut_x])
    return hist_x, fut_x, full_x


def event_relative_values(event: np.ndarray, row: np.ndarray, node: np.ndarray, index: np.ndarray, in_steps: int, out_steps: int) -> np.ndarray:
    starts = index[row, 0]
    mids = index[row, 1]
    hist = np.stack([event[starts + lag, node] for lag in range(in_steps)], axis=1)
    fut = np.stack([event[mids + lag, node] for lag in range(out_steps)], axis=1)
    return np.concatenate([hist, fut], axis=1)


def speed_relative_values(speed: np.ndarray, row: np.ndarray, node: np.ndarray, index: np.ndarray, in_steps: int, out_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    starts = index[row, 0]
    mids = index[row, 1]
    hist = np.stack([speed[starts + lag, node] for lag in range(in_steps)], axis=1)
    fut = np.stack([speed[mids + lag, node] for lag in range(out_steps)], axis=1)
    return hist, fut


def model_case_metrics(predictions: Dict[str, np.ndarray], row: np.ndarray, node: np.ndarray, true_future: np.ndarray) -> Dict[str, np.ndarray]:
    metrics = {}
    valid = true_future > 1e-5
    for label, prediction in predictions.items():
        pred_case = prediction[row, :, node, 0]
        err = pred_case - true_future
        abs_err = np.abs(err)
        masked_abs = np.where(valid, abs_err, np.nan)
        metrics[f"{label}:mae"] = np.nanmean(masked_abs, axis=1)
        metrics[f"{label}:bias"] = np.nanmean(np.where(valid, err, np.nan), axis=1)
        for horizon in (1, 3, 6, 12):
            if horizon <= pred_case.shape[1]:
                metrics[f"{label}:bias_h{horizon}"] = err[:, horizon - 1]
                metrics[f"{label}:abs_h{horizon}"] = abs_err[:, horizon - 1]
    return metrics


def select_cases(
    rows: np.ndarray,
    nodes: np.ndarray,
    speed: np.ndarray,
    event: np.ndarray,
    index: np.ndarray,
    predictions: Dict[str, np.ndarray],
    in_steps: int,
    out_steps: int,
    num_cases: int,
    mode: str,
) -> Tuple[List[int], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    hist, fut = speed_relative_values(speed, rows, nodes, index, in_steps, out_steps)
    event_rel = event_relative_values(event, rows, nodes, index, in_steps, out_steps)
    metrics = model_case_metrics(predictions, rows, nodes, fut)
    if mode == "top_drop":
        score = hist[:, -1] - fut.min(axis=1)
    elif mode == "top_bias":
        bias_values = [value for key, value in metrics.items() if key.endswith(":bias")]
        score = np.nanmean(np.stack(bias_values, axis=1), axis=1)
    else:
        mae_values = [value for key, value in metrics.items() if key.endswith(":mae")]
        score = np.nanmean(np.stack(mae_values, axis=1), axis=1)
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    return order[:num_cases].tolist(), metrics, hist, fut, event_rel


def timestamp_at(time_index: Optional[pd.DatetimeIndex], offset: int) -> str:
    if time_index is None or offset >= len(time_index):
        return str(offset)
    return str(time_index[offset])


def add_event_spans(ax: plt.Axes, full_x: np.ndarray, event_values: np.ndarray) -> None:
    added_label = False
    for x_value, event_value in zip(full_x, event_values):
        if event_value <= 0:
            continue
        label = "event slot" if not added_label else None
        ax.axvspan(x_value - 0.45, x_value + 0.45, color=COLORS["event"], alpha=0.13, linewidth=0, label=label)
        added_label = True


def plot_case(
    output_dir: Path,
    case_rank: int,
    sample_idx: int,
    node_idx: int,
    speed: np.ndarray,
    event: np.ndarray,
    index: np.ndarray,
    predictions: Dict[str, np.ndarray],
    spec: DatasetSpec,
    time_index: Optional[pd.DatetimeIndex],
) -> None:
    start, mid, end = index[sample_idx]
    hist_x, fut_x, full_x = relative_x_positions(spec.in_steps, spec.out_steps)
    history = speed[start:mid, node_idx]
    future = speed[mid:end, node_idx]
    event_values = event[start:end, node_idx]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    add_event_spans(ax, full_x, event_values)
    ax.plot(hist_x, history, color=COLORS["history"], linewidth=2.0, marker="o", markersize=3.0, label="history truth")
    ax.plot(fut_x, future, color=COLORS["truth"], linewidth=2.2, marker="o", markersize=3.2, label="future truth")

    color_cycle = plt.cm.tab10.colors
    for idx, (label, prediction) in enumerate(predictions.items()):
        pred = prediction[sample_idx, :, node_idx, 0]
        ax.plot(fut_x, pred, linewidth=1.8, marker=".", markersize=4.0, color=color_cycle[idx % len(color_cycle)], label=label)

    ax.axvline(0.5, color=COLORS["boundary"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("relative slot (0 = last history slot)")
    ax.set_ylabel("speed")
    title = (
        f"{spec.name} post-last case {case_rank}: sample={sample_idx}, node={node_idx}, "
        f"target start={timestamp_at(time_index, int(mid))}"
    )
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    base = output_dir / f"case_{case_rank:02d}_sample{sample_idx}_node{node_idx}"
    fig.savefig(f"{base}.png", dpi=220)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)


def plot_mean_curve(
    output_dir: Path,
    rows: np.ndarray,
    nodes: np.ndarray,
    speed: np.ndarray,
    event: np.ndarray,
    index: np.ndarray,
    predictions: Dict[str, np.ndarray],
    spec: DatasetSpec,
) -> None:
    hist_x, fut_x, full_x = relative_x_positions(spec.in_steps, spec.out_steps)
    hist, fut = speed_relative_values(speed, rows, nodes, index, spec.in_steps, spec.out_steps)
    event_rel = event_relative_values(event, rows, nodes, index, spec.in_steps, spec.out_steps)
    event_fraction = (event_rel > 0).mean(axis=0)

    fig, (ax, event_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.85], "hspace": 0.08},
    )
    ax.plot(hist_x, hist.mean(axis=0), color=COLORS["history"], linewidth=2.1, marker="o", markersize=3.0, label="history truth mean")
    ax.plot(fut_x, fut.mean(axis=0), color=COLORS["truth"], linewidth=2.2, marker="o", markersize=3.2, label="future truth mean")
    color_cycle = plt.cm.tab10.colors
    for idx, (label, prediction) in enumerate(predictions.items()):
        pred_mean = prediction[rows, :, nodes, 0].mean(axis=0)
        ax.plot(fut_x, pred_mean, linewidth=1.9, marker=".", markersize=4.0, color=color_cycle[idx % len(color_cycle)], label=label)

    ax.axvline(0.5, color=COLORS["boundary"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("speed")
    ax.set_title(f"{spec.name} post-last mean curve over {len(rows):,} node-windows", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    event_ax.bar(full_x, event_fraction, width=0.82, color=COLORS["event"], alpha=0.6)
    event_ax.set_ylabel("event\nfraction", rotation=0, labelpad=24, va="center")
    event_ax.set_xlabel("relative slot (0 = last history slot)")
    event_ax.set_ylim(0, max(1.0, float(event_fraction.max()) * 1.15))
    event_ax.axvline(0.5, color=COLORS["boundary"], linestyle="--", linewidth=1.0, alpha=0.7)
    event_ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "post_last_mean_curve.png", dpi=240)
    fig.savefig(output_dir / "post_last_mean_curve.pdf")
    plt.close(fig)


def write_case_table(
    output_dir: Path,
    selected: Iterable[int],
    rows: np.ndarray,
    nodes: np.ndarray,
    index: np.ndarray,
    hist: np.ndarray,
    fut: np.ndarray,
    event_rel: np.ndarray,
    metrics: Dict[str, np.ndarray],
    time_index: Optional[pd.DatetimeIndex],
    spec: DatasetSpec,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for rank, pos in enumerate(selected, start=1):
        sample_idx = int(rows[pos])
        node_idx = int(nodes[pos])
        start, mid, end = index[sample_idx]
        event_slots = np.where(event_rel[pos] > 0)[0].tolist()
        relative_slots = [slot - spec.in_steps + 1 for slot in event_slots]
        record: Dict[str, object] = {
            "rank": rank,
            "sample_idx": sample_idx,
            "node_idx": node_idx,
            "start_offset": int(start),
            "mid_offset": int(mid),
            "end_offset": int(end),
            "target_start_time": timestamp_at(time_index, int(mid)),
            "history_last_speed": float(hist[pos, -1]),
            "future_h1_speed": float(fut[pos, 0]),
            "future_h12_speed": float(fut[pos, -1]),
            "future_min_speed": float(fut[pos].min()),
            "event_relative_slots": " ".join(map(str, relative_slots)),
        }
        for name, values in metrics.items():
            record[safe_name(name)] = float(values[pos])
        records.append(record)
    pd.DataFrame(records).to_csv(output_dir / "selected_post_last_cases.csv", index=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default="reproduction/ConFormer/data")
    parser.add_argument("--dataset", default="SD", choices=sorted(DATASET_SPECS))
    parser.add_argument("--result", action="append", help="Prediction result as LABEL=PATH. Can be repeated.")
    parser.add_argument("--output-dir", default="reproduction/metric_recalculation/post_last_visualization")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--selection", choices=["top_mae", "top_bias", "top_drop"], default="top_mae")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    spec = DATASET_SPECS[args.dataset]
    data_root = resolve_path(args.data_root, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_specs = parse_result_arg(args.result) or DEFAULT_RESULTS.get(args.dataset, [])
    speed, event, index, time_index = load_dataset(data_root, spec)
    event_bool = event > 0
    post_last = event_bool[index[:, 1] - 1]
    rows, nodes = np.where(post_last)

    predictions: Dict[str, np.ndarray] = {}
    skipped = {}
    for label, raw_path in result_specs:
        result_path = resolve_path(raw_path, repo_root)
        if not result_path.exists():
            skipped[label] = f"missing: {result_path}"
            continue
        expected_shape = (len(index), spec.out_steps, speed.shape[1], 1)
        prediction, target = load_prediction_result(result_path, expected_shape=expected_shape)
        if prediction.shape[:3] != (len(index), spec.out_steps, speed.shape[1]):
            skipped[label] = f"shape mismatch: prediction {prediction.shape}, expected {(len(index), spec.out_steps, speed.shape[1])}"
            continue
        predictions[label] = prediction
        if target is not None and target.shape[:3] == prediction.shape[:3]:
            raw_target = speed_relative_values(speed, rows[: min(len(rows), 256)], nodes[: min(len(nodes), 256)], index, spec.in_steps, spec.out_steps)[1]
            target_diff = float(np.nanmax(np.abs(target[rows[: min(len(rows), 256)], :, nodes[: min(len(nodes), 256)], 0] - raw_target)))
            skipped[f"{label}:target_check_max_abs_diff_first256"] = target_diff

    if not predictions:
        raise RuntimeError(f"No usable prediction results found. Skipped: {skipped}")
    if len(rows) == 0:
        raise RuntimeError(f"No post-last event windows found for {args.dataset}")

    selected, metrics, hist, fut, event_rel = select_cases(
        rows,
        nodes,
        speed,
        event,
        index,
        predictions,
        spec.in_steps,
        spec.out_steps,
        min(args.num_cases, len(rows)),
        args.selection,
    )

    plot_mean_curve(output_dir, rows, nodes, speed, event, index, predictions, spec)
    case_records = write_case_table(output_dir, selected, rows, nodes, index, hist, fut, event_rel, metrics, time_index, spec)
    for rank, pos in enumerate(selected, start=1):
        plot_case(
            output_dir,
            rank,
            int(rows[pos]),
            int(nodes[pos]),
            speed,
            event,
            index,
            predictions,
            spec,
            time_index,
        )

    summary = {
        "dataset": args.dataset,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "post_last_node_windows": int(len(rows)),
        "selection": args.selection,
        "num_cases": len(case_records),
        "models": list(predictions.keys()),
        "skipped_or_checks": skipped,
        "files": {
            "mean_curve_png": str(output_dir / "post_last_mean_curve.png"),
            "mean_curve_pdf": str(output_dir / "post_last_mean_curve.pdf"),
            "cases_csv": str(output_dir / "selected_post_last_cases.csv"),
        },
    }
    with open(output_dir / "post_last_visualization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
