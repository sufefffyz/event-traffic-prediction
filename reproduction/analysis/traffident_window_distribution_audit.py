#!/usr/bin/env python3
"""Audit 12-12 traffic-state distributions for incident vs non-incident windows.

The analysis uses the official-all TraffiDent D5 matched incidents and expands
each incident to same-freeway/same-direction sensors within a post-mile radius.
It compares raw flow/occupancy/speed window statistics for event-containing
12-history + 12-future windows against no-event windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from traffident_state_response_by_distance import (
    build_expanded_event_mask,
    build_time_index,
    fill_missing,
    load_raw_channels,
    parse_channel_map,
    parse_months,
    prepare_incidents,
)


GROUPS = (
    "no_event",
    "event_any",
    "history_any",
    "future_any",
    "future_onset",
    "history_only",
    "ongoing",
    "post_last_slot",
)


@dataclass
class MetricAccumulator:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = np.inf
    max_value: float = -np.inf
    sample_size: int = 0
    sample_seen: int = 0
    sample_filled: int = 0
    sample: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __post_init__(self) -> None:
        if self.sample.size == 0 and self.sample_size > 0:
            self.sample = np.empty(self.sample_size, dtype=np.float64)

    def update(self, values: np.ndarray, rng: np.random.Generator) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            new_count = self.count + batch_count
            self.mean += delta * batch_count / new_count
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / new_count
            self.count = new_count
        self.min_value = min(self.min_value, float(values.min()))
        self.max_value = max(self.max_value, float(values.max()))
        self._update_sample(values, rng)

    def _update_sample(self, values: np.ndarray, rng: np.random.Generator) -> None:
        if self.sample_size <= 0:
            return
        self.sample_seen += int(values.size)
        max_batch_take = min(values.size, self.sample_size, 10000)
        if values.size > max_batch_take:
            values = rng.choice(values, size=max_batch_take, replace=False)
        if self.sample_filled < self.sample_size:
            take = min(values.size, self.sample_size - self.sample_filled)
            self.sample[self.sample_filled : self.sample_filled + take] = values[:take]
            self.sample_filled += take
            values = values[take:]
        if values.size == 0:
            return
        current = self.sample_values()
        merged = np.concatenate([current, values])
        if merged.size <= self.sample_size:
            self.sample[: merged.size] = merged
            self.sample_filled = int(merged.size)
        else:
            keep = rng.choice(merged, size=self.sample_size, replace=False)
            self.sample[:] = keep
            self.sample_filled = self.sample_size

    @property
    def variance(self) -> float:
        return self.m2 / max(1, self.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def sample_values(self) -> np.ndarray:
        return self.sample[: self.sample_filled].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/basicts/TraffiDent_D5_2023Q1_OfficialAll"),
    )
    parser.add_argument("--zip-path", type=Path, default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction/analysis/traffident_d5_window_distribution_audit"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument("--channel-map", default="flow:0,occupancy:1,speed:2")
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--same-direction-only", action="store_true", default=True)
    parser.add_argument("--allow-opposite-direction", dest="same_direction_only", action="store_false")
    parser.add_argument("--event-window-slots", type=int, default=2)
    parser.add_argument("--use-duration", action="store_true", default=True)
    parser.add_argument("--no-duration", dest="use_duration", action="store_false")
    parser.add_argument("--max-duration-slots", type=int, default=288)
    parser.add_argument("--fill-missing", default="interpolate", choices=["interpolate", "zero", "none"])
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-sample-size", type=int, default=50000)
    return parser.parse_args()


def make_index(num_steps: int, input_len: int, output_len: int) -> np.ndarray:
    num_samples = num_steps - input_len - output_len
    if num_samples <= 0:
        raise ValueError(f"Not enough timesteps: T={num_steps}, input_len={input_len}, output_len={output_len}")
    starts = np.arange(num_samples, dtype=np.int64)
    return np.stack([starts, starts + input_len, starts + input_len + output_len], axis=-1)


def make_prefix(values: np.ndarray) -> np.ndarray:
    prefix = np.zeros((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    prefix[1:] = np.cumsum(values.astype(np.float64), axis=0)
    return prefix


def event_group_masks(event_prefix: np.ndarray, index_chunk: np.ndarray) -> Dict[str, np.ndarray]:
    start, mid, end = index_chunk[:, 0], index_chunk[:, 1], index_chunk[:, 2]
    hist_count = event_prefix[mid] - event_prefix[start]
    fut_count = event_prefix[end] - event_prefix[mid]
    total_count = hist_count + fut_count
    hist_any = hist_count > 0
    fut_any = fut_count > 0
    return {
        "no_event": total_count == 0,
        "event_any": total_count > 0,
        "history_any": hist_any,
        "future_any": fut_any,
        "future_onset": (~hist_any) & fut_any,
        "history_only": hist_any & (~fut_any),
        "ongoing": hist_any & fut_any,
        "post_last_slot": (event_prefix[mid] - event_prefix[mid - 1]) > 0,
    }


def window_stats(prefix: np.ndarray, index_chunk: np.ndarray, input_len: int, output_len: int) -> Dict[str, np.ndarray]:
    start, mid, end = index_chunk[:, 0], index_chunk[:, 1], index_chunk[:, 2]
    hist_mean = (prefix[mid] - prefix[start]) / float(input_len)
    future_mean = (prefix[end] - prefix[mid]) / float(output_len)
    full_mean = (prefix[end] - prefix[start]) / float(input_len + output_len)
    return {
        "history_mean": hist_mean,
        "future_mean": future_mean,
        "full_mean": full_mean,
        "future_minus_history": future_mean - hist_mean,
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return np.nan
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def accumulator_rows(accumulators: Dict[tuple[str, str], MetricAccumulator]) -> list[dict]:
    rows = []
    for (group, metric), acc in sorted(accumulators.items()):
        sample = acc.sample_values()
        qs = np.quantile(sample, [0.1, 0.25, 0.5, 0.75, 0.9]) if sample.size else [np.nan] * 5
        rows.append(
            {
                "group": group,
                "metric": metric,
                "n": acc.count,
                "mean": acc.mean,
                "std": acc.std,
                "min": acc.min_value if acc.count else np.nan,
                "q10_sample": float(qs[0]),
                "q25_sample": float(qs[1]),
                "median_sample": float(qs[2]),
                "q75_sample": float(qs[3]),
                "q90_sample": float(qs[4]),
                "max": acc.max_value if acc.count else np.nan,
                "sample_n": int(sample.size),
            }
        )
    return rows


def comparison_rows(accumulators: Dict[tuple[str, str], MetricAccumulator]) -> list[dict]:
    rows = []
    metrics = sorted({metric for _, metric in accumulators})
    for metric in metrics:
        base = accumulators.get(("no_event", metric))
        if base is None or base.count == 0:
            continue
        base_sample = base.sample_values()
        base_median = float(np.median(base_sample)) if base_sample.size else np.nan
        for group in GROUPS:
            if group == "no_event":
                continue
            acc = accumulators.get((group, metric))
            if acc is None or acc.count == 0:
                continue
            pooled_std = np.sqrt((base.variance + acc.variance) / 2.0)
            se = np.sqrt(base.variance / max(base.count, 1) + acc.variance / max(acc.count, 1))
            mean_diff = acc.mean - base.mean
            sample = acc.sample_values()
            median = float(np.median(sample)) if sample.size else np.nan
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_event_group": acc.count,
                    "n_no_event": base.count,
                    "mean_group": acc.mean,
                    "mean_no_event": base.mean,
                    "mean_diff": mean_diff,
                    "mean_diff_ci95_low": mean_diff - 1.96 * se,
                    "mean_diff_ci95_high": mean_diff + 1.96 * se,
                    "standardized_mean_diff": mean_diff / max(float(pooled_std), 1e-12),
                    "median_group_sample": median,
                    "median_no_event_sample": base_median,
                    "median_diff_sample": median - base_median if np.isfinite(median) and np.isfinite(base_median) else np.nan,
                    "ks_stat_sample": ks_statistic(sample, base_sample),
                }
            )
    return rows


def plot_smd_heatmap(comparisons: list[dict], output_dir: Path) -> None:
    rows = [row for row in comparisons if row["group"] != "no_event"]
    groups = [g for g in GROUPS if g != "no_event" and any(row["group"] == g for row in rows)]
    metrics = sorted({row["metric"] for row in rows})
    matrix = np.full((len(groups), len(metrics)), np.nan)
    g_to_i = {g: i for i, g in enumerate(groups)}
    m_to_j = {m: j for j, m in enumerate(metrics)}
    for row in rows:
        matrix[g_to_i[row["group"]], m_to_j[row["metric"]]] = float(row["standardized_mean_diff"])
    vmax = max(float(np.nanpercentile(np.abs(matrix), 95)), 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(max(10.0, 0.55 * len(metrics)), 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_title("Incident-window vs no-event standardized mean differences")
    for i in range(len(groups)):
        for j in range(len(metrics)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="SMD")
    fig.tight_layout()
    fig.savefig(output_dir / "window_distribution_smd_heatmap.png", dpi=180)
    fig.savefig(output_dir / "window_distribution_smd_heatmap.pdf")
    plt.close(fig)


def plot_key_distributions(accumulators: Dict[tuple[str, str], MetricAccumulator], output_dir: Path, max_n: int) -> None:
    key_metrics = [
        "full_mean_speed",
        "full_mean_flow",
        "full_mean_occupancy",
        "future_minus_history_speed",
        "future_minus_history_flow",
        "future_minus_history_occupancy",
        "transition_congestion_score",
    ]
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(9, 2.2 * len(key_metrics)))
    for ax, metric in zip(axes, key_metrics):
        for group, color in (("no_event", "#2f6f9f"), ("event_any", "#c43c39")):
            acc = accumators_get(accumulators, group, metric)
            sample = acc.sample_values()
            if sample.size == 0:
                continue
            if sample.size > max_n:
                rng = np.random.default_rng(123)
                sample = rng.choice(sample, size=max_n, replace=False)
            ax.hist(sample, bins=80, density=True, histtype="step", linewidth=1.5, color=color, label=group)
        ax.set_title(metric)
        ax.grid(True, color="#eeeeee", linewidth=0.8)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "window_distribution_key_histograms.png", dpi=180)
    fig.savefig(output_dir / "window_distribution_key_histograms.pdf")
    plt.close(fig)


def accumators_get(accumulators: Dict[tuple[str, str], MetricAccumulator], group: str, metric: str) -> MetricAccumulator:
    return accumulators.get((group, metric), MetricAccumulator(sample_size=0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    channel_map = parse_channel_map(args.channel_map)
    rng = np.random.default_rng(args.seed)

    times = build_time_index(args.year, months)
    incidents, meta = prepare_incidents(args.dataset_dir, args.zip_path, args.year, times)
    global_indices = meta["global_index"].to_numpy(dtype=np.int64)
    channels = load_raw_channels(args.zip_path, args.year, months, global_indices, channel_map)
    channels = {name: fill_missing(values, args.fill_missing) for name, values in channels.items()}
    num_steps, num_nodes = next(iter(channels.values())).shape
    index = make_index(num_steps, args.input_len, args.output_len)

    event_mask = build_expanded_event_mask(incidents, meta, num_steps, num_nodes, args)
    event_prefix = make_prefix(event_mask.astype(np.float32))
    prefixes = {name: make_prefix(values) for name, values in channels.items()}
    raw_scales = {name: float(np.nanstd(values)) for name, values in channels.items()}

    accumulators: Dict[tuple[str, str], MetricAccumulator] = defaultdict(
        lambda: MetricAccumulator(sample_size=args.sample_size)
    )

    for left in range(0, len(index), args.chunk_size):
        right = min(len(index), left + args.chunk_size)
        index_chunk = index[left:right]
        masks = event_group_masks(event_prefix, index_chunk)
        stats = {name: window_stats(prefix, index_chunk, args.input_len, args.output_len) for name, prefix in prefixes.items()}
        metrics: dict[str, np.ndarray] = {}
        for name, stat in stats.items():
            for stat_name, values in stat.items():
                metrics[f"{stat_name}_{name}"] = values
        transition = (
            -metrics["future_minus_history_speed"] / max(raw_scales["speed"], 1e-6)
            -metrics["future_minus_history_flow"] / max(raw_scales["flow"], 1e-6)
            +metrics["future_minus_history_occupancy"] / max(raw_scales["occupancy"], 1e-6)
        )
        level = (
            -metrics["full_mean_speed"] / max(raw_scales["speed"], 1e-6)
            -metrics["full_mean_flow"] / max(raw_scales["flow"], 1e-6)
            +metrics["full_mean_occupancy"] / max(raw_scales["occupancy"], 1e-6)
        )
        metrics["transition_congestion_score"] = transition
        metrics["level_congestion_score"] = level

        for group, mask in masks.items():
            if not mask.any():
                continue
            for metric, values in metrics.items():
                accumulators[(group, metric)].update(values[mask], rng)

    summaries = accumulator_rows(accumulators)
    comparisons = comparison_rows(accumulators)
    write_csv(args.output_dir / "window_distribution_group_summary.csv", summaries)
    write_csv(args.output_dir / "window_distribution_event_vs_no_event.csv", comparisons)
    plot_smd_heatmap(comparisons, args.output_dir)
    plot_key_distributions(accumulators, args.output_dir, args.plot_sample_size)

    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "zip_path": str(args.zip_path),
        "year": args.year,
        "months": months,
        "channel_map": channel_map,
        "raw_scales": raw_scales,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "expanded_event_rule": {
            "max_distance": args.max_distance,
            "same_direction_only": bool(args.same_direction_only),
            "use_duration": bool(args.use_duration),
            "event_window_slots": args.event_window_slots,
            "max_duration_slots": args.max_duration_slots,
        },
        "matched_unique_incidents": int(len(incidents)),
        "num_nodes": int(num_nodes),
        "num_timesteps": int(num_steps),
        "num_windows": int(len(index)),
        "num_node_windows": int(len(index) * num_nodes),
        "event_slot_node_coverage": float(event_mask.mean()),
        "outputs": [
            "window_distribution_group_summary.csv",
            "window_distribution_event_vs_no_event.csv",
            "window_distribution_smd_heatmap.png",
            "window_distribution_key_histograms.png",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
