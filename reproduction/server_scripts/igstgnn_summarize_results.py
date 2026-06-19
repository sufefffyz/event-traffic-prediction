#!/usr/bin/env python3
"""Summarize IGSTGNN exported test_result_s*.npz files into CSV."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize IGSTGNN test result npz files.")
    parser.add_argument(
        "--results",
        nargs="*",
        type=Path,
        default=[],
        help="Explicit test_result_s*.npz files.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Scan this experiments/igstgnn directory for test_result_s*.npz files.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def discover_results(args):
    results = [path.resolve() for path in args.results]
    if args.experiments_root is not None:
        results.extend(sorted(args.experiments_root.resolve().glob("*/test_result_s*.npz")))
    unique = []
    seen = set()
    for path in results:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("No test_result_s*.npz files found.")
    return unique


def row_from_npz(path):
    import numpy as np

    data = np.load(path, allow_pickle=True)
    metrics = np.asarray(data["metrics_by_horizon"], dtype=float)
    average = np.asarray(data["metrics_average"], dtype=float)
    row = {
        "dataset": str(np.asarray(data["dataset"]).item()),
        "seed": int(np.asarray(data["seed"]).item()),
        "test_result": str(path),
        "checkpoint": str(np.asarray(data["checkpoint"]).item()),
        "avg_mae": average[0],
        "avg_mape": average[1],
        "avg_rmse": average[2],
    }
    for idx, values in enumerate(metrics, start=1):
        row[f"h{idx}_mae"] = values[0]
        row[f"h{idx}_mape"] = values[1]
        row[f"h{idx}_rmse"] = values[2]
    return row


def main():
    args = parse_args()
    rows = [row_from_npz(path) for path in discover_results(args)]
    rows.sort(key=lambda row: row["dataset"])

    fieldnames = list(rows[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
