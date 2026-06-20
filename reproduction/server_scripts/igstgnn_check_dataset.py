#!/usr/bin/env python3
"""Check IGSTGNN released datasets before launching reproduction runs."""

import argparse
import json
import sys
import types
from pathlib import Path


REQUIRED_FILES = [
    "adj_matrix.npy",
    "desc_mapping.json",
    "incident_all.npy",
    "incident_stats.npz",
    "sensors.csv",
    "type_mapping.json",
    "incident_train.npy",
    "incident_val.npy",
    "incident_test.npy",
]


def install_numpy_pickle_compat(np):
    """Allow NumPy 1.x to load object arrays pickled by NumPy 2.x."""
    if "numpy._core" in sys.modules:
        return
    numpy_core = types.ModuleType("numpy._core")
    numpy_core.__dict__.update(np.core.__dict__)
    numpy_core.multiarray = np.core.multiarray
    numpy_core._multiarray_umath = np.core._multiarray_umath
    sys.modules["numpy._core"] = numpy_core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath


def parse_args():
    parser = argparse.ArgumentParser(description="Validate IGSTGNN dataset directories.")
    parser.add_argument(
        "--xtraffic-root",
        type=Path,
        required=True,
        help="Directory containing Alameda, Contra_Costa, and Orange.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Alameda", "Contra_Costa", "Orange"],
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def sample_shape(sample, key):
    import numpy as np

    value = sample.get(key)
    return list(np.asarray(value).shape) if value is not None else None


def check_dataset(root, dataset):
    import numpy as np

    install_numpy_pickle_compat(np)
    dataset_dir = root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset}: missing files: {missing}")

    summary = {"dataset": dataset, "path": str(dataset_dir), "splits": {}}
    all_samples = np.load(dataset_dir / "incident_all.npy", allow_pickle=True)
    summary["incident_all_samples"] = int(len(all_samples))
    for split in ["train", "val", "test"]:
        split_samples = np.load(dataset_dir / f"incident_{split}.npy", allow_pickle=True)
        first = split_samples[0].item() if hasattr(split_samples[0], "item") else split_samples[0]
        summary["splits"][split] = {
            "samples": int(len(split_samples)),
            "x_shape": sample_shape(first, "x_data"),
            "y_shape": sample_shape(first, "y_data"),
            "has_incident_features": "incident_features" in first,
            "has_incident_position": "incident_position" in first,
            "has_incident_distances": "incident_distances" in first,
        }

    stats = np.load(dataset_dir / "incident_stats.npz", allow_pickle=True)
    summary["stats"] = {
        "mean": np.asarray(stats["mean"]).tolist(),
        "std": np.asarray(stats["std"]).tolist(),
    }
    return summary


def main():
    args = parse_args()
    root = args.xtraffic_root.resolve()
    summaries = [check_dataset(root, dataset) for dataset in args.datasets]
    text = json.dumps(summaries, indent=2, ensure_ascii=False)
    print(text)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
