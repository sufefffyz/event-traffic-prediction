import argparse
import os
import sys

import numpy as np


def load_object_array(path):
    # Kaggle's object array was saved with NumPy 2.x module paths.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    return np.load(path, allow_pickle=True)


def compute_train_stats(samples):
    total = 0.0
    total_sq = 0.0
    count = 0
    for sample in samples:
        values = sample["x_data"][..., 0].astype(np.float64, copy=False)
        total += values.sum()
        total_sq += np.square(values).sum()
        count += values.size
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    std = np.sqrt(var)
    if std == 0:
        std = 1.0
    return np.float32(mean), np.float32(std)


def normalize_sample(sample, mean, std):
    normalized = dict(sample)
    for key in ("x_data", "y_data"):
        values = np.array(sample[key], copy=True)
        values[..., 0] = (values[..., 0] - mean) / std
        normalized[key] = values.astype(np.float32, copy=False)
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/xtraffic/Alameda")
    parser.add_argument("--source", default="incidents_traffic_data.npy")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    source_path = os.path.join(args.data_dir, args.source)
    samples = load_object_array(source_path)
    num_samples = len(samples)
    train_end = int(num_samples * args.train_ratio)
    val_end = int(num_samples * (args.train_ratio + args.val_ratio))

    raw_splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    print(
        "AI-supplemented preprocessing: official code reads incident_data_train/val/test.npy "
        "and incident_data_stats.npz, but the public Kaggle archive provides incidents_traffic_data.npy. "
        "This script reconstructs those files without changing the official model/training code."
    )
    print(
        f"chronological split ratios: train={args.train_ratio}, "
        f"val={args.val_ratio}, test={1.0 - args.train_ratio - args.val_ratio}"
    )

    mean, std = compute_train_stats(raw_splits["train"])
    splits = {
        name: np.array(
            [normalize_sample(sample, mean, std) for sample in split_samples],
            dtype=object,
        )
        for name, split_samples in raw_splits.items()
    }

    for name, split_samples in splits.items():
        out_path = os.path.join(args.data_dir, f"incident_data_{name}.npy")
        np.save(out_path, split_samples, allow_pickle=True)
        print(f"{name}: {len(split_samples)} -> {out_path}")

    stats_path = os.path.join(args.data_dir, "incident_data_stats.npz")
    np.savez(stats_path, mean=mean, std=std)
    print(f"stats: mean={mean}, std={std} -> {stats_path}")
    print("normalized x_data/y_data traffic channel 0 using train statistics")


if __name__ == "__main__":
    main()
