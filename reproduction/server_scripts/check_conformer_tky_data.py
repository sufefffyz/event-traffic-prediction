#!/usr/bin/env python3
from pathlib import Path
import sys


def main():
    repo_dir = Path(__file__).resolve().parents[2]
    conformer_dir = repo_dir / "reproduction" / "ConFormer"
    sys.path.insert(0, str(conformer_dir))

    from lib.data_prepare import get_dataloaders_from_index_data

    train_loader, val_loader, test_loader, scaler = get_dataloaders_from_index_data(
        str(conformer_dir / "data" / "TKY"),
        tod=True,
        dow=True,
        acc=True,
        reg=True,
        batch_size=16,
        in_steps=6,
        out_steps=6,
    )
    x, y = next(iter(train_loader))
    print("train_batches", len(train_loader))
    print("val_batches", len(val_loader))
    print("test_batches", len(test_loader))
    print("x_shape", tuple(x.shape))
    print("y_shape", tuple(y.shape))
    print("scaler_mean", float(scaler.mean))
    print("scaler_std", float(scaler.std))


if __name__ == "__main__":
    main()
