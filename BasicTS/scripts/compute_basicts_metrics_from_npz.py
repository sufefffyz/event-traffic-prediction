#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basicts.metrics import masked_mae, masked_mape, masked_rmse


def _load_array(npz: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> np.ndarray:
    for key in candidates:
        if key in npz.files:
            return npz[key]
    raise KeyError(f"None of {list(candidates)} found in {npz.files}")


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 3:
        array = array[..., None]
    return torch.from_numpy(array.astype(np.float32, copy=False))


def _compute(prediction: torch.Tensor, target: torch.Tensor, null_val: float) -> Dict[str, float]:
    mae = masked_mae(prediction=prediction, target=target, null_val=null_val).item()
    mape = masked_mape(prediction=prediction, target=target, null_val=null_val).item()
    rmse = masked_rmse(prediction=prediction, target=target, null_val=null_val).item()
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MAPE_percent": mape * 100.0,
    }


def compute_metrics(path: str, null_val: float, horizons: Iterable[int]) -> Dict[str, Dict[str, float]]:
    data = np.load(path, allow_pickle=True)
    prediction = _to_tensor(_load_array(data, ["prediction", "predictions", "pred"]))
    target = _to_tensor(_load_array(data, ["target", "targets", "label", "labels"]))
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch for {path}: pred={prediction.shape}, target={target.shape}")

    results = {
        "path": path,
        "shape": list(prediction.shape),
        "null_val": null_val,
        "overall": _compute(prediction, target, null_val),
    }
    for horizon in horizons:
        index = horizon - 1
        if index < 0 or index >= prediction.shape[1]:
            continue
        results[f"horizon_{horizon}"] = _compute(
            prediction[:, index : index + 1],
            target[:, index : index + 1],
            null_val,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute saved test_result.npz files with BasicTS masked metrics."
    )
    parser.add_argument("paths", nargs="+", help="One or more .npz result files.")
    parser.add_argument("--null-val", type=float, default=0.0)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    all_results = [compute_metrics(path, args.null_val, horizons) for path in args.paths]

    print(json.dumps(all_results, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
