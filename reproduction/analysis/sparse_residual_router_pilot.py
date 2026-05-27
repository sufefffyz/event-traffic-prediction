#!/usr/bin/env python3
"""Training-free-ish CPU pilot for sparse accident residual routing.

This is not the final neural model. It tests a necessary condition:
can a lightweight residual expert, using only pre-event traffic and event/node
metadata, improve over persistence on post-incident event windows?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COUNTIES = ("LosAngeles", "Orange", "Alameda", "ContraCosta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/yuzhang_fei/TraffiDent/basicts")
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/yuzhang_fei/code/event-traffic-prediction-git/"
            "reproduction/analysis/sparse_residual_router_pilot"
        ),
    )
    parser.add_argument("--counties", nargs="+", default=list(COUNTIES))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--flow-channel", type=int, default=0)
    parser.add_argument("--start-time", default="2023-01-01 00:00:00")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--gate-quantile",
        type=float,
        default=0.75,
        help="Train-set mean absolute residual quantile used as the high-impact gate threshold.",
    )
    return parser.parse_args()


def slot_from_dt(series: pd.Series, start_time: str) -> np.ndarray:
    start = pd.Timestamp(start_time)
    dt = pd.to_datetime(series)
    minutes = (dt - start).dt.total_seconds().to_numpy() / 60.0
    return np.floor(minutes / 5.0).astype(int)


def masked_mae(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def horizon_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.array([masked_mae(pred[:, h], true[:, h]) for h in range(true.shape[1])])


def parse_width(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("mph", "").replace("ft", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_county(data_root: Path, county: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = data_root / f"TraffiDent_{county}_2023Q1"
    data = np.load(data_dir / "data.npz")["data"]
    flow = data[:, :, args.flow_channel].astype(float)
    index = np.load(data_dir / "index.npz")
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv")
    incidents = pd.read_csv(data_dir / "matched_incidents.csv")

    meta = meta.copy()
    for col in ("Road Width", "Lane Width", "Design Speed Limit"):
        if col in meta:
            meta[col + "_num"] = meta[col].map(parse_width)

    gid_to_node = {int(gid): i for i, gid in enumerate(meta["global_index"].to_numpy())}
    incidents = incidents.copy()
    incidents["node_idx"] = incidents["global_index"].map(gid_to_node)
    incidents = incidents.dropna(subset=["node_idx", "dt", "Type", "Fwy", "incident_abs_pm"])
    incidents["node_idx"] = incidents["node_idx"].astype(int)
    incidents["event_slot"] = slot_from_dt(incidents["dt"], args.start_time)
    incidents["target_start"] = incidents["event_slot"] + 1

    node_meta_cols = [
        "global_index",
        "Direction",
        "Sensor Type",
        "HOV",
        "Road Width_num",
        "Lane Width_num",
        "Design Speed Limit_num",
        "Abs PM",
    ]
    keep_cols = [c for c in node_meta_cols if c in meta.columns]
    node_meta = meta[keep_cols].rename(columns={"global_index": "global_index"})
    incidents = incidents.merge(node_meta, on="global_index", how="left")

    train_left, train_right = int(index["train"][0, 1]), int(index["train"][-1, 2])
    test_left, test_right = int(index["test"][0, 1]), int(index["test"][-1, 2])

    rows = []
    for row in incidents.itertuples(index=False):
        start = int(row.target_start)
        node = int(row.node_idx)
        if start - args.history < 0 or start + args.horizon > flow.shape[0]:
            continue
        split = None
        if train_left <= start and start + args.horizon <= train_right:
            split = "train"
        elif test_left <= start and start + args.horizon <= test_right:
            split = "test"
        if split is None:
            continue
        pre = flow[start - args.history : start, node]
        future = flow[start : start + args.horizon, node]
        if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(future)):
            continue
        tod = start % 288
        dow = (start // 288) % 7
        last = float(pre[-1])
        rec = {
            "county": county,
            "split": split,
            "target_start": start,
            "node_idx": node,
            "incident_id": getattr(row, "incident_id"),
            "type": str(getattr(row, "Type")),
            "fwy": float(getattr(row, "Fwy")),
            "distance": float(getattr(row, "distance")),
            "sensor_abs_pm": float(getattr(row, "sensor_abs_pm")),
            "incident_abs_pm": float(getattr(row, "incident_abs_pm")),
            "abs_pm_gap": float(getattr(row, "incident_abs_pm") - getattr(row, "sensor_abs_pm")),
            "tod_sin": float(np.sin(2 * np.pi * tod / 288.0)),
            "tod_cos": float(np.cos(2 * np.pi * tod / 288.0)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
            "pre_last": last,
            "pre_mean": float(np.mean(pre)),
            "pre_std": float(np.std(pre)),
            "pre_slope": float((pre[-1] - pre[0]) / max(1, len(pre) - 1)),
        }
        for col in ("Road Width_num", "Lane Width_num", "Design Speed Limit_num"):
            rec[col] = float(getattr(row, col, np.nan)) if hasattr(row, col) else np.nan
        for h in range(args.horizon):
            rec[f"y{h+1}"] = float(future[h])
            rec[f"persist{h+1}"] = last
            rec[f"resid{h+1}"] = float(future[h] - last)
        rows.append(rec)

    df = pd.DataFrame(rows)
    return df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()


def make_design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric_cols = [
        "fwy",
        "distance",
        "sensor_abs_pm",
        "incident_abs_pm",
        "abs_pm_gap",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "pre_last",
        "pre_mean",
        "pre_std",
        "pre_slope",
        "Road Width_num",
        "Lane Width_num",
        "Design Speed Limit_num",
    ]
    numeric_cols = [c for c in numeric_cols if c in train.columns]
    train_num = train[numeric_cols].astype(float)
    test_num = test[numeric_cols].astype(float)
    means = train_num.mean(axis=0).fillna(0.0)
    stds = train_num.std(axis=0).replace(0, 1.0).fillna(1.0)
    train_num = ((train_num.fillna(means) - means) / stds).to_numpy()
    test_num = ((test_num.fillna(means) - means) / stds).to_numpy()

    cats = pd.concat(
        [
            train[["county", "type"]].astype(str),
            test[["county", "type"]].astype(str),
        ],
        axis=0,
    )
    onehot = pd.get_dummies(cats, columns=["county", "type"], dtype=float)
    train_cat = onehot.iloc[: len(train)].to_numpy()
    test_cat = onehot.iloc[len(train) :].to_numpy()
    names = numeric_cols + list(onehot.columns)
    x_train = np.concatenate([np.ones((len(train), 1)), train_num, train_cat], axis=1)
    x_test = np.concatenate([np.ones((len(test), 1)), test_num, test_cat], axis=1)
    return x_train, x_test, ["intercept"] + names


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    reg = alpha * np.eye(x_train.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)
    return x_test @ coef


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_parts, test_parts = [], []
    for county in args.counties:
        tr, te = load_county(data_root, county, args)
        train_parts.append(tr)
        test_parts.append(te)

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    y_cols = [f"y{i+1}" for i in range(args.horizon)]
    p_cols = [f"persist{i+1}" for i in range(args.horizon)]
    r_cols = [f"resid{i+1}" for i in range(args.horizon)]

    x_train, x_test, feature_names = make_design(train, test)
    y_train = train[r_cols].to_numpy(dtype=float)
    resid_pred = ridge_fit_predict(x_train, y_train, x_test, args.ridge_alpha)
    train_impact = np.nanmean(np.abs(y_train), axis=1)
    gate_threshold = float(np.quantile(train_impact, args.gate_quantile))
    impact_score = ridge_fit_predict(
        x_train,
        train_impact[:, None],
        x_test,
        args.ridge_alpha,
    ).reshape(-1)
    impact_gate = impact_score >= gate_threshold

    high_impact_train = train_impact >= gate_threshold
    high_impact_resid_pred = ridge_fit_predict(
        x_train[high_impact_train],
        y_train[high_impact_train],
        x_test,
        args.ridge_alpha,
    )

    y_true = test[y_cols].to_numpy(dtype=float)
    persistence = test[p_cols].to_numpy(dtype=float)
    ridge_pred = persistence + resid_pred
    global_resid = persistence + np.nanmean(y_train, axis=0)[None, :]
    gated_ridge_pred = persistence.copy()
    gated_ridge_pred[impact_gate] = ridge_pred[impact_gate]
    gated_high_impact_pred = persistence.copy()
    gated_high_impact_pred[impact_gate] = (
        persistence[impact_gate] + high_impact_resid_pred[impact_gate]
    )

    county_ridge_pred = persistence.copy()
    train_counties = train["county"].to_numpy()
    test_counties = test["county"].to_numpy()
    for county in args.counties:
        train_mask = train_counties == county
        test_mask = test_counties == county
        county_resid = ridge_fit_predict(
            x_train[train_mask],
            y_train[train_mask],
            x_test[test_mask],
            args.ridge_alpha,
        )
        county_ridge_pred[test_mask] = persistence[test_mask] + county_resid

    methods = {
        "persistence": persistence,
        "global_residual_ridge_target_mean": global_resid,
        "ridge_residual_expert": ridge_pred,
        f"impact_score_gated_ridge_q{int(args.gate_quantile * 100)}": gated_ridge_pred,
        f"high_impact_train_gated_ridge_q{int(args.gate_quantile * 100)}": gated_high_impact_pred,
        "county_specific_ridge_residual_expert": county_ridge_pred,
    }

    rows = []
    h_rows = []
    for county in args.counties:
        mask = test["county"].to_numpy() == county
        for name, pred in methods.items():
            rows.append(
                {
                    "county": county,
                    "method": name,
                    "n_train_events": int((train["county"] == county).sum()),
                    "n_test_events": int(mask.sum()),
                    "mae": masked_mae(pred[mask], y_true[mask]),
                }
            )
            for h, value in enumerate(horizon_mae(pred[mask], y_true[mask]), start=1):
                h_rows.append({"county": county, "method": name, "horizon": h, "mae": value})

    rows_df = pd.DataFrame(rows)
    h_df = pd.DataFrame(h_rows)
    rows_df.to_csv(out_dir / "sparse_residual_router_pilot_summary.csv", index=False)
    h_df.to_csv(out_dir / "sparse_residual_router_pilot_by_horizon.csv", index=False)

    meta = {
        "ridge_alpha": args.ridge_alpha,
        "gate_quantile": args.gate_quantile,
        "gate_threshold_train_mean_abs_residual": gate_threshold,
        "test_gate_positive_count": int(np.sum(impact_gate)),
        "test_gate_positive_rate": float(np.mean(impact_gate)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": feature_names,
        "train_events_by_county": train["county"].value_counts().to_dict(),
        "test_events_by_county": test["county"].value_counts().to_dict(),
        "train_events_by_type": train["type"].value_counts().to_dict(),
        "test_events_by_type": test["type"].value_counts().to_dict(),
    }
    with (out_dir / "sparse_residual_router_pilot_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(rows_df.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
