#!/usr/bin/env python3
"""Prepare paper-style TraffiDent area subsets for BasicTS.

This is intentionally separate from ``prepare_county_basicts.py`` because the
paper's post-incident forecasting experiment is described as D5/Monterey in
the table/appendix, while the main text also mentions San Bernardino. Keeping a
small area adapter avoids changing the county datasets used by the other
experiments.
"""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from prepare_county_basicts import (
    ACCIDENT_TYPES,
    add_time_features,
    build_index,
    build_time_index,
    fill_event_channel,
    fill_missing_flow,
    find_member,
    load_county_flow,
    load_incidents,
    load_metadata,
    official_style_match,
    parse_csv_floats,
    parse_csv_ints,
    period_label,
    read_csv_member,
)


AREA_FILTERS = {
    "D5": {
        "description": "Caltrans District 5; paper appendix calls this D5 (Monterey).",
        "filters": {"District": 5},
    },
    "Monterey": {
        "description": "County == Monterey; stricter than the paper's D5 wording.",
        "filters": {"County": "Monterey"},
    },
    "SanBernardino": {
        "description": "County == San Bernardino; mentioned in the paper main text.",
        "filters": {"County": "San Bernardino"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice paper-style TraffiDent areas into BasicTS data.npz/index.npz files."
    )
    parser.add_argument("--zip-path", type=Path, default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/yuzhang_fei/TraffiDent/basicts"))
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", default="1,2,3")
    parser.add_argument("--area", default="D5", choices=sorted(AREA_FILTERS))
    parser.add_argument(
        "--sensor-type",
        default="all",
        help="Sensor Type filter in sensor_meta_feature.csv. Use 'all' for no Type filter.",
    )
    parser.add_argument("--traffic-channel", type=int, default=0)
    parser.add_argument("--fill-missing", default="interpolate", choices=["interpolate", "zero", "none"])
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--split-ratio", default="0.6,0.2,0.2")
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--event-window-slots", type=int, default=2)
    parser.add_argument("--event-types", default="accident", choices=["accident", "all"])
    parser.add_argument(
        "--match-scope",
        default="subset",
        choices=["subset", "all"],
        help="Use only selected-area sensors for matching, or match globally then keep selected sensors.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_name(area: str, year: int, months: Sequence[int], sensor_type: str) -> str:
    suffix = "" if sensor_type == "all" else sensor_type
    return f"TraffiDent_{area}{suffix}_{year}{period_label(months)}"


def area_mask(meta: pd.DataFrame, area: str) -> pd.Series:
    mask = pd.Series(True, index=meta.index)
    for column, expected in AREA_FILTERS[area]["filters"].items():
        if column not in meta.columns:
            raise KeyError(f"Missing metadata column {column!r} for area={area}")
        mask &= meta[column] == expected
    return mask


def load_subset_matrix(zf: zipfile.ZipFile, member_name: str, global_indices: np.ndarray) -> np.ndarray:
    with zf.open(member_name) as fh:
        matrix = np.load(fh)
    matrix = matrix[np.ix_(global_indices, global_indices)].astype(np.float32, copy=False)
    return matrix


def maybe_write_graph_sidecars(
    zf: zipfile.ZipFile,
    out_dir: Path,
    global_indices: np.ndarray,
    summary: dict,
) -> None:
    sidecars = {}
    for source_name, output_name in (("adj_matrix.npy", "adj_matrix.npy"), ("dis_matrix.npy", "dis_matrix.npy")):
        try:
            member = find_member(zf, rf"{source_name}$")
        except FileNotFoundError:
            continue
        matrix = load_subset_matrix(zf, member, global_indices)
        np.save(out_dir / output_name, matrix)
        sidecars[output_name] = {
            "source_member": member,
            "shape": list(matrix.shape),
            "nonzero": int(np.count_nonzero(matrix)),
        }
        if output_name == "adj_matrix.npy":
            with (out_dir / "adj_mx.pkl").open("wb") as fh:
                pickle.dump(matrix, fh)
            sidecars["adj_mx.pkl"] = {
                "source_member": member,
                "format": "raw adjacency matrix pickle for BasicTS load_adj",
                "shape": list(matrix.shape),
                "nonzero": int(np.count_nonzero(matrix)),
            }
    summary["graph_sidecars"] = sidecars


def prepare_area(
    zf: zipfile.ZipFile,
    meta: pd.DataFrame,
    incidents: pd.DataFrame,
    args: argparse.Namespace,
    months: Sequence[int],
    split_ratio: Sequence[float],
) -> Path:
    selected = meta[area_mask(meta, args.area)].copy()
    if args.sensor_type != "all":
        selected = selected[selected["Type"] == args.sensor_type].copy()
    selected = selected.sort_values("global_index").reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No sensors for area={args.area}, sensor_type={args.sensor_type}")

    name = dataset_name(args.area, args.year, months, args.sensor_type)
    out_dir = args.output_root / name
    if not args.overwrite and (out_dir / "data.npz").exists() and (out_dir / "index.npz").exists():
        print(f"[skip] {name} already exists. Use --overwrite to rewrite.", flush=True)
        return out_dir

    print(
        f"[area] {args.area}: sensors={len(selected)}, sensor_type={args.sensor_type}, "
        f"filter={AREA_FILTERS[args.area]['filters']}",
        flush=True,
    )
    times = build_time_index(args.year, months)
    active_incidents = incidents[
        (incidents["dt"] >= times[0]) & (incidents["dt"] < times[-1] + pd.Timedelta(minutes=5))
    ].copy()
    candidate_meta = selected if args.match_scope == "subset" else meta
    matched = official_style_match(active_incidents, candidate_meta, args.distance_threshold)
    selected_indices = set(selected["global_index"].astype(int))
    matched = matched[matched["global_index"].astype(int).isin(selected_indices)].copy()

    global_indices = selected["global_index"].to_numpy(dtype=np.int64)
    flow = load_county_flow(
        zf,
        args.year,
        months,
        global_indices,
        args.traffic_channel,
        total_nodes=len(meta),
    )
    flow, missing_summary = fill_missing_flow(flow, args.fill_missing)
    if len(times) != flow.shape[0]:
        raise ValueError(f"Time index length {len(times)} does not match flow length {flow.shape[0]}")

    data = add_time_features(flow, times)
    node_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(global_indices)}
    fill_event_channel(data, matched, node_lookup, times, args.event_window_slots)
    index = build_index(data.shape[0], args.input_len, args.output_len, split_ratio)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "data.npz", data=data)
    np.savez_compressed(out_dir / "index.npz", **index)
    selected.to_csv(out_dir / "sensor_meta_feature.csv", index=False)
    matched.to_csv(out_dir / "matched_incidents.csv", index=False)

    summary = {
        "dataset": name,
        "area": args.area,
        "area_description": AREA_FILTERS[args.area]["description"],
        "area_filter": AREA_FILTERS[args.area]["filters"],
        "sensor_type": args.sensor_type,
        "num_nodes": int(data.shape[1]),
        "num_timesteps": int(data.shape[0]),
        "features": ["flow", "time_of_day", "day_of_week", "accident_binary"],
        "traffic_channel": args.traffic_channel,
        "year": args.year,
        "months": list(months),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "split_ratio": list(split_ratio),
        "split_sizes": {key: int(len(value)) for key, value in index.items()},
        "official_matching": {
            "source": "XTraffic/process/traffic_incident_match.py",
            "rule": "same Fwy, nearest Abs PM within distance threshold",
            "distance_threshold": args.distance_threshold,
            "match_scope": args.match_scope,
        },
        "adapter_choices": {
            "event_types": args.event_types,
            "accident_types": list(ACCIDENT_TYPES) if args.event_types == "accident" else "all",
            "event_window_slots": args.event_window_slots,
            "fill_missing": missing_summary,
        },
        "paper_reproduction_note": (
            "TraffiDent Section 4.2 says San Bernardino (561 mainline sensors), "
            "while Table 3 and Appendix A.8 say D5/Monterey. In the released "
            "metadata, District 5 has 565 sensors and 421 Mainline sensors; "
            "San Bernardino has 893 sensors and 452 Mainline sensors."
        ),
        "matched_incidents": int(len(matched)),
        "event_active_slots": int(data[:, :, 3].sum()),
    }
    maybe_write_graph_sidecars(zf, out_dir, global_indices, summary)
    with (out_dir / "preprocess_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(
        f"[done] {name}: data={data.shape}, matched={len(matched)}, "
        f"event_slot_nodes={int(data[:, :, 3].sum())}, out={out_dir}",
        flush=True,
    )
    return out_dir


def main() -> None:
    args = parse_args()
    months = parse_csv_ints(args.months)
    split_ratio = parse_csv_floats(args.split_ratio)
    args.output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        meta = load_metadata(zf)
        incidents = load_incidents(zf, args.year, args.event_types)
        print(
            f"[source] sensors={len(meta)}, incidents={len(incidents)}, "
            f"year={args.year}, months={months}",
            flush=True,
        )
        prepare_area(zf, meta, incidents, args, months, split_ratio)


if __name__ == "__main__":
    main()
