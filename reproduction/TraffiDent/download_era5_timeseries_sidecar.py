#!/usr/bin/env python3
"""Download ERA5 point time series for the grid points in a TraffiDent sidecar.

This uses the CDS ARCO-backed time-series catalogue entry:
`reanalysis-era5-single-levels-timeseries`.

It preserves the data as one raw CSV per ERA5 grid point.  It does not join
weather into traffic tensors.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_VARIABLES = [
    "total_precipitation",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 time-series CSVs for sidecar ERA5 grid points."
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/era5_sidecar_2023q1_counties"),
        help="Root produced by prepare_era5_sidecar.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for raw CSVs. Defaults to sidecar-root/raw_era5_timeseries.",
    )
    parser.add_argument(
        "--date",
        default="2023-01-01/2023-04-01",
        help="CDS date range, e.g. 2023-01-01/2023-01-02.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated variables supported by the time-series dataset.",
    )
    parser.add_argument(
        "--max-grids",
        type=int,
        default=None,
        help="Limit number of unique ERA5 grids, useful for smoke tests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per grid before marking it failed.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=int,
        default=300,
        help="Seconds to sleep after a queued/temporary CDS rejection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download CSVs that already exist.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_unique_grids(sidecar_root: Path) -> List[Dict[str, Any]]:
    index_path = sidecar_root / "index" / "sensor_to_era5_grid_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing sidecar grid index: {index_path}")
    grids: Dict[str, Dict[str, Any]] = {}
    with index_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            grid_id = row["era5_grid_id"]
            if grid_id not in grids:
                grids[grid_id] = {
                    "era5_grid_id": grid_id,
                    "era5_lat": float(row["era5_lat"]),
                    "era5_lon": float(row["era5_lon"]),
                    "sensor_count": 0,
                }
            grids[grid_id]["sensor_count"] += 1
    return sorted(grids.values(), key=lambda item: item["era5_grid_id"])


def is_temporary_cds_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "temporarily limited" in lowered
        or "too many" in lowered
        or "queued requests" in lowered
        or "timeout" in lowered
    )


def download_one_grid(
    client: Any,
    grid: Dict[str, Any],
    variables: List[str],
    date_range: str,
    target: Path,
    retries: int,
    retry_sleep: int,
    overwrite: bool,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "era5_grid_id": grid["era5_grid_id"],
        "target": str(target),
        "latitude": grid["era5_lat"],
        "longitude": grid["era5_lon"],
        "sensor_count": grid["sensor_count"],
        "attempts": 0,
        "downloaded": False,
        "blocked_reason": None,
    }
    if target.exists() and not overwrite:
        status.update(
            {
                "downloaded": True,
                "blocked_reason": "target_exists",
                "size_bytes": target.stat().st_size,
            }
        )
        return status

    request = {
        "variable": variables,
        "location": {
            "latitude": grid["era5_lat"],
            "longitude": grid["era5_lon"],
        },
        "date": [date_range],
        "data_format": "csv",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        status["attempts"] = attempt
        try:
            client.retrieve("reanalysis-era5-single-levels-timeseries", request, str(target))
            status["downloaded"] = target.exists()
            if target.exists():
                status["size_bytes"] = target.stat().st_size
            status["blocked_reason"] = None
            return status
        except Exception as exc:
            message = str(exc)
            status.update(
                {
                    "downloaded": False,
                    "blocked_reason": "cdsapi_error",
                    "error_type": type(exc).__name__,
                    "error_message": message,
                }
            )
            if attempt < retries and is_temporary_cds_error(message):
                time.sleep(retry_sleep)
                continue
            return status
    return status


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("cdsapi") is None:
        raise RuntimeError("cdsapi is not installed in this Python environment.")

    import cdsapi  # type: ignore

    variables = parse_csv_list(args.variables)
    output_dir = args.output_dir or (args.sidecar_root / "raw_era5_timeseries")
    status_path = args.sidecar_root / "era5_timeseries_download_status.json"
    grids = load_unique_grids(args.sidecar_root)
    if args.max_grids is not None:
        grids = grids[: args.max_grids]

    client = cdsapi.Client()
    statuses = []
    for grid in grids:
        target = output_dir / f"{grid['era5_grid_id']}.csv"
        status = download_one_grid(
            client=client,
            grid=grid,
            variables=variables,
            date_range=args.date,
            target=target,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            overwrite=args.overwrite,
        )
        statuses.append(status)
        write_json(
            status_path,
            {
                "dataset": "reanalysis-era5-single-levels-timeseries",
                "date": args.date,
                "variables": variables,
                "output_dir": str(output_dir),
                "completed_grids": len(statuses),
                "total_grids": len(grids),
                "downloaded_grids": sum(item.get("downloaded", False) for item in statuses),
                "in_progress": len(statuses) < len(grids),
                "per_grid": statuses,
            },
        )

    final_status = {
        "dataset": "reanalysis-era5-single-levels-timeseries",
        "date": args.date,
        "variables": variables,
        "output_dir": str(output_dir),
        "completed_grids": len(statuses),
        "total_grids": len(grids),
        "downloaded_grids": sum(item.get("downloaded", False) for item in statuses),
        "in_progress": False,
        "per_grid": statuses,
    }
    write_json(status_path, final_status)
    print(f"Wrote status: {status_path}")
    print(f"Downloaded {final_status['downloaded_grids']}/{final_status['total_grids']} grid CSVs")


if __name__ == "__main__":
    main()
