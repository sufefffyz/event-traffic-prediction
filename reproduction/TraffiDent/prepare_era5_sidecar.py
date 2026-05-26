#!/usr/bin/env python3
"""Prepare a shallow ERA5 sidecar for TraffiDent county data.

This script intentionally does not rewrite or aggregate traffic arrays.  It
creates an independent sidecar directory containing:

* a manifest that points to the existing traffic/event files;
* an ERA5 CDS request JSON for the traffic period and sensor bounding box;
* a sensor-to-ERA5-grid index for later shallow joins.

If --download is passed, the script also tries to call the CDS API.  Missing
cdsapi or missing credentials are recorded as status rather than modifying any
traffic data.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo


COUNTY_LABELS = {
    "LosAngeles": "Los Angeles",
    "Orange": "Orange",
    "Alameda": "Alameda",
    "ContraCosta": "Contra Costa",
}

DEFAULT_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_cloud_cover",
    "total_precipitation",
    "visibility",
]


@dataclass
class CountyDataset:
    slug: str
    name: str
    path: Path
    summary: Dict[str, Any]
    sensor_meta_path: Path
    matched_incidents_path: Path
    data_path: Path
    index_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a shallow ERA5 sidecar for TraffiDent county data."
    )
    parser.add_argument(
        "--basicts-root",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/basicts"),
        help="Root containing TraffiDent_* county BasicTS folders.",
    )
    parser.add_argument(
        "--xtraffic-zip",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/xtraffic.zip"),
        help="Original TraffiDent/XTraffic zip. Only referenced in the manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/yuzhang_fei/TraffiDent/era5_sidecar"),
        help="Independent output directory for ERA5 sidecar files.",
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--months",
        default="1,2,3",
        help="Comma-separated local-calendar months covered by the traffic view.",
    )
    parser.add_argument(
        "--counties",
        default="LosAngeles,Orange,Alameda,ContraCosta",
        help=f"Comma-separated county slugs. Choices: {','.join(COUNTY_LABELS)}",
    )
    parser.add_argument(
        "--timezone",
        default="America/Los_Angeles",
        help="Local timezone used to request the corresponding UTC ERA5 range.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.25,
        help="ERA5 single-level grid resolution in degrees.",
    )
    parser.add_argument(
        "--bbox-margin-deg",
        type=float,
        default=0.50,
        help="Margin added around all selected sensors before rounding to grid.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated ERA5 single-level variables.",
    )
    parser.add_argument(
        "--target-name",
        default="era5_single_levels_traffident_2023q1_california.nc",
        help="ERA5 NetCDF filename under output-root/raw_era5.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Try to download ERA5 via cdsapi after writing sidecar metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing ERA5 target file.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_months(value: str) -> List[int]:
    months = [int(part) for part in parse_csv_list(value)]
    if not months:
        raise ValueError("At least one month is required.")
    if any(month < 1 or month > 12 for month in months):
        raise ValueError(f"Months must be in 1..12, got {months}")
    return months


def period_label(months: Sequence[int]) -> str:
    if list(months) == [1, 2, 3]:
        return "Q1"
    if list(months) == list(range(1, 13)):
        return "FullYear"
    return "M" + "-".join(f"{month:02d}" for month in months)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def discover_datasets(
    basicts_root: Path,
    counties: Sequence[str],
    year: int,
    months: Sequence[int],
) -> List[CountyDataset]:
    datasets = []
    label = period_label(months)
    for slug in counties:
        if slug not in COUNTY_LABELS:
            raise ValueError(f"Unknown county slug {slug}; choices={sorted(COUNTY_LABELS)}")
        name = f"TraffiDent_{slug}_{year}{label}"
        path = basicts_root / name
        summary_path = path / "preprocess_summary.json"
        required = {
            "summary": summary_path,
            "sensor_meta": path / "sensor_meta_feature.csv",
            "matched_incidents": path / "matched_incidents.csv",
            "data": path / "data.npz",
            "index": path / "index.npz",
        }
        missing = [str(p) for p in required.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{name} is incomplete; missing {missing}")
        datasets.append(
            CountyDataset(
                slug=slug,
                name=name,
                path=path,
                summary=read_json(summary_path),
                sensor_meta_path=required["sensor_meta"],
                matched_incidents_path=required["matched_incidents"],
                data_path=required["data"],
                index_path=required["index"],
            )
        )
    return datasets


def read_sensor_rows(dataset: CountyDataset) -> List[Dict[str, str]]:
    with dataset.sensor_meta_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for idx, row in enumerate(reader):
            row = dict(row)
            row["node_order"] = str(idx)
            rows.append(row)
        return rows


def as_float(value: str, field: str, dataset: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}={value!r} in {dataset}") from exc


def round_to_grid(value: float, resolution: float) -> float:
    return round(round(value / resolution) * resolution, 6)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_sensor_grid_index(
    datasets: Sequence[CountyDataset],
    resolution: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    lats: List[float] = []
    lngs: List[float] = []
    for dataset in datasets:
        for sensor in read_sensor_rows(dataset):
            try:
                lat = as_float(sensor.get("Lat", ""), "Lat", dataset.name)
                lon = as_float(sensor.get("Lng", ""), "Lng", dataset.name)
            except ValueError:
                skipped_rows.append(
                    {
                        "dataset": dataset.name,
                        "county_slug": dataset.slug,
                        "node_order": sensor["node_order"],
                        "global_index": sensor.get("global_index", ""),
                        "station_id": sensor.get("station_id", ""),
                        "county": sensor.get("County", ""),
                        "lat": sensor.get("Lat", ""),
                        "lng": sensor.get("Lng", ""),
                        "reason": "missing_or_invalid_lat_lng",
                    }
                )
                continue
            era5_lat = round_to_grid(lat, resolution)
            era5_lon = round_to_grid(lon, resolution)
            lats.append(lat)
            lngs.append(lon)
            rows.append(
                {
                    "dataset": dataset.name,
                    "county_slug": dataset.slug,
                    "county": sensor.get("County", ""),
                    "node_order": sensor["node_order"],
                    "global_index": sensor.get("global_index", ""),
                    "station_id": sensor.get("station_id", ""),
                    "sensor_type": sensor.get("Type", ""),
                    "sensor_class": sensor.get("Sensor Type", ""),
                    "fwy": sensor.get("Fwy", ""),
                    "direction": sensor.get("Direction", ""),
                    "abs_pm": sensor.get("Abs PM", ""),
                    "lat": lat,
                    "lng": lon,
                    "era5_lat": era5_lat,
                    "era5_lon": era5_lon,
                    "era5_grid_id": f"era5_{era5_lat:.3f}_{era5_lon:.3f}",
                    "nearest_grid_distance_km": round(
                        haversine_km(lat, lon, era5_lat, era5_lon), 4
                    ),
                }
            )
    if not rows:
        raise ValueError("No sensors with valid Lat/Lng were found.")
    bounds = {
        "south": min(lats),
        "north": max(lats),
        "west": min(lngs),
        "east": max(lngs),
    }
    return rows, skipped_rows, bounds


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def local_period_bounds(year: int, months: Sequence[int]) -> tuple[datetime, datetime]:
    first_month = min(months)
    last_month = max(months)
    start = datetime(year, first_month, 1, 0, 0, 0)
    last_day = calendar.monthrange(year, last_month)[1]
    end = datetime(year, last_month, last_day, 23, 55, 0)
    return start, end


def utc_hour_bounds(
    start_local: datetime,
    end_local: datetime,
    timezone_name: str,
) -> tuple[datetime, datetime]:
    tz = ZoneInfo(timezone_name)
    start_utc = start_local.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    end_utc = end_local.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    start_utc = start_utc.replace(minute=0, second=0, microsecond=0)
    if end_utc.minute or end_utc.second or end_utc.microsecond:
        end_utc = end_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        end_utc = end_utc.replace(minute=0, second=0, microsecond=0)
    return start_utc, end_utc


def dates_between(start: datetime, end: datetime) -> List[datetime]:
    current = datetime(start.year, start.month, start.day)
    last = datetime(end.year, end.month, end.day)
    dates = []
    while current <= last:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def round_area(bounds: Dict[str, float], margin: float, resolution: float) -> Dict[str, float]:
    south = math.floor((bounds["south"] - margin) / resolution) * resolution
    west = math.floor((bounds["west"] - margin) / resolution) * resolution
    north = math.ceil((bounds["north"] + margin) / resolution) * resolution
    east = math.ceil((bounds["east"] + margin) / resolution) * resolution
    return {
        "north": round(north, 6),
        "west": round(west, 6),
        "south": round(south, 6),
        "east": round(east, 6),
    }


def build_era5_request(
    variables: Sequence[str],
    area: Dict[str, float],
    year: int,
    month: int,
    days: Sequence[int],
) -> Dict[str, Any]:
    times = [f"{hour:02d}:00" for hour in range(24)]
    return {
        "product_type": ["reanalysis"],
        "variable": list(variables),
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in days],
        "time": times,
        "area": [area["north"], area["west"], area["south"], area["east"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def build_era5_request_bundle(
    variables: Sequence[str],
    area: Dict[str, float],
    start_utc: datetime,
    end_utc: datetime,
    raw_era5_dir: Path,
    target_name: str,
) -> Dict[str, Any]:
    dates = dates_between(start_utc, end_utc)
    by_month: Dict[tuple[int, int], List[int]] = {}
    for dt in dates:
        by_month.setdefault((dt.year, dt.month), []).append(dt.day)

    target_stem = Path(target_name).stem
    requests = []
    for (year, month), day_values in sorted(by_month.items()):
        days = sorted(set(day_values))
        request = build_era5_request(variables, area, year, month, days)
        target_path = raw_era5_dir / f"{target_stem}_{year}{month:02d}.nc"
        requests.append(
            {
                "name": f"{year}-{month:02d}",
                "target_path": str(target_path),
                "utc_days": [f"{year:04d}-{month:02d}-{day:02d}" for day in days],
                "request": request,
            }
        )
    return {
        "dataset": "reanalysis-era5-single-levels",
        "note": (
            "Monthly requests avoid downloading unnecessary whole months while "
            "keeping raw ERA5 NetCDF files separate from traffic data."
        ),
        "requests": requests,
    }


def write_traffic_sources(path: Path, datasets: Sequence[CountyDataset]) -> List[Dict[str, Any]]:
    rows = []
    for dataset in datasets:
        rows.append(
            {
                "dataset": dataset.name,
                "county_slug": dataset.slug,
                "county": dataset.summary.get("county", ""),
                "num_nodes": dataset.summary.get("num_nodes", ""),
                "num_timesteps": dataset.summary.get("num_timesteps", ""),
                "features": "|".join(dataset.summary.get("features", [])),
                "data_npz": str(dataset.data_path),
                "index_npz": str(dataset.index_path),
                "sensor_meta_feature_csv": str(dataset.sensor_meta_path),
                "matched_incidents_csv": str(dataset.matched_incidents_path),
                "preprocess_summary_json": str(dataset.path / "preprocess_summary.json"),
            }
        )
    write_csv(path, rows)
    return rows


def has_cds_credentials() -> bool:
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return True
    return Path.home().joinpath(".cdsapirc").exists()


def attempt_download(
    request: Dict[str, Any],
    target: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "target": str(target),
        "attempted": False,
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
    if not has_cds_credentials():
        status["blocked_reason"] = "missing_cds_credentials"
        return status
    if importlib.util.find_spec("cdsapi") is None:
        status["blocked_reason"] = "missing_python_package_cdsapi"
        return status

    target.parent.mkdir(parents=True, exist_ok=True)
    status["attempted"] = True
    import cdsapi  # type: ignore

    client = cdsapi.Client()
    try:
        client.retrieve("reanalysis-era5-single-levels", request, str(target))
    except Exception as exc:  # CDS errors are best recorded for later reruns.
        message = str(exc)
        reason = "cdsapi_error"
        if "required licences not accepted" in message.lower():
            reason = "required_licences_not_accepted"
        status.update(
            {
                "downloaded": False,
                "blocked_reason": reason,
                "error_type": type(exc).__name__,
                "error_message": message,
            }
        )
        return status
    status["downloaded"] = target.exists()
    if target.exists():
        status["size_bytes"] = target.stat().st_size
    return status


def main() -> None:
    args = parse_args()
    months = parse_months(args.months)
    counties = parse_csv_list(args.counties)
    variables = parse_csv_list(args.variables)
    datasets = discover_datasets(args.basicts_root, counties, args.year, months)

    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_era5_dir = args.output_root / "raw_era5"
    index_dir = args.output_root / "index"
    manifest_path = args.output_root / "traffic_era5_sidecar_manifest.json"
    request_path = args.output_root / "era5_request.json"
    download_status_path = args.output_root / "era5_download_status.json"

    sensor_grid_rows, skipped_sensor_rows, sensor_bounds = build_sensor_grid_index(
        datasets, resolution=args.grid_resolution
    )
    area = round_area(sensor_bounds, args.bbox_margin_deg, args.grid_resolution)
    local_start, local_end = local_period_bounds(args.year, months)
    utc_start, utc_end = utc_hour_bounds(local_start, local_end, args.timezone)
    request_bundle = build_era5_request_bundle(
        variables,
        area,
        utc_start,
        utc_end,
        raw_era5_dir,
        args.target_name,
    )

    sensor_grid_path = index_dir / "sensor_to_era5_grid_index.csv"
    skipped_sensor_path = index_dir / "sensors_missing_location.csv"
    traffic_sources_path = index_dir / "traffic_sources.csv"
    write_csv(sensor_grid_path, sensor_grid_rows)
    if skipped_sensor_rows:
        write_csv(skipped_sensor_path, skipped_sensor_rows)
    traffic_sources = write_traffic_sources(traffic_sources_path, datasets)
    write_json(request_path, request_bundle)

    unique_grids = sorted({row["era5_grid_id"] for row in sensor_grid_rows})
    target_paths = [entry["target_path"] for entry in request_bundle["requests"]]
    manifest = {
        "created_by": Path(__file__).name,
        "policy": {
            "traffic_arrays": "referenced_only_not_rewritten",
            "traffic_resampling": "none",
            "era5_storage": "raw_cds_netcdf_if_downloaded",
            "join_depth": "shallow_manifest_plus_sensor_grid_index",
        },
        "traffic": {
            "source_zip": str(args.xtraffic_zip),
            "local_timezone_assumption": args.timezone,
            "local_regular_5min_period": {
                "start": local_start.isoformat(sep=" "),
                "end": local_end.isoformat(sep=" "),
                "note": (
                    "The existing BasicTS traffic view is a regular 5-minute "
                    "local-clock index. The sidecar does not insert/remove DST slots."
                ),
            },
            "datasets": traffic_sources,
        },
        "era5": {
            "dataset": "reanalysis-era5-single-levels",
            "target_paths": target_paths,
            "request_json": str(request_path),
            "variables": variables,
            "grid_resolution_degree": args.grid_resolution,
            "sensor_bounds": sensor_bounds,
            "request_area_north_west_south_east": [area["north"], area["west"], area["south"], area["east"]],
            "utc_hour_period": {
                "start": utc_start.isoformat(sep=" "),
                "end": utc_end.isoformat(sep=" "),
            },
        },
        "sidecar_files": {
            "sensor_to_era5_grid_index": str(sensor_grid_path),
            "sensors_missing_location": str(skipped_sensor_path) if skipped_sensor_rows else None,
            "traffic_sources": str(traffic_sources_path),
            "download_status": str(download_status_path),
        },
        "summary": {
            "num_datasets": len(datasets),
            "num_sensors_total": len(sensor_grid_rows) + len(skipped_sensor_rows),
            "num_sensors_mapped_to_era5_grid": len(sensor_grid_rows),
            "num_sensors_missing_location": len(skipped_sensor_rows),
            "num_unique_era5_grids": len(unique_grids),
            "unique_era5_grids_preview": unique_grids[:20],
        },
    }
    write_json(manifest_path, manifest)

    status = {
        "targets": target_paths,
        "download_requested": bool(args.download),
        "attempted": False,
        "downloaded": all(Path(path).exists() for path in target_paths),
        "blocked_reason": None,
    }
    if args.download:
        statuses = []
        for entry in request_bundle["requests"]:
            statuses.append(
                attempt_download(
                    entry["request"],
                    Path(entry["target_path"]),
                    overwrite=args.overwrite,
                )
            )
        status = {
            "download_requested": True,
            "targets": target_paths,
            "attempted": any(item.get("attempted") for item in statuses),
            "downloaded": all(item.get("downloaded") for item in statuses),
            "per_target": statuses,
        }
        blocked = sorted(
            {
                str(item.get("blocked_reason"))
                for item in statuses
                if item.get("blocked_reason")
            }
        )
        status["blocked_reason"] = "|".join(blocked) if blocked else None
    write_json(download_status_path, status)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote ERA5 request: {request_path}")
    print(f"Wrote sensor-grid index: {sensor_grid_path}")
    print(f"Wrote traffic sources: {traffic_sources_path}")
    print(f"Wrote download status: {download_status_path}")
    if status.get("blocked_reason"):
        print(f"ERA5 download status: {status['blocked_reason']}")
    elif status.get("downloaded"):
        print(f"ERA5 targets are available: {len(target_paths)} file(s)")


if __name__ == "__main__":
    main()
