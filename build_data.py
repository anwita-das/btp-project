#!/usr/bin/env python3
"""
Build a clean, tabular dataset from the City of Austin WZDx GeoJSON feed.

Default input:
    data/wzdx_raw.geojson
Default output:
    data/wzdx_clean.csv

Run from the btp_proj directory:
    python build_data.py

Optional paths:
    python build_data.py --input data/wzdx_raw.geojson --output data/wzdx_clean.csv
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def haversine_km(lon1, lat1, lon2, lat2):
    """Approximate distance in km between two longitude/latitude points."""
    if any(pd.isna(v) for v in (lon1, lat1, lon2, lat2)):
        return float("nan")

    earth_radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(min(1.0, math.sqrt(a)))


def first_or_none(value):
    """Return the first list item, or the original value if it is not a list."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def feature_to_row(feature):
    """Flatten one WZDx GeoJSON Feature into one CSV row."""
    props = feature.get("properties") or {}
    core = props.get("core_details") or {}
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or []

    # This Austin feed uses LineString geometry. Use the first and last points
    # as endpoints and their midpoint as a simple representative location.
    start_lon = start_lat = end_lon = end_lat = float("nan")
    centroid_lon = centroid_lat = float("nan")
    geometry_length_km = float("nan")

    valid_points = []
    if geometry.get("type") == "LineString" and isinstance(coordinates, list):
        for point in coordinates:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    lon, lat = float(point[0]), float(point[1])
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        valid_points.append((lon, lat))
                except (TypeError, ValueError):
                    continue

    if valid_points:
        start_lon, start_lat = valid_points[0]
        end_lon, end_lat = valid_points[-1]
        centroid_lon = sum(p[0] for p in valid_points) / len(valid_points)
        centroid_lat = sum(p[1] for p in valid_points) / len(valid_points)
        if len(valid_points) >= 2:
            geometry_length_km = sum(
                haversine_km(a[0], a[1], b[0], b[1])
                for a, b in zip(valid_points[:-1], valid_points[1:])
            )

    worker = props.get("worker_presence") or {}
    road_names = core.get("road_names") or []

    return {
        "event_id": feature.get("id"),
        "event_name": core.get("name"),
        "event_type": core.get("event_type"),
        "data_source_id": core.get("data_source_id"),
        "road_names": "; ".join(str(x) for x in road_names) if isinstance(road_names, list) else str(road_names),
        "direction": core.get("direction"),
        "description": core.get("description"),
        "start_time": props.get("start_date"),
        "end_time": props.get("end_date"),
        "start_date_verified": props.get("is_start_date_verified"),
        "end_date_verified": props.get("is_end_date_verified"),
        "start_position_verified": props.get("is_start_position_verified"),
        "end_position_verified": props.get("is_end_position_verified"),
        "location_method": props.get("location_method"),
        "work_zone_type": props.get("work_zone_type"),
        "vehicle_impact": props.get("vehicle_impact"),
        "workers_present": worker.get("are_workers_present"),
        "worker_presence_method": worker.get("method"),
        "worker_presence_confidence": worker.get("confidence"),
        "geometry_type": geometry.get("type"),
        "start_lon": start_lon,
        "start_lat": start_lat,
        "end_lon": end_lon,
        "end_lat": end_lat,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "geometry_length_km": geometry_length_km,
    }


def main():
    parser = argparse.ArgumentParser(description="Flatten and clean a WZDx GeoJSON feed.")
    parser.add_argument("--input", default="data/wzdx_raw.geojson", help="Input WZDx GeoJSON path")
    parser.add_argument("--output", default="data/wzdx_clean.csv", help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Place the downloaded file at data/wzdx_raw.geojson or pass --input."
        )

    with input_path.open("r", encoding="utf-8") as file:
        feed = json.load(file)

    if not isinstance(feed, dict) or not isinstance(feed.get("features"), list):
        raise ValueError("Expected a GeoJSON FeatureCollection with a 'features' list.")

    rows = [feature_to_row(feature) for feature in feed["features"]]
    df = pd.DataFrame(rows)

    # Parse timestamps as UTC. Invalid/missing timestamps become NaT.
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", utc=True)

    # Drop records without a stable ID, valid geometry location, or usable
    # start/end timestamps. Do not drop records merely because verification
    # flags are false: unverified does not necessarily mean incorrect.
    before = len(df)
    df = df.dropna(
        subset=["event_id", "centroid_lon", "centroid_lat", "start_time", "end_time"]
    ).copy()
    df = df.drop_duplicates(subset=["event_id"], keep="last").copy()

    # Reject impossible date ranges; retain zero-duration records for review.
    df = df[df["end_time"] >= df["start_time"]].copy()

    # Basic temporal features derived from the source timestamps.
    duration = df["end_time"] - df["start_time"]
    df["duration_hours"] = duration.dt.total_seconds() / 3600
    df["duration_days"] = duration.dt.total_seconds() / 86400
    df["start_hour_utc"] = df["start_time"].dt.hour
    df["start_day_of_week_utc"] = df["start_time"].dt.dayofweek
    df["start_month_utc"] = df["start_time"].dt.month

    # A useful, interpretable severity proxy based on the WZDx vehicle-impact
    # category. This is a feature, not a measured traffic-risk label.
    impact_scores = {
        "none": 0,
        "some-lanes-closed": 1,
        "all-lanes-closed": 2,
        "alternating-one-way": 1,
        "some-lanes-closed-merge-left": 1,
        "some-lanes-closed-merge-right": 1,
        "some-lanes-closed-split": 1,
        "some-lanes-closed-shift-left": 1,
        "some-lanes-closed-shift-right": 1,
        "some-lanes-closed-follow-detour": 2,
        "all-lanes-closed-follow-detour": 2,
        "unknown": 0,
    }
    df["vehicle_impact_score"] = (
        df["vehicle_impact"].fillna("unknown").astype(str).str.lower().map(impact_scores).fillna(0).astype(int)
    )

    # Keep original categorical/source fields; downstream feature engineering
    # can encode them without losing the human-readable values.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    print(f"Input records:  {len(rows)}")
    print(f"Output records: {len(df)}")
    print(f"Removed during cleaning: {before - len(df)}")
    print(f"Saved cleaned dataset: {output_path}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
    print("\nNote: vehicle_impact_score is only a simple feature proxy, not ground-truth traffic risk.")


if __name__ == "__main__":
    main()
