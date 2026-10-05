
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
INPUT_PATH = Path("data/wzdx_analysis.csv")
OUTPUT_PATH = Path("data/wzdx_features.csv")

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("Input shape:", df.shape)

required_columns = [
    "start_time", "end_time",
    "start_lon", "start_lat",
    "end_lon", "end_lat",
    "centroid_lon", "centroid_lat",
    "geometry_length_km", "duration_hours",
    "duration_days", "vehicle_impact",
    "vehicle_impact_score", "work_zone_type",
    "workers_present",
    "start_date_verified", "end_date_verified",
    "start_position_verified", "end_position_verified",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# --------------------------------------------------
# 1. Temporal feature engineering
# --------------------------------------------------
start = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
end = pd.to_datetime(df["end_time"], utc=True, errors="coerce")

df["timestamps_valid"] = (start.notna() & end.notna()).astype(int)

df["timestamps_ordered"] = (
    (end >= start) & start.notna() & end.notna()
).astype(int)

# Use the actual timestamps to check the existing duration.
actual_duration_hours = (end - start).dt.total_seconds() / 3600

df["duration_consistency_error_hours"] = (
    actual_duration_hours - df["duration_hours"]
).abs()

df["duration_consistent"] = (
    df["timestamps_valid"].eq(1)
    & df["timestamps_ordered"].eq(1)
    & df["duration_consistency_error_hours"].le(1.0)
).astype(int)

df["duration_valid"] = (
    df["duration_hours"].notna()
    & df["duration_hours"].gt(0)
    & df["timestamps_ordered"].eq(1)
).astype(int)

# A flag for long durations, not proof of an error or attack.
df["duration_over_365_days"] = (
    df["duration_days"] > 365
).astype(int)

# Additional time features.
df["start_year_utc"] = start.dt.year
df["end_hour_utc"] = end.dt.hour
df["end_day_of_week_utc"] = end.dt.dayofweek
df["end_month_utc"] = end.dt.month

# Cyclical encodings preserve the circular nature of time.
df["start_hour_sin"] = np.sin(
    2 * np.pi * df["start_hour_utc"] / 24
)
df["start_hour_cos"] = np.cos(
    2 * np.pi * df["start_hour_utc"] / 24
)

df["start_day_sin"] = np.sin(
    2 * np.pi * df["start_day_of_week_utc"] / 7
)
df["start_day_cos"] = np.cos(
    2 * np.pi * df["start_day_of_week_utc"] / 7
)

# --------------------------------------------------
# 2. Spatial feature engineering
# --------------------------------------------------
# Straight-line distance between the start and end coordinates.
# Haversine formula; coordinates are in degrees.

lon1 = np.radians(df["start_lon"])
lat1 = np.radians(df["start_lat"])
lon2 = np.radians(df["end_lon"])
lat2 = np.radians(df["end_lat"])

delta_lon = lon2 - lon1
delta_lat = lat2 - lat1

a = (
    np.sin(delta_lat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(delta_lon / 2) ** 2
)

a = a.clip(0, 1)

df["endpoint_straight_distance_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(a))
)

# Compare geometry length with straight-line distance.
# A ratio near 1 means the geometry is nearly straight.
# Very large ratios may indicate a winding geometry or data issue.
df["geometry_to_straight_distance_ratio"] = np.where(
    df["endpoint_straight_distance_km"] > 0,
    df["geometry_length_km"]
    / df["endpoint_straight_distance_km"],
    np.nan,
)

# Basic coordinate validity check.
df["coordinates_valid"] = (
    df["start_lon"].between(-180, 180)
    & df["end_lon"].between(-180, 180)
    & df["start_lat"].between(-90, 90)
    & df["end_lat"].between(-90, 90)
    & df["centroid_lon"].between(-180, 180)
    & df["centroid_lat"].between(-90, 90)
).astype(int)

# --------------------------------------------------
# 3. Trust and worker-presence features
# --------------------------------------------------
boolean_columns = [
    "start_date_verified",
    "end_date_verified",
    "start_position_verified",
    "end_position_verified",
    "workers_present",
]

for col in boolean_columns:
    df[f"{col}_feature"] = (
        df[col].astype("boolean").astype("Int64")
    )

# Overall verification count: 0 to 4.
verification_features = [
    "start_date_verified_feature",
    "end_date_verified_feature",
    "start_position_verified_feature",
    "end_position_verified_feature",
]

df["verified_fields_count"] = (
    df[verification_features].sum(axis=1)
)

# --------------------------------------------------
# 4. Encode selected categorical features
# --------------------------------------------------
categorical_columns = [
    "vehicle_impact",
    "work_zone_type",
    "location_method",
    "worker_presence_method",
    "worker_presence_confidence",
    "geometry_type",
]

for col in categorical_columns:
    df[col] = df[col].fillna("unknown").astype(str)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    prefix=categorical_columns,
    dtype=int,
)

# --------------------------------------------------
# 5. Save without modifying the source dataset
# --------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature engineering complete.")
print("Output:", OUTPUT_PATH)
print("Output shape:", df.shape)
print("\nNew feature columns:")

original_columns = pd.read_csv(INPUT_PATH, nrows=0).columns.tolist()

for col in df.columns:
    if col not in original_columns:
        print(" -", col)

print("\nDuration consistency:")
print(df["duration_consistent"].value_counts(dropna=False))

print("\nDuration validity:")
print(df["duration_valid"].value_counts(dropna=False))

print("\nLong-duration records:")
print(df["duration_over_365_days"].value_counts(dropna=False))
