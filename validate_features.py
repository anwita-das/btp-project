
import pandas as pd

df = pd.read_csv("data/wzdx_features.csv")

print("Dataset shape:", df.shape)

print("\nMissing values in generated features:")
cols = [
    "duration_consistency_error_hours",
    "endpoint_straight_distance_km",
    "geometry_to_straight_distance_ratio",
    "coordinates_valid",
    "verified_fields_count",
]

print(df[cols].isna().sum())

print("\nCoordinate validity:")
print(df["coordinates_valid"].value_counts(dropna=False))

print("\nStraight-line distance (km):")
print(df["endpoint_straight_distance_km"].describe())

print("\nGeometry-to-distance ratio:")
print(df["geometry_to_straight_distance_ratio"].describe())

print("\nVerification count:")
print(df["verified_fields_count"].value_counts().sort_index())

print("\nVehicle-impact feature totals:")
impact_cols = [
    c for c in df.columns if c.startswith("vehicle_impact_")
]
print(df[impact_cols].sum())

print("\nDuplicate event IDs:", df["event_id"].duplicated().sum())
