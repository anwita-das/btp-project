import pandas as pd

df = pd.read_csv("data/wzdx_clean.csv", low_memory=False)

# Inspect records with zero or negative duration
duration = pd.to_numeric(df["duration_hours"], errors="coerce")
problem_rows = df.loc[duration <= 0]

columns = [
    "event_id",
    "event_name",
    "road_names",
    "start_time",
    "end_time",
    "duration_hours",
    "start_date_verified",
    "end_date_verified",
]

columns = [c for c in columns if c in df.columns]

print("\n=== ZERO OR NEGATIVE DURATION RECORDS ===")
print(problem_rows[columns].to_string(index=False))

# Inspect the three potential duplicate combinations
duplicate_cols = [
    "road_names",
    "start_time",
    "end_time",
    "centroid_lat",
    "centroid_lon",
]

duplicate_mask = df.duplicated(
    subset=duplicate_cols,
    keep=False,
)

print("\n=== POSSIBLE DUPLICATE RECORDS ===")
print(
    df.loc[duplicate_mask, columns]
    .sort_values(["road_names", "start_time"])
    .to_string(index=False)
)