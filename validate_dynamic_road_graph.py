
from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
GRAPH = BASE / "data" / "dynamic_road_graph"

nodes = pd.read_csv(GRAPH / "road_graph_nodes.csv")
edges = pd.read_csv(GRAPH / "road_graph_edges.csv")
index = pd.read_csv(GRAPH / "snapshot_index.csv")

snapshot_dir = GRAPH / "snapshots"

print("Checking static graph...")

assert nodes["directed_segment_id"].is_unique
assert len(nodes) == 243014

valid_ids = set(nodes["directed_segment_id"].astype(str))

assert edges["source"].astype(str).isin(valid_ids).all()
assert edges["target"].astype(str).isin(valid_ids).all()

print("Static nodes:", len(nodes))
print("Static edges:", len(edges))

print("\nChecking daily snapshots...")

required = [
    "date",
    "directed_segment_id",
    "active_event_count",
    "high_confidence_event_count",
    "medium_confidence_event_count",
    "max_vehicle_impact_score",
    "mean_vehicle_impact_score",
    "all_lanes_closed_count",
    "some_lanes_closed_count",
    "workers_present_count",
    "mean_event_duration_hours",
    "mean_verified_fields_count",
    "mean_event_geometry_length_km"
]

total_rows = 0
total_files = 0
all_node_ids_valid = True
all_values_valid = True
duplicate_rows = 0

for _, row in index.iterrows():

    filename = row["file"]
    path = snapshot_dir / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")

    if df["directed_segment_id"].duplicated().any():
        duplicate_rows += int(
            df["directed_segment_id"].duplicated().sum()
        )

    ids = set(df["directed_segment_id"].astype(str))

    if not ids.issubset(valid_ids):
        all_node_ids_valid = False

    numeric = df[required[2:]]

    if numeric.isna().any().any():
        all_values_valid = False

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        all_values_valid = False

    if (df["active_event_count"] < 1).any():
        all_values_valid = False

    total_rows += len(df)
    total_files += 1

print("Snapshot files:", total_files)
print("Snapshot rows:", total_rows)
print("Duplicate node rows:", duplicate_rows)
print("All node IDs valid:", all_node_ids_valid)
print("All feature values valid:", all_values_valid)

assert duplicate_rows == 0
assert all_node_ids_valid
assert all_values_valid
assert total_files == len(index)

result = {
    "static_nodes": len(nodes),
    "static_edges": len(edges),
    "snapshots": total_files,
    "daily_feature_rows": total_rows,
    "duplicate_node_rows": duplicate_rows,
    "all_node_ids_valid": all_node_ids_valid,
    "all_feature_values_valid": all_values_valid,
    "first_date": str(index.iloc[0]["date"]),
    "last_date": str(index.iloc[-1]["date"])
}

with open(GRAPH / "validation_summary.json", "w") as f:
    json.dump(result, f, indent=4)

print("\nVALIDATION PASSED")
print(json.dumps(result, indent=4))