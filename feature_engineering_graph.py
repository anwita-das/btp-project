
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = BASE_DIR / "data" / "wzdx_features.csv"
EDGES_PATH = BASE_DIR / "data" / "graph" / "edges.csv"

OUTPUT_PATH = BASE_DIR / "data" / "wzdx_graph_features.csv"


def find_edge_columns(edges):
    """Identify the source and target columns in edges.csv."""

    candidates = [
        ("source", "target"),
        ("source_event_id", "target_event_id"),
        ("source_id", "target_id"),
        ("from_event_id", "to_event_id"),
        ("from", "to"),
    ]

    for source_col, target_col in candidates:
        if source_col in edges.columns and target_col in edges.columns:
            return source_col, target_col

    raise ValueError(
        "Could not identify edge endpoint columns.\n"
        f"Available columns: {list(edges.columns)}\n"
        "Set source_col and target_col manually in the script."
    )


def main():
    # Load existing node features and spatial edges
    nodes = pd.read_csv(FEATURES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    required = {
        "event_id",
        "start_time",
        "end_time",
        "duration_hours",
        "vehicle_impact_score",
        "verified_fields_count",
        "workers_present_feature",
        "geometry_length_km",
    }

    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing node columns: {sorted(missing)}")

    if nodes["event_id"].isna().any():
        raise ValueError("event_id contains missing values.")

    if nodes["event_id"].astype(str).duplicated().any():
        raise ValueError("event_id contains duplicate values.")

    nodes["event_id"] = nodes["event_id"].astype(str)

    source_col, target_col = find_edge_columns(edges)

    # Normalize edge endpoints
    edges = edges[[source_col, target_col]].copy()
    edges.columns = ["source", "target"]

    edges = edges.dropna()
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    # Remove self-loops and duplicate undirected edges
    edges = edges[edges["source"] != edges["target"]].copy()

    edges["_left"] = edges[["source", "target"]].min(axis=1)
    edges["_right"] = edges[["source", "target"]].max(axis=1)
    edges = edges.drop_duplicates(["_left", "_right"])

    valid_ids = set(nodes["event_id"])
    invalid = (
        ~edges["source"].isin(valid_ids)
        | ~edges["target"].isin(valid_ids)
    )

    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} edges reference unknown event IDs."
        )

    edges = edges[["source", "target"]]

    # --------------------------------------------------
    # 1. BUILD UNDIRECTED NEIGHBOR RELATIONSHIPS
    # --------------------------------------------------

    neighbors = pd.concat(
        [
            edges.rename(
                columns={"source": "event_id", "target": "neighbor_id"}
            ),
            edges.rename(
                columns={"target": "event_id", "source": "neighbor_id"}
            ),
        ],
        ignore_index=True,
    )

    # Include events with zero neighbors
    degree = neighbors.groupby("event_id").size()

    nodes["neighbor_count"] = (
        nodes["event_id"].map(degree).fillna(0).astype(int)
    )

    # --------------------------------------------------
    # 2. AGGREGATE NEIGHBOR FEATURES
    # --------------------------------------------------

    neighbor_attributes = nodes[
        [
            "event_id",
            "duration_hours",
            "vehicle_impact_score",
            "verified_fields_count",
            "workers_present_feature",
            "geometry_length_km",
        ]
    ].rename(
        columns={
            "event_id": "neighbor_id",
            "duration_hours": "_duration",
            "vehicle_impact_score": "_impact",
            "verified_fields_count": "_verified",
            "workers_present_feature": "_workers",
            "geometry_length_km": "_geometry_length",
        }
    )

    neighbor_data = neighbors.merge(
        neighbor_attributes,
        on="neighbor_id",
        how="left",
        validate="many_to_one",
    )

    grouped = neighbor_data.groupby("event_id")

    aggregates = grouped.agg(
        neighbor_mean_duration_hours=("_duration", "mean"),
        neighbor_max_duration_hours=("_duration", "max"),
        neighbor_mean_vehicle_impact=("_impact", "mean"),
        neighbor_mean_verified_fields=("_verified", "mean"),
        neighbor_workers_present_count=("_workers", "sum"),
        neighbor_mean_geometry_length_km=("_geometry_length", "mean"),
    )

    nodes = nodes.merge(
        aggregates,
        left_on="event_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    # No neighbors means no available neighbor average.
    # Keep these averages as NaN rather than inventing a value.
    # The neighbor count remains zero.

    # --------------------------------------------------
    # 3. TEMPORAL OVERLAP WITH SPATIAL NEIGHBORS
    # --------------------------------------------------

    nodes["start_time"] = pd.to_datetime(
        nodes["start_time"], utc=True, errors="coerce"
    )
    nodes["end_time"] = pd.to_datetime(
        nodes["end_time"], utc=True, errors="coerce"
    )

    times = nodes.set_index("event_id")[["start_time", "end_time"]]

    # Attach each event's timestamps to its neighbor pairs
    pairs = neighbors.merge(
        times.rename(
            columns={
                "start_time": "_start",
                "end_time": "_end",
            }
        ),
        left_on="event_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    pairs = pairs.merge(
        times.rename(
            columns={
                "start_time": "_neighbor_start",
                "end_time": "_neighbor_end",
            }
        ),
        left_on="neighbor_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    valid_times = (
        pairs["_start"].notna()
        & pairs["_end"].notna()
        & pairs["_neighbor_start"].notna()
        & pairs["_neighbor_end"].notna()
    )

    # Half-open intervals: [start_time, end_time)
    overlaps = (
        valid_times
        & (pairs["_start"] < pairs["_neighbor_end"])
        & (pairs["_neighbor_start"] < pairs["_end"])
    )

    pairs["_overlap"] = overlaps.astype(int)

    overlap_count = pairs.groupby("event_id")["_overlap"].sum()

    nodes["overlapping_time_neighbor_count"] = (
        nodes["event_id"].map(overlap_count).fillna(0).astype(int)
    )

    nodes["overlapping_time_neighbor_fraction"] = (
        nodes["overlapping_time_neighbor_count"]
        / nodes["neighbor_count"].replace(0, float("nan"))
    )

    # --------------------------------------------------
    # 4. SAVE SEPARATELY
    # --------------------------------------------------

    nodes.to_csv(OUTPUT_PATH, index=False)

    new_features = [
        "neighbor_count",
        "neighbor_mean_duration_hours",
        "neighbor_max_duration_hours",
        "neighbor_mean_vehicle_impact",
        "neighbor_mean_verified_fields",
        "neighbor_workers_present_count",
        "neighbor_mean_geometry_length_km",
        "overlapping_time_neighbor_count",
        "overlapping_time_neighbor_fraction",
    ]

    print(f"Input rows: {len(nodes)}")
    print(f"Unique spatial edges: {len(edges)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\nNew features:")
    print(nodes[new_features].describe().round(3))


if __name__ == "__main__":
    main()