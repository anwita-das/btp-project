
"""
Build change-driven temporal graph snapshots from WZDx data.

Inputs:
    data/wzdx_features.csv
    data/graph/edges.csv

Outputs:
    data/temporal_graph/snapshots/snapshot_00000_nodes.csv
    data/temporal_graph/snapshots/snapshot_00000_edges.csv
    ...
    data/temporal_graph/snapshot_index.csv

Each node represents a WZDx work-zone event.
Each edge represents spatial proximity between two events.

An event is active at time t when:
    start_time <= t < end_time

A snapshot is saved only when the active graph structure changes.
All timestamps are handled in UTC.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = BASE_DIR / "data" / "wzdx_features.csv"
EDGES_PATH = BASE_DIR / "data" / "graph" / "edges.csv"

OUTPUT_DIR = BASE_DIR / "data" / "temporal_graph"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"

# Half-open activity interval: start_time <= t < end_time
# This means an event ending at t is inactive at t.
# An event starting at t is active at t.

# If True, save the active node and edge CSVs for every
# distinct graph structure, even if only the active nodes'
# timestamps/features differ but the IDs and edges are unchanged.
SAVE_ONLY_STRUCTURAL_CHANGES = True


# ============================================================
# 2. LOAD AND VALIDATE INPUT DATA
# ============================================================

def load_data():
    """Load features and spatial edges, then validate columns."""

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_PATH}"
        )

    if not EDGES_PATH.exists():
        raise FileNotFoundError(
            f"Edges file not found: {EDGES_PATH}"
        )

    nodes = pd.read_csv(FEATURES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    required_node_columns = {
        "event_id",
        "start_time",
        "end_time",
    }

    required_edge_columns = {
        "source",
        "target",
        "distance_meters",
    }

    missing_nodes = required_node_columns - set(nodes.columns)
    missing_edges = required_edge_columns - set(edges.columns)

    if missing_nodes:
        raise ValueError(
            f"Missing columns in features CSV: {missing_nodes}"
        )

    if missing_edges:
        raise ValueError(
            f"Missing columns in edges CSV: {missing_edges}"
        )

    if nodes["event_id"].isna().any():
        raise ValueError("Some nodes have missing event_id values.")

    if nodes["event_id"].duplicated().any():
        raise ValueError(
            "Duplicate event_id values found in features CSV."
        )

    # Treat IDs consistently as strings when matching nodes to edges.
    nodes["event_id"] = nodes["event_id"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    # Parse timestamps and normalize them to UTC.
    nodes["start_time"] = pd.to_datetime(
        nodes["start_time"],
        utc=True,
        errors="coerce",
    )

    nodes["end_time"] = pd.to_datetime(
        nodes["end_time"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = (
        nodes["start_time"].isna()
        | nodes["end_time"].isna()
    )

    if invalid_timestamps.any():
        count = int(invalid_timestamps.sum())
        raise ValueError(
            f"{count} events have invalid start/end timestamps. "
            "Inspect the source CSV before building snapshots."
        )

    invalid_intervals = (
        nodes["start_time"] >= nodes["end_time"]
    )

    if invalid_intervals.any():
        count = int(invalid_intervals.sum())
        raise ValueError(
            f"{count} events have start_time >= end_time. "
            "Inspect these records before building snapshots."
        )

    # Edges must reference events present in the node table.
    known_ids = set(nodes["event_id"])

    unknown_endpoints = (
        ~edges["source"].isin(known_ids)
        | ~edges["target"].isin(known_ids)
    )

    if unknown_endpoints.any():
        count = int(unknown_endpoints.sum())
        raise ValueError(
            f"{count} edges reference unknown event IDs."
        )

    # Remove self-loops and duplicate undirected edges.
    edges = edges[edges["source"] != edges["target"]].copy()

    edges["_u"] = edges[["source", "target"]].min(axis=1)
    edges["_v"] = edges[["source", "target"]].max(axis=1)

    edges = edges.drop_duplicates(subset=["_u", "_v"])

    edges = edges.drop(columns=["_u", "_v"]).reset_index(drop=True)

    return nodes, edges


# ============================================================
# 3. PREPARE EVENT START/END TIMELINE
# ============================================================

def build_event_timeline(nodes):
    """
    Return unique timestamps at which at least one event starts
    or ends.

    No fixed 15-minute/hourly grid is created.
    """

    starts = pd.DataFrame({
        "timestamp": nodes["start_time"],
        "event_id": nodes["event_id"],
        "action": "start",
    })

    ends = pd.DataFrame({
        "timestamp": nodes["end_time"],
        "event_id": nodes["event_id"],
        "action": "end",
    })

    timeline = pd.concat(
        [starts, ends],
        ignore_index=True,
    )

    # Unique timestamps, ordered chronologically.
    timestamps = sorted(timeline["timestamp"].unique())

    return timestamps


# ============================================================
# 4. CREATE CHANGE-DRIVEN SNAPSHOTS
# ============================================================

def create_snapshots(nodes, edges, timestamps):
    """
    At each event boundary:
      1. Determine active events.
      2. Keep spatial edges whose endpoints are both active.
      3. Save a snapshot only if the graph structure changed.

    Structural identity is determined by:
      - active event IDs
      - active undirected edge pairs
    """

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare a fast lookup by event ID.
    nodes = nodes.set_index("event_id", drop=False)

    # Build edge endpoint arrays once.
    edge_sources = edges["source"].to_numpy()
    edge_targets = edges["target"].to_numpy()

    # Sort edge rows so snapshots have deterministic output.
    edges = edges.sort_values(
        ["source", "target"]
    ).reset_index(drop=True)

    edge_sources = edges["source"].to_numpy()
    edge_targets = edges["target"].to_numpy()

    # Event start/end maps.
    start_map = {}
    end_map = {}

    for event_id, row in nodes.iterrows():
        start_map.setdefault(row["start_time"], []).append(event_id)
        end_map.setdefault(row["end_time"], []).append(event_id)

    active_ids = set()

    previous_structure = None
    snapshot_records = []

    for timestamp in timestamps:
        timestamp = pd.Timestamp(timestamp)

        # Half-open intervals:
        # Events ending at this timestamp are no longer active.
        for event_id in end_map.get(timestamp, []):
            active_ids.discard(event_id)

        # Events starting at this timestamp become active.
        for event_id in start_map.get(timestamp, []):
            active_ids.add(event_id)

        # Deterministic node order.
        active_list = sorted(active_ids)

        # Identify edges with both endpoints active.
        if active_ids:
            source_active = pd.Series(edge_sources).isin(active_ids).to_numpy()
            target_active = pd.Series(edge_targets).isin(active_ids).to_numpy()
            active_edge_mask = source_active & target_active
            active_edges = edges.loc[active_edge_mask].copy()
        else:
            active_edges = edges.iloc[0:0].copy()

        # Canonical graph structure for change detection.
        structure = (
            tuple(active_list),
            tuple(
                sorted(
                    (
                        min(source, target),
                        max(source, target),
                    )
                    for source, target in zip(
                        active_edges["source"],
                        active_edges["target"],
                    )
                )
            ),
        )

        # Skip redundant snapshots if graph structure is identical.
        if (
            SAVE_ONLY_STRUCTURAL_CHANGES
            and structure == previous_structure
        ):
            continue

        snapshot_number = len(snapshot_records)
        snapshot_name = f"snapshot_{snapshot_number:05d}"

        # Save active nodes with all their original engineered features.
        if active_list:
            active_nodes = (
                nodes.loc[active_list]
                .reset_index(drop=True)
                .copy()
            )
        else:
            active_nodes = nodes.iloc[0:0].reset_index(drop=True).copy()

        # Remove internal timestamp columns only if you prefer
        # a leaner file; for now retain them for analysis.
        active_nodes.to_csv(
            SNAPSHOTS_DIR / f"{snapshot_name}_nodes.csv",
            index=False,
        )

        active_edges.to_csv(
            SNAPSHOTS_DIR / f"{snapshot_name}_edges.csv",
            index=False,
        )

        snapshot_records.append({
            "snapshot_id": snapshot_name,
            "timestamp_utc": timestamp.isoformat(),
            "active_node_count": len(active_nodes),
            "active_edge_count": len(active_edges),
            "new_nodes_since_previous": (
                len(active_ids)
                if previous_structure is None
                else len(active_ids - set(previous_structure[0]))
            ),
            "graph_structure_changed": True,
        })

        previous_structure = structure

        if snapshot_number % 100 == 0:
            print(
                f"Saved {snapshot_number + 1} snapshots; "
                f"current timestamp: {timestamp}"
            )

    return pd.DataFrame(snapshot_records)


# ============================================================
# 5. SAVE INDEX AND SUMMARY
# ============================================================

def main():
    print("Loading WZDx features and spatial graph...")

    nodes, edges = load_data()

    print(f"Events loaded: {len(nodes):,}")
    print(f"Spatial edges loaded: {len(edges):,}")

    timestamps = build_event_timeline(nodes)

    print(f"Unique event-boundary timestamps: {len(timestamps):,}")
    print("Creating change-driven snapshots...")

    index = create_snapshots(nodes, edges, timestamps)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_path = OUTPUT_DIR / "snapshot_index.csv"
    index.to_csv(index_path, index=False)

    summary = {
        "input_event_count": int(len(nodes)),
        "input_spatial_edge_count": int(len(edges)),
        "unique_event_boundary_timestamps": int(len(timestamps)),
        "saved_snapshot_count": int(len(index)),
        "earliest_event_start_utc": nodes["start_time"].min().isoformat(),
        "latest_event_end_utc": nodes["end_time"].max().isoformat(),
        "snapshot_policy": "event-boundary structural changes",
        "activity_interval": "start_time <= t < end_time",
        "timezone": "UTC",
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTemporal graph generation complete.")
    print(f"Snapshots saved: {len(index):,}")
    print(f"Snapshot index: {index_path}")
    print(f"Summary: {summary_path}")
    print(f"Snapshot files: {SNAPSHOTS_DIR}")

    if not index.empty:
        print("\nFirst snapshots:")
        print(index.head(5).to_string(index=False))

        print("\nSnapshot statistics:")
        print(
            index[
                ["active_node_count", "active_edge_count"]
            ].describe().to_string()
        )


if __name__ == "__main__":
    main()