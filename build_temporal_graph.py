
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = BASE_DIR / "data" / "wzdx_features.csv"
EDGES_PATH = BASE_DIR / "data" / "graph" / "edges.csv"

OUTPUT_DIR = BASE_DIR / "data" / "new_temporal_graph"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"

# Fixed interval: one UTC calendar day
INTERVAL = pd.Timedelta(days=1)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    for path in [FEATURES_PATH, EDGES_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    nodes = pd.read_csv(FEATURES_PATH, low_memory=False)
    edges = pd.read_csv(EDGES_PATH, low_memory=False)

    required_nodes = {"event_id", "start_time", "end_time"}
    required_edges = {"source", "target", "distance_meters"}

    missing_nodes = required_nodes - set(nodes.columns)
    missing_edges = required_edges - set(edges.columns)

    if missing_nodes:
        raise ValueError(
            f"Missing node columns: {sorted(missing_nodes)}"
        )

    if missing_edges:
        raise ValueError(
            f"Missing edge columns: {sorted(missing_edges)}"
        )

    if nodes.empty:
        raise ValueError("No WZDx events found.")

    if nodes["event_id"].isna().any():
        raise ValueError("Missing event IDs.")

    nodes["event_id"] = nodes["event_id"].astype(str)

    if nodes["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs found.")

    # Parse all timestamps in UTC
    for col in ["start_time", "end_time"]:
        nodes[col] = pd.to_datetime(
            nodes[col],
            errors="coerce",
            utc=True
        )

    if nodes[["start_time", "end_time"]].isna().any().any():
        raise ValueError("Invalid timestamps found.")

    if (nodes["start_time"] >= nodes["end_time"]).any():
        raise ValueError("Invalid event durations found.")

    # Validate edges
    if edges[["source", "target"]].isna().any().any():
        raise ValueError("Missing edge endpoints.")

    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    known_ids = set(nodes["event_id"])

    invalid_edges = (
        ~edges["source"].isin(known_ids)
        | ~edges["target"].isin(known_ids)
    )

    if invalid_edges.any():
        raise ValueError(
            f"{invalid_edges.sum()} edges contain unknown IDs."
        )

    if edges["distance_meters"].isna().any():
        raise ValueError("Missing edge distances.")

    # Remove self-loops
    edges = edges[
        edges["source"] != edges["target"]
    ].copy()

    # Normalize undirected edge pairs
    normalized = np.sort(
        edges[["source", "target"]].to_numpy(dtype=str),
        axis=1
    )

    edges["source"] = normalized[:, 0]
    edges["target"] = normalized[:, 1]

    # Remove duplicate edges and sort
    edges = (
        edges.drop_duplicates(["source", "target"])
        .sort_values(["source", "target"])
        .reset_index(drop=True)
    )

    nodes = nodes.sort_values("event_id").reset_index(drop=True)

    return nodes, edges


# ============================================================
# 3. GENERATE FIXED DAILY SNAPSHOTS
# ============================================================

def build_snapshots(nodes, edges):

    # First and last timestamps in the dataset
    first_day = nodes["start_time"].min().floor("D")
    final_end = nodes["end_time"].max()

    # Generate a fixed timeline, one timestamp per day
    dates = pd.date_range(
        start=first_day,
        end=final_end,
        freq="D",
        tz="UTC",
        inclusive="left"
    )

    if len(dates) == 0:
        raise ValueError("No valid snapshot dates.")

    # Convert timestamps to integer nanoseconds for fast comparisons
    start_ns = nodes["start_time"].astype("int64").to_numpy()
    end_ns = nodes["end_time"].astype("int64").to_numpy()

    event_ids = nodes["event_id"].to_numpy()

    # Map each event ID to its row position
    id_to_position = {
        event_id: i
        for i, event_id in enumerate(event_ids)
    }

    # Map edge endpoints to node positions
    src_positions = edges["source"].map(
        id_to_position
    ).to_numpy(dtype=int)

    dst_positions = edges["target"].map(
        id_to_position
    ).to_numpy(dtype=int)

    # For each edge, determine when BOTH events are active
    edge_start_ns = np.maximum(
        start_ns[src_positions],
        start_ns[dst_positions]
    )

    edge_end_ns = np.minimum(
        end_ns[src_positions],
        end_ns[dst_positions]
    )

    # Create the NEW output folder
    SNAPSHOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove stale snapshots only from the NEW folder
    for pattern in [
        "snapshot_*_nodes.csv",
        "snapshot_*_edges.csv"
    ]:
        for path in SNAPSHOTS_DIR.glob(pattern):
            path.unlink()

    records = []
    last_active_ids = set()

    interval_ns = int(INTERVAL.value)

    # ========================================================
    # 4. PROCESS EVERY CALENDAR DAY
    # ========================================================

    for number, day_start in enumerate(dates):

        start_value = int(day_start.value)
        end_value = start_value + interval_ns

        # A node is included if its event is active
        # at any point during this calendar day
        active_node_mask = (
            (start_ns < end_value)
            & (end_ns > start_value)
        )

        # An edge is included only when BOTH events
        # overlap in time during this day
        active_edge_mask = (
            (edge_start_ns < edge_end_ns)
            & (edge_start_ns < end_value)
            & (edge_end_ns > start_value)
        )

        active_nodes = nodes.loc[active_node_mask]
        active_edges = edges.loc[active_edge_mask]

        active_ids = set(event_ids[active_node_mask])

        snapshot_id = f"snapshot_{number:05d}"

        # Save nodes and edges, including unchanged days
        active_nodes.to_csv(
            SNAPSHOTS_DIR / f"{snapshot_id}_nodes.csv",
            index=False
        )

        active_edges.to_csv(
            SNAPSHOTS_DIR / f"{snapshot_id}_edges.csv",
            index=False
        )

        # Snapshot statistics
        records.append({
            "snapshot_id": snapshot_id,
            "interval_start_utc": day_start.isoformat(),
            "interval_end_utc": (
                day_start + INTERVAL
            ).isoformat(),
            "active_node_count": int(active_node_mask.sum()),
            "active_edge_count": int(active_edge_mask.sum()),
            "new_nodes_since_previous": len(
                active_ids - last_active_ids
            ),
            "removed_nodes_since_previous": len(
                last_active_ids - active_ids
            )
        })

        last_active_ids = active_ids

        # Progress update every 100 snapshots
        if (number + 1) % 100 == 0 or number == 0:
            print(
                f"Saved {number + 1}/{len(dates)} "
                f"daily snapshots ({day_start.date()})"
            )

    # ========================================================
    # 5. SAVE INDEX AND SUMMARY
    # ========================================================

    index = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index.to_csv(
        OUTPUT_DIR / "snapshot_index.csv",
        index=False
    )

    summary = {
        "input_event_count": len(nodes),
        "input_spatial_edge_count": len(edges),
        "saved_snapshot_count": len(index),
        "first_day_utc": dates[0].isoformat(),
        "last_day_utc": dates[-1].isoformat(),
        "snapshot_frequency": "1 day",
        "snapshot_timezone": "UTC",
        "node_rule": "event overlaps daily interval [start, end)",
        "edge_rule": (
            "both endpoint events simultaneously active "
            "during daily interval"
        ),
        "includes_unchanged_and_empty_days": True
    }

    with (
        OUTPUT_DIR / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return index


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def main():

    nodes, edges = load_data()

    print(
        f"Loaded {len(nodes):,} event nodes "
        f"and {len(edges):,} spatial edges."
    )

    print("Creating fixed daily temporal snapshots...")

    index = build_snapshots(nodes, edges)

    print("\nDaily temporal graph complete.")
    print(f"Daily snapshots: {len(index):,}")
    print(f"Output folder: {OUTPUT_DIR}")

    print("\nFirst snapshots:")
    print(index.head(5).to_string(index=False))

    print("\nNode and edge count statistics:")
    print(
        index[
            ["active_node_count", "active_edge_count"]
        ].describe().round(2).to_string()
    )


if __name__ == "__main__":
    main()
