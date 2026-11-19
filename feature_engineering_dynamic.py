"""Derive complementary event-level metrics from existing daily graph snapshots.

This script does not rebuild the graph. It summarizes daily event graph
structure into one row per event for exploratory analysis and downstream
feature selection. Aggregate features use the full available timeline, so
they are descriptive and should not be used as point-in-time predictors
without recomputing them using only data available at the prediction time.

Run from any directory with: python feature_engineering_dynamic.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "wzdx_graph_features.csv"
GRAPH_DIR = BASE_DIR / "data" / "new_temporal_graph"
INDEX_PATH = GRAPH_DIR / "snapshot_index.csv"
SNAPSHOTS_DIR = GRAPH_DIR / "snapshots"
OUTPUT_PATH = BASE_DIR / "data" / "wzdx_dynamic_features.csv"


def normalized_edges(edges: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    required = {"source", "target", "distance_meters"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"{snapshot_id} is missing edge columns: {sorted(missing)}")

    result = edges[["source", "target", "distance_meters"]].copy()
    result = result.dropna(subset=["source", "target"])
    result["source"] = result["source"].astype(str)
    result["target"] = result["target"].astype(str)
    result["distance_meters"] = pd.to_numeric(
        result["distance_meters"], errors="coerce"
    )
    if result["distance_meters"].isna().any() or (result["distance_meters"] < 0).any():
        raise ValueError(f"{snapshot_id} contains invalid edge distances")

    result = result[result["source"] != result["target"]].copy()
    result["left"] = result[["source", "target"]].min(axis=1)
    result["right"] = result[["source", "target"]].max(axis=1)
    # The snapshots are undirected. Keep one copy of each edge; distance is
    # symmetric and duplicate rows should not inflate any graph metric.
    return result.drop_duplicates(["left", "right"])[
        ["left", "right", "distance_meters"]
    ]


def main() -> None:
    for path in (FEATURES_PATH, INDEX_PATH, SNAPSHOTS_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Missing required graph input: {path}")

    events = pd.read_csv(FEATURES_PATH, low_memory=False)
    index = pd.read_csv(INDEX_PATH)
    required_events = {"event_id"}
    required_index = {
        "snapshot_id", "interval_start_utc", "interval_end_utc",
        "active_node_count", "active_edge_count",
    }
    if missing := required_events - set(events.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if missing := required_index - set(index.columns):
        raise ValueError(f"Snapshot index is missing columns: {sorted(missing)}")
    if events.empty or events["event_id"].isna().any():
        raise ValueError("Feature table must contain non-empty event IDs")

    events["event_id"] = events["event_id"].astype(str)
    if events["event_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate event IDs")

    index["interval_start_utc"] = pd.to_datetime(
        index["interval_start_utc"], utc=True, errors="coerce"
    )
    index["interval_end_utc"] = pd.to_datetime(
        index["interval_end_utc"], utc=True, errors="coerce"
    )
    index = index.sort_values("interval_start_utc").reset_index(drop=True)
    if index.empty or index[["interval_start_utc", "interval_end_utc"]].isna().any().any():
        raise ValueError("Snapshot index is empty or has invalid UTC intervals")
    starts = index["interval_start_utc"]
    ends = index["interval_end_utc"]
    if ((ends - starts) != pd.Timedelta(days=1)).any():
        raise ValueError("All graph snapshots must cover one UTC day")
    if starts.duplicated().any() or (starts.iloc[1:].reset_index(drop=True)
                                      != ends.iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("Snapshot index must contain unique, contiguous days")

    known_ids = set(events["event_id"])
    # Per-event daily series are accumulated in chronological order.
    series: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observed_neighbors: dict[str, set[str]] = defaultdict(set)
    incident_edge_days: dict[str, int] = defaultdict(int)
    active_days: dict[str, int] = defaultdict(int)
    previous_active: set[str] = set()
    previous_degree: dict[str, int] = {}

    for position, row in index.iterrows():
        snapshot_id = str(row["snapshot_id"])
        node_path = SNAPSHOTS_DIR / f"{snapshot_id}_nodes.csv"
        edge_path = SNAPSHOTS_DIR / f"{snapshot_id}_edges.csv"
        if not node_path.is_file() or not edge_path.is_file():
            raise FileNotFoundError(f"Missing node or edge file for {snapshot_id}")

        node_frame = pd.read_csv(node_path, usecols=["event_id"], dtype={"event_id": str})
        active = set(node_frame["event_id"].dropna())
        if len(active) != len(node_frame) or not active.issubset(known_ids):
            raise ValueError(f"{snapshot_id} has duplicate or unknown event IDs")
        if len(active) != int(row["active_node_count"]):
            raise ValueError(f"Node count does not match index for {snapshot_id}")

        edges = normalized_edges(pd.read_csv(edge_path), snapshot_id)
        if len(edges) != int(row["active_edge_count"]):
            raise ValueError(f"Edge count does not match index for {snapshot_id}")
        invalid_edges = ~edges["left"].isin(active) | ~edges["right"].isin(active)
        if invalid_edges.any():
            raise ValueError(f"{snapshot_id} has edges whose endpoints are inactive")

        adjacency: dict[str, set[str]] = {event_id: set() for event_id in active}
        weighted_degree: dict[str, float] = defaultdict(float)
        for left, right, distance_meters in edges.itertuples(index=False, name=None):
            distance_km = float(distance_meters) / 1000.0
            weight = 1.0 / (1.0 + distance_km)
            adjacency[left].add(right)
            adjacency[right].add(left)
            weighted_degree[left] += weight
            weighted_degree[right] += weight
            observed_neighbors[left].add(right)
            observed_neighbors[right].add(left)
            incident_edge_days[left] += 1
            incident_edge_days[right] += 1

        node_count = len(active)
        degrees = {event_id: len(neighbors) for event_id, neighbors in adjacency.items()}
        for event_id in active:
            neighbors = adjacency[event_id]
            degree = degrees[event_id]
            possible_pairs = degree * (degree - 1) / 2
            triangle_pairs = sum(
                len(neighbors.intersection(adjacency[neighbor]))
                for neighbor in neighbors
            ) / 2
            clustering = triangle_pairs / possible_pairs if possible_pairs else 0.0
            series[event_id]["degree"].append(float(degree))
            series[event_id]["normalized_degree"].append(
                degree / (node_count - 1) if node_count > 1 else 0.0
            )
            series[event_id]["weighted_degree"].append(weighted_degree[event_id])
            series[event_id]["clustering"].append(clustering)
            active_days[event_id] += 1

        for event_id in active & previous_active:
            change = degrees[event_id] - previous_degree[event_id]
            series[event_id]["absolute_degree_change"].append(float(abs(change)))
            series[event_id]["degree_changed"].append(float(change != 0))

        previous_active = active
        previous_degree = degrees
        if (position + 1) % 100 == 0 or position == 0:
            print(f"Processed {position + 1}/{len(index)} snapshots")

    records = []
    day_count = len(index)
    for event_id in events["event_id"]:
        metrics = series[event_id]
        degrees = np.asarray(metrics["degree"], dtype=float)
        normalized_degrees = np.asarray(metrics["normalized_degree"], dtype=float)
        weighted_degrees = np.asarray(metrics["weighted_degree"], dtype=float)
        clustering = np.asarray(metrics["clustering"], dtype=float)
        degree_changes = np.asarray(metrics["absolute_degree_change"], dtype=float)
        neighbors = observed_neighbors[event_id]
        event_edge_days = incident_edge_days[event_id]
        possible_event_edge_days = len(neighbors) * active_days[event_id]

        records.append({
            "event_id": event_id,
            "dynamic_active_snapshot_days": active_days[event_id],
            "dynamic_active_snapshot_fraction": active_days[event_id] / day_count,
            "dynamic_unique_neighbor_count": len(neighbors),
            "dynamic_mean_daily_degree": float(degrees.mean()) if len(degrees) else 0.0,
            "dynamic_max_daily_degree": float(degrees.max()) if len(degrees) else 0.0,
            "dynamic_degree_std": float(degrees.std(ddof=0)) if len(degrees) else 0.0,
            "dynamic_mean_normalized_degree": (
                float(normalized_degrees.mean()) if len(normalized_degrees) else 0.0
            ),
            "dynamic_mean_weighted_degree": (
                float(weighted_degrees.mean()) if len(weighted_degrees) else 0.0
            ),
            "dynamic_max_weighted_degree": (
                float(weighted_degrees.max()) if len(weighted_degrees) else 0.0
            ),
            "dynamic_mean_clustering_coefficient": (
                float(clustering.mean()) if len(clustering) else 0.0
            ),
            "dynamic_max_clustering_coefficient": (
                float(clustering.max()) if len(clustering) else 0.0
            ),
            "dynamic_mean_absolute_degree_change": (
                float(degree_changes.mean()) if len(degree_changes) else 0.0
            ),
            "dynamic_degree_change_days": int(sum(metrics["degree_changed"])),
            "dynamic_edge_persistence_fraction": (
                event_edge_days / possible_event_edge_days
                if possible_event_edge_days else 0.0
            ),
        })

    result = pd.DataFrame.from_records(records)
    if result["event_id"].duplicated().any() or len(result) != len(events):
        raise AssertionError("Dynamic feature output does not have one row per event")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(result):,} events and {len(result.columns) - 1} dynamic features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
