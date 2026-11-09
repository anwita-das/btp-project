
from collections import defaultdict
from pathlib import Path
import json
import os
import shutil

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FEATURES_FILE = DATA_DIR / "wzdx_graph_features.csv"
GRAPH_DIR = DATA_DIR / "new_temporal_graph"
INDEX_FILE = GRAPH_DIR / "snapshot_index.csv"
SNAPSHOT_DIR = GRAPH_DIR / "snapshots"


def main():
    if not FEATURES_FILE.is_file() or not INDEX_FILE.is_file():
        raise FileNotFoundError(
            "Missing wzdx_graph_features.csv or snapshot_index.csv"
        )

    df = pd.read_csv(FEATURES_FILE, low_memory=False)
    index = pd.read_csv(INDEX_FILE)

    needed = {"event_id", "start_time", "end_time"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"Missing feature columns: {sorted(needed - set(df.columns))}"
        )

    required_index = {
        "snapshot_id",
        "interval_start_utc",
        "interval_end_utc"
    }
    if not required_index.issubset(index.columns):
        raise ValueError("Snapshot index lacks daily interval columns")

    if (
        df["event_id"].isna().any()
        or df["event_id"].astype(str).duplicated().any()
    ):
        raise ValueError("Missing or duplicated event IDs")

    df["event_id"] = df["event_id"].astype(str)

    starts = pd.to_datetime(
        df["start_time"], utc=True, errors="coerce"
    )
    ends = pd.to_datetime(
        df["end_time"], utc=True, errors="coerce"
    )

    if (
        starts.isna().any()
        or ends.isna().any()
        or (ends <= starts).any()
    ):
        raise ValueError("Missing or invalid event timestamps")

    start_lookup = dict(zip(df["event_id"], starts))
    end_lookup = dict(zip(df["event_id"], ends))
    ids = set(df["event_id"])

    index_starts = pd.to_datetime(
        index["interval_start_utc"], utc=True, errors="coerce"
    )
    index_ends = pd.to_datetime(
        index["interval_end_utc"], utc=True, errors="coerce"
    )

    if (
        index.empty
        or index_starts.isna().any()
        or index_ends.isna().any()
    ):
        raise ValueError("Missing or invalid snapshot timestamps")

    if (
        index_ends - index_starts
        != pd.Timedelta(days=1)
    ).any():
        raise ValueError("Snapshots must use one-day UTC intervals")

    if (
        not index_starts.is_monotonic_increasing
        or index["snapshot_id"].duplicated().any()
    ):
        raise ValueError("Snapshot index must be ordered and unique")

    if len(index) > 1:
        next_starts = index_starts.iloc[1:].reset_index(drop=True)
        previous_ends = index_ends.iloc[:-1].reset_index(drop=True)

        if (next_starts != previous_ends).any():
            raise ValueError("Snapshot days are not contiguous")

    # Time-dependent feature sequences
    daily_neighbors = defaultdict(list)
    daily_changes = defaultdict(list)
    daily_elapsed = defaultdict(list)

    previous_counts = {}

    print(f"Events: {len(df):,}")
    print(f"Daily snapshots: {len(index):,}")

    # --------------------------------------------------
    # PROCESS EACH DAILY SNAPSHOT
    # --------------------------------------------------

    for number, row in index.iterrows():
        name = str(row["snapshot_id"])

        node_path = SNAPSHOT_DIR / f"{name}_nodes.csv"
        edge_path = SNAPSHOT_DIR / f"{name}_edges.csv"

        if not node_path.is_file() or not edge_path.is_file():
            raise FileNotFoundError(
                f"Missing snapshot files for {name}"
            )

        active = pd.read_csv(
            node_path,
            usecols=["event_id"],
            dtype=str
        )["event_id"]

        edge_df = pd.read_csv(
            edge_path,
            usecols=["source", "target"],
            dtype=str
        )

        active_ids = set(active)

        if (
            len(active_ids) != len(active)
            or not active_ids.issubset(ids)
        ):
            raise ValueError(
                f"Invalid or duplicate node IDs in {name}"
            )

        invalid_edges = (
            ~edge_df["source"].isin(active_ids)
            | ~edge_df["target"].isin(active_ids)
        )

        if invalid_edges.any():
            raise ValueError(
                f"Invalid edge endpoints in {name}"
            )

        if "active_node_count" in index.columns:
            if int(row["active_node_count"]) != len(active_ids):
                raise ValueError(
                    f"Node count mismatch in {name}"
                )

        if "active_edge_count" in index.columns:
            if int(row["active_edge_count"]) != len(edge_df):
                raise ValueError(
                    f"Edge count mismatch in {name}"
                )

        # Count active graph edges incident on each node
        endpoints = pd.concat(
            [edge_df["source"], edge_df["target"]],
            ignore_index=True
        )

        counts = endpoints.value_counts().to_dict()

        interval_end = index_ends.iloc[number]

        for event_id in active_ids:
            count = int(counts.get(event_id, 0))

            # Change compared with the preceding daily snapshot
            # An event absent yesterday starts from a baseline of 0
            change = count - previous_counts.get(event_id, 0)

            # Scheduled duration elapsed by the end of this day
            elapsed = (
                min(end_lookup[event_id], interval_end)
                - start_lookup[event_id]
            ).total_seconds() / 86400

            daily_neighbors[event_id].append(
                [number, count]
            )
            daily_changes[event_id].append(
                [number, change]
            )
            daily_elapsed[event_id].append(
                [number, round(max(0.0, elapsed), 6)]
            )

        # Remember counts for the next day
        previous_counts = {
            event_id: int(counts.get(event_id, 0))
            for event_id in active_ids
        }

        if (number + 1) % 100 == 0 or number == 0:
            print(
                f"Processed {number + 1}/{len(index)} snapshots"
            )

    # --------------------------------------------------
    # ADD TIME-DEPENDENT COLUMNS TO THE CSV
    # --------------------------------------------------

    def as_json(mapping, event_id):
        return json.dumps(
            mapping.get(event_id, []),
            separators=(",", ":")
        )

    df["active_neighbor_count"] = df["event_id"].map(
        lambda eid: as_json(daily_neighbors, eid)
    )

    df["neighbor_count_change"] = df["event_id"].map(
        lambda eid: as_json(daily_changes, eid)
    )

    df["event_active_duration_at_snapshot"] = df["event_id"].map(
        lambda eid: as_json(daily_elapsed, eid)
    )

    # --------------------------------------------------
    # ADD NUMERIC SUMMARY FEATURES
    # --------------------------------------------------

    df["snapshot_active_days"] = df["event_id"].map(
        lambda eid: len(daily_neighbors.get(eid, []))
    ).astype(int)

    df["snapshot_mean_active_neighbor_count"] = df["event_id"].map(
        lambda eid: (
            sum(v for _, v in daily_neighbors[eid])
            / len(daily_neighbors[eid])
            if daily_neighbors.get(eid)
            else float("nan")
        )
    )

    df["snapshot_max_active_neighbor_count"] = df["event_id"].map(
        lambda eid: max(
            (v for _, v in daily_neighbors[eid]),
            default=0
        )
    ).astype(int)

    df["snapshot_days_neighbor_count_changed"] = df["event_id"].map(
        lambda eid: sum(
            v != 0
            for _, v in daily_changes.get(eid, [])
        )
    ).astype(int)

    df["snapshot_max_elapsed_days"] = df["event_id"].map(
        lambda eid: max(
            (v for _, v in daily_elapsed[eid]),
            default=0.0
        )
    )

    # --------------------------------------------------
    # SAFELY UPDATE THE EXISTING CSV
    # --------------------------------------------------

    backup_file = FEATURES_FILE.with_name(
        FEATURES_FILE.name + ".bak"
    )
    temp_file = FEATURES_FILE.with_name(
        FEATURES_FILE.name + ".tmp"
    )

    shutil.copy2(FEATURES_FILE, backup_file)

    try:
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, FEATURES_FILE)
    finally:
        if temp_file.exists():
            temp_file.unlink()

    print("\nSnapshot feature engineering complete.")
    print(f"Updated: {FEATURES_FILE}")
    print(f"Backup: {backup_file}")
    print(f"Output shape: {df.shape}")
    print(
        "The first three features are JSON sequences "
        "of [snapshot_number, value]."
    )


if __name__ == "__main__":
    main()
