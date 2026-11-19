
from pathlib import Path
import json
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

DATA = BASE / "data"
TOPOLOGY = DATA / "road_network" / "topology"
MATCH_DIR = DATA / "road_network" / "event_matching"

EVENT_FILE = DATA / "wzdx_features.csv"
MATCH_FILE = MATCH_DIR / "event_road_matches.csv"
DIRECTED_FILE = TOPOLOGY / "directed_segments.csv"
EDGE_FILE = TOPOLOGY / "road_segment_edges.csv"

OUT = DATA / "dynamic_road_graph"
SNAPSHOT_DIR = OUT / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def to_bool(series):
    """Convert common boolean representations to 0/1."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .astype(int)
    )


def find_edge_columns(edges, valid_ids):
    """
    Detect the source and target directed-segment columns.
    First try common column names, then inspect ID overlap.
    """

    source_names = [
        "source_segment_id", "source", "from_segment_id",
        "from_segment", "src", "u"
    ]

    target_names = [
        "target_segment_id", "target", "to_segment_id",
        "to_segment", "dst", "v"
    ]

    source = next(
        (c for c in source_names if c in edges.columns), None
    )
    target = next(
        (c for c in target_names if c in edges.columns), None
    )

    if source and target:
        return source, target

    # Fallback: find columns whose values match directed IDs.
    scores = {}

    for col in edges.columns:
        if edges[col].dtype == "object":
            scores[col] = edges[col].isin(valid_ids).sum()

    ranked = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )

    if len(ranked) >= 2 and ranked[1][1] > 0:
        return ranked[0][0], ranked[1][0]

    raise ValueError(
        "Could not identify edge source/target columns. "
        f"Available columns: {list(edges.columns)}"
    )


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading event features...")
events = pd.read_csv(EVENT_FILE, low_memory=False)

print("Loading event-road matches...")
matches = pd.read_csv(MATCH_FILE, low_memory=False)

print("Loading directed road segments...")
roads = pd.read_csv(DIRECTED_FILE, low_memory=False)

print("Loading road topology edges...")
edges = pd.read_csv(EDGE_FILE, low_memory=False)

print("\nInput sizes:")
print("Events:", len(events))
print("Event-road matches:", len(matches))
print("Directed road nodes:", len(roads))
print("Road connectivity edges:", len(edges))


# ============================================================
# 2. VALIDATE INPUTS
# ============================================================

required_event = [
    "event_id", "start_time", "end_time",
    "vehicle_impact_score", "duration_hours",
    "workers_present_feature", "verified_fields_count",
    "geometry_length_km"
]

required_match = [
    "event_id", "segment_id", "confidence"
]

required_road = [
    "physical_segment_id", "directed_segment_id",
    "length_m", "highway"
]

for name, df, required in [
    ("events", events, required_event),
    ("matches", matches, required_match),
    ("roads", roads, required_road)
]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )

if events["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs in event features.")

if roads["directed_segment_id"].duplicated().any():
    raise ValueError("Duplicate directed road segment IDs.")

# Only use high and medium confidence matches.
matches = matches[
    matches["confidence"].isin(["high", "medium"])
].copy()

# Remove duplicate event-to-physical-road associations.
matches = matches.drop_duplicates(
    subset=["event_id", "segment_id"]
)

# Check that every matched physical segment exists.
valid_physical_ids = set(roads["physical_segment_id"].dropna())

invalid_matches = matches[
    ~matches["segment_id"].isin(valid_physical_ids)
]

if len(invalid_matches):
    raise ValueError(
        f"{len(invalid_matches)} matches refer to unknown "
        "physical road segments."
    )

print("\nValidated matches:", len(matches))


# ============================================================
# 3. PREPARE EVENT DATA
# ============================================================

events["start_time"] = pd.to_datetime(
    events["start_time"], utc=True, errors="coerce"
)

events["end_time"] = pd.to_datetime(
    events["end_time"], utc=True, errors="coerce"
)

events = events.dropna(
    subset=["start_time", "end_time"]
).copy()

events = events[
    events["end_time"] > events["start_time"]
].copy()

# Convert numerical features safely.
numeric_columns = [
    "vehicle_impact_score",
    "duration_hours",
    "verified_fields_count",
    "geometry_length_km"
]

for col in numeric_columns:
    events[col] = pd.to_numeric(
        events[col], errors="coerce"
    )

events["workers_present_feature"] = to_bool(
    events["workers_present_feature"]
)

events["all_lanes_closed"] = to_bool(
    events["vehicle_impact_all-lanes-closed"]
)

events["some_lanes_closed"] = to_bool(
    events["vehicle_impact_some-lanes-closed"]
)

# Join cleaned event attributes with road associations.
event_matches = matches.merge(
    events,
    on="event_id",
    how="inner",
    validate="many_to_one"
)

if event_matches.empty:
    raise ValueError("No event-road associations survived the join.")

print("Joined event-road rows:", len(event_matches))


# ============================================================
# 4. MAP PHYSICAL SEGMENTS TO DIRECTED ROAD NODES
# ============================================================

# A physical road segment may have one or two directed versions.
# Each direction receives the work-zone features associated
# with that physical segment.

event_directed = event_matches.merge(
    roads[
        ["physical_segment_id", "directed_segment_id"]
    ],
    left_on="segment_id",
    right_on="physical_segment_id",
    how="inner",
    validate="many_to_many"
)

if event_directed.empty:
    raise ValueError("No event associations mapped to directed nodes.")

event_directed = event_directed.drop_duplicates(
    subset=["event_id", "directed_segment_id"]
)

print(
    "Event-to-directed-node associations:",
    len(event_directed)
)


# ============================================================
# 5. BUILD PERSISTENT ROAD GRAPH
# ============================================================

valid_directed_ids = set(
    roads["directed_segment_id"].astype(str)
)

source_col, target_col = find_edge_columns(
    edges, valid_directed_ids
)

graph_edges = edges[
    [source_col, target_col]
].copy()

graph_edges.columns = ["source", "target"]

graph_edges["source"] = graph_edges["source"].astype(str)
graph_edges["target"] = graph_edges["target"].astype(str)

# Validate topology references.
bad_edges = graph_edges[
    ~graph_edges["source"].isin(valid_directed_ids)
    | ~graph_edges["target"].isin(valid_directed_ids)
]

if len(bad_edges):
    raise ValueError(
        f"{len(bad_edges)} graph edges reference missing nodes."
    )

graph_edges.to_csv(
    OUT / "road_graph_edges.csv",
    index=False
)

# Static node table: every road segment exists in every snapshot.
static_nodes = roads.copy()

static_nodes.to_csv(
    OUT / "road_graph_nodes.csv",
    index=False
)

print("\nPersistent graph:")
print("Nodes:", len(static_nodes))
print("Edges:", len(graph_edges))


# ============================================================
# 6. GENERATE DAILY EVENT ACTIVITY
# ============================================================

print("\nGenerating daily activity...")

activity_records = []

feature_columns = [
    "event_id",
    "directed_segment_id",
    "start_time",
    "end_time",
    "confidence",
    "vehicle_impact_score",
    "duration_hours",
    "workers_present_feature",
    "all_lanes_closed",
    "some_lanes_closed",
    "verified_fields_count",
    "geometry_length_km"
]

for row in event_directed[feature_columns].itertuples(
    index=False, name=None
):

    (
        event_id,
        directed_id,
        start,
        end,
        confidence,
        impact,
        duration,
        workers,
        all_closed,
        some_closed,
        verified,
        geometry_length
    ) = row

    # Daily snapshots use UTC calendar days.
    first_day = start.normalize()
    last_day = (end - pd.Timedelta(nanoseconds=1)).normalize()

    days = pd.date_range(
        first_day, last_day, freq="D"
    )

    for day in days:

        activity_records.append({

            "date": day.strftime("%Y-%m-%d"),
            "event_id": event_id,
            "directed_segment_id": directed_id,

            "confidence": confidence,
            "vehicle_impact_score": impact,
            "duration_hours": duration,

            "workers_present": workers,
            "all_lanes_closed": all_closed,
            "some_lanes_closed": some_closed,

            "verified_fields_count": verified,
            "geometry_length_km": geometry_length
        })

print("Daily event activity rows:", len(activity_records))

activity = pd.DataFrame(activity_records)

if activity.empty:
    raise ValueError("No daily activity rows generated.")


# ============================================================
# 7. AGGREGATE DAILY NODE FEATURES
# ============================================================

print("\nAggregating daily road-node features...")

daily_features = (
    activity
    .groupby(["date", "directed_segment_id"])
    .agg(
        active_event_count=("event_id", "nunique"),

        high_confidence_event_count=(
            "confidence",
            lambda x: (x == "high").sum()
        ),

        medium_confidence_event_count=(
            "confidence",
            lambda x: (x == "medium").sum()
        ),

        max_vehicle_impact_score=(
            "vehicle_impact_score", "max"
        ),

        mean_vehicle_impact_score=(
            "vehicle_impact_score", "mean"
        ),

        all_lanes_closed_count=(
            "all_lanes_closed", "sum"
        ),

        some_lanes_closed_count=(
            "some_lanes_closed", "sum"
        ),

        workers_present_count=(
            "workers_present", "sum"
        ),

        mean_event_duration_hours=(
            "duration_hours", "mean"
        ),

        mean_verified_fields_count=(
            "verified_fields_count", "mean"
        ),

        mean_event_geometry_length_km=(
            "geometry_length_km", "mean"
        )
    )
    .reset_index()
)

# Fill missing aggregate values with zero.
feature_cols = [
    c for c in daily_features.columns
    if c not in ["date", "directed_segment_id"]
]

daily_features[feature_cols] = (
    daily_features[feature_cols].fillna(0)
)


# ============================================================
# 8. WRITE DAILY SNAPSHOTS
# ============================================================

print("\nWriting daily graph snapshots...")

all_days = pd.date_range(
    activity["date"].min(),
    activity["date"].max(),
    freq="D"
)

snapshot_index = []

for day in all_days:

    date_string = day.strftime("%Y-%m-%d")

    snapshot = daily_features[
        daily_features["date"] == date_string
    ].copy()

    output_file = (
        SNAPSHOT_DIR / f"snapshot_{date_string}.csv"
    )

    snapshot.to_csv(output_file, index=False)

    snapshot_index.append({
        "date": date_string,
        "file": output_file.name,
        "active_road_nodes": int(len(snapshot))
    })

    if len(snapshot_index) % 30 == 0:
        print(
            f"Written {len(snapshot_index)}/{len(all_days)} snapshots"
        )

pd.DataFrame(snapshot_index).to_csv(
    OUT / "snapshot_index.csv",
    index=False
)


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

summary = {

    "graph_type": "Persistent directed road graph with daily dynamic node features",

    "static_road_nodes": int(len(static_nodes)),
    "static_road_edges": int(len(graph_edges)),

    "cleaned_events": int(events["event_id"].nunique()),
    "used_event_road_pairs": int(
        matches[["event_id", "segment_id"]]
        .drop_duplicates().shape[0]
    ),

    "directed_event_associations": int(len(event_directed)),

    "daily_snapshots": int(len(all_days)),

    "daily_node_feature_rows": int(len(daily_features)),

    "daily_feature_columns": feature_cols,

    "first_snapshot": str(all_days.min().date()),
    "last_snapshot": str(all_days.max().date()),

    "edge_source_column": source_col,
    "edge_target_column": target_col
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n========== DYNAMIC ROAD GRAPH COMPLETE ==========")
print(json.dumps(summary, indent=4))
print("\nOutput folder:", OUT)