
from pathlib import Path
from difflib import SequenceMatcher
import json
import re

import pandas as pd
import geopandas as gpd

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent

EVENT_FILE = BASE / "data" / "wzdx_features.csv"
RAW_GEOMETRY_FILE = BASE / "data" / "raw" / "wzdx_raw.geojson"
ROAD_FILE = (
    BASE / "data" / "road_network" /
    "topology" / "physical_segments.geojson"
)

OUTPUT_DIR = BASE / "data" / "road_network" / "event_matching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRS_METRES = "EPSG:32614"

# Search within 30 metres of the actual event geometry.
SEARCH_RADIUS_M = 30

# ============================================================
# 2. ROAD NAME NORMALIZATION
# ============================================================

ABBREVIATIONS = {
    "ST": "STREET",
    "RD": "ROAD",
    "AVE": "AVENUE",
    "BLVD": "BOULEVARD",
    "DR": "DRIVE",
    "HWY": "HIGHWAY",
    "LN": "LANE",
    "PKWY": "PARKWAY",
    "CT": "COURT",
    "TRL": "TRAIL",
    "CIR": "CIRCLE",
    "N": "NORTH",
    "S": "SOUTH",
    "E": "EAST",
    "W": "WEST",
}


def normalize_name(value):

    if value is None or pd.isna(value):
        return ""

    value = str(value).upper()
    value = re.sub(r"[^A-Z0-9 ]", " ", value)

    words = value.split()
    words = [ABBREVIATIONS.get(word, word) for word in words]

    return " ".join(words)


def name_similarity(event_name, road_name, road_ref):

    event_name = normalize_name(event_name)
    road_name = normalize_name(road_name)
    road_ref = normalize_name(road_ref)

    if not event_name:
        return 0.0, False

    scores = [
        SequenceMatcher(None, event_name, candidate).ratio()
        for candidate in [road_name, road_ref]
        if candidate
    ]

    if not scores:
        return 0.0, False

    exact = event_name in [road_name, road_ref]

    return max(scores), exact


# ============================================================
# 3. LOAD CLEANED EVENT DATA
# ============================================================

print("Loading cleaned WZDx features...")

events = pd.read_csv(EVENT_FILE, low_memory=False)

required_columns = [
    "event_id",
    "road_names",
    "start_time",
    "end_time",
]

missing = [c for c in required_columns if c not in events.columns]

if missing:
    raise ValueError(f"Missing event columns: {missing}")

if events["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs in cleaned dataset.")

print("Cleaned events:", len(events))


# ============================================================
# 4. LOAD ORIGINAL GEOMETRIES
# ============================================================

print("\nLoading original WZDx geometries...")

raw = gpd.read_file(RAW_GEOMETRY_FILE)

if raw.crs is None:
    raise ValueError("Original WZDx GeoJSON has no CRS.")

if "id" not in raw.columns:
    raise ValueError("Original GeoJSON does not contain 'id'.")

if raw["id"].duplicated().any():
    raise ValueError("Duplicate IDs in original GeoJSON.")

# The cleaned CSV's event_id corresponds to raw GeoJSON's id.
geometry_lookup = raw.set_index("id").geometry

events["geometry"] = events["event_id"].map(geometry_lookup)

missing_geometry = events["geometry"].isna().sum()

if missing_geometry:
    raise ValueError(
        f"{missing_geometry} cleaned events have no original geometry."
    )

event_gdf = gpd.GeoDataFrame(
    events.copy(),
    geometry="geometry",
    crs=raw.crs
)

# Validate geometry.
invalid = (
    event_gdf.geometry.is_empty |
    ~event_gdf.geometry.is_valid
)

if invalid.any():
    raise ValueError(
        f"{int(invalid.sum())} invalid event geometries found."
    )

print("Original geometries successfully joined:", len(event_gdf))
print("Geometry types:", event_gdf.geom_type.value_counts().to_dict())


# ============================================================
# 5. LOAD ROAD NETWORK
# ============================================================

print("\nLoading physical road segments...")

roads = gpd.read_file(ROAD_FILE)

if roads.crs is None:
    raise ValueError("Road network has no CRS.")

required_road_columns = [
    "segment_id",
    "osm_way_id",
    "highway",
    "name",
    "ref",
]

missing = [
    c for c in required_road_columns
    if c not in roads.columns
]

if missing:
    raise ValueError(f"Missing road columns: {missing}")

if roads["segment_id"].duplicated().any():
    raise ValueError("Duplicate road segment IDs.")

roads = roads[
    roads.geometry.notna() &
    ~roads.geometry.is_empty
].copy()

# Project both datasets to metres.
event_gdf = event_gdf.to_crs(CRS_METRES)
roads = roads.to_crs(CRS_METRES)

roads = roads.reset_index(drop=True)

print("Physical road segments:", len(roads))


# ============================================================
# 6. BUILD SPATIAL INDEX
# ============================================================

print("\nBuilding road spatial index...")

spatial_index = roads.sindex


# ============================================================
# 7. MATCH EVENTS TO ROAD SEGMENTS
# ============================================================

print("\nMatching events to road segments...")

candidate_records = []
unmatched_records = []

for count, (_, event) in enumerate(event_gdf.iterrows(), start=1):

    event_id = event["event_id"]
    event_geometry = event.geometry
    event_road_name = event["road_names"]

    # Find roads within 30 metres of the actual event geometry.
    candidate_indices = spatial_index.query(
        event_geometry.buffer(SEARCH_RADIUS_M),
        predicate="intersects"
    )

    event_candidates = []

    for road_idx in candidate_indices:

        road = roads.iloc[road_idx]

        road_geometry = road.geometry

        distance = event_geometry.distance(road_geometry)

        intersection = event_geometry.intersection(road_geometry)

        overlap_length = intersection.length

        similarity, exact_match = name_similarity(
            event_road_name,
            road["name"],
            road["ref"]
        )

        # Retain candidates supported by either:
        # 1. Road-name similarity and reasonable proximity.
        # 2. Meaningful geometric overlap.
        plausible = (
            (
                similarity >= 0.55
                and distance <= SEARCH_RADIUS_M
            )
            or (
                overlap_length >= 5
                and distance <= 3
            )
        )

        if exact_match and distance <= 15:
            confidence = "high"

        elif similarity >= 0.75 and distance <= 15:
            confidence = "high"

        elif plausible:
            confidence = "medium"

        else:
            confidence = "low"

        event_candidates.append({

            "event_id": event_id,
            "event_name": event["event_name"],
            "event_road_name": event_road_name,

            "segment_id": road["segment_id"],
            "osm_way_id": road["osm_way_id"],

            "road_name": road["name"],
            "road_ref": road["ref"],
            "highway": road["highway"],

            "distance_m": round(float(distance), 3),
            "overlap_length_m": round(float(overlap_length), 3),

            "name_similarity": round(float(similarity), 3),
            "exact_name_match": bool(exact_match),

            "is_plausible_match": bool(plausible),
            "confidence": confidence,
        })

    if not event_candidates:

        unmatched_records.append({
            "event_id": event_id,
            "event_road_name": event_road_name,
            "reason": "No road segment within search radius"
        })

    else:

        candidate_records.extend(event_candidates)

    if count % 250 == 0:
        print(f"Processed {count}/{len(event_gdf)} events")


# ============================================================
# 8. RANK CANDIDATES
# ============================================================

candidates = pd.DataFrame(candidate_records)

if candidates.empty:
    raise RuntimeError("No candidate road segments were found.")

candidates = candidates.sort_values(
    by=[
        "event_id",
        "is_plausible_match",
        "exact_name_match",
        "name_similarity",
        "distance_m",
    ],
    ascending=[True, False, False, False, True]
).copy()

candidates["candidate_rank"] = (
    candidates.groupby("event_id").cumcount() + 1
)

candidates["is_best_candidate"] = (
    candidates["candidate_rank"] == 1
)

# Keep plausible associations separately.
matches = candidates[
    candidates["is_plausible_match"]
].copy()

unmatched = pd.DataFrame(unmatched_records)


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

candidates.to_csv(
    OUTPUT_DIR / "event_road_candidates.csv",
    index=False
)

matches.to_csv(
    OUTPUT_DIR / "event_road_matches.csv",
    index=False
)

candidates[
    candidates["is_best_candidate"]
].to_csv(
    OUTPUT_DIR / "event_road_best_candidates.csv",
    index=False
)

unmatched.to_csv(
    OUTPUT_DIR / "unmatched_events.csv",
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

summary = {

    "cleaned_event_count": int(len(event_gdf)),

    "events_with_candidates": int(
        candidates["event_id"].nunique()
    ),

    "events_with_plausible_matches": int(
        matches["event_id"].nunique()
    ),

    "events_without_candidates": int(len(unmatched)),

    "total_candidate_pairs": int(len(candidates)),

    "plausible_event_segment_pairs": int(len(matches)),

    "confidence_distribution": {
        str(k): int(v)
        for k, v in matches["confidence"].value_counts().items()
    },

    "distance_statistics_m": {
        "mean": round(float(matches["distance_m"].mean()), 3)
        if len(matches) else None,

        "median": round(float(matches["distance_m"].median()), 3)
        if len(matches) else None,

        "maximum": round(float(matches["distance_m"].max()), 3)
        if len(matches) else None,
    }
}

with open(OUTPUT_DIR / "matching_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n========== MATCHING COMPLETE ==========")
print(json.dumps(summary, indent=4))

print("\nOutput directory:")
print(OUTPUT_DIR)