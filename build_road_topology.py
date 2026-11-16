
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString, mapping


# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_FILE = Path("data/road_network/road_ways.geojson")
OUTPUT_DIR = Path("data/road_network/topology")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def haversine_distance(coords):
    """Calculate length of a LineString in metres."""

    radius = 6371000
    total = 0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]

        lat1, lat2 = math.radians(lat1), math.radians(lat2)
        dlat = lat2 - lat1
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        total += 2 * radius * math.asin(math.sqrt(a))

    return total


def is_oneway(properties):
    """Interpret OSM one-way tags."""

    value = str(properties.get("oneway", "")).lower().strip()

    if value in {"yes", "true", "1"}:
        return "forward"

    if value == "-1":
        return "reverse"

    if value in {"no", "false", "0"}:
        return "both"

    # Roundabouts are normally one-way unless explicitly overridden.
    if properties.get("junction") == "roundabout":
        return "forward"

    return "both"


# --------------------------------------------------
# LOAD ROAD DATA
# --------------------------------------------------

print("Loading extracted road network...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    geojson = json.load(f)

features = geojson["features"]

print(f"Loaded {len(features):,} road ways.")


# --------------------------------------------------
# IDENTIFY INTERSECTION NODES
# --------------------------------------------------

print("\nIdentifying shared OSM nodes...")

# Each OSM node ID maps to the set of road ways using it.
node_to_ways = defaultdict(set)

for feature in features:
    props = feature["properties"]
    way_id = str(props["osm_way_id"])

    for node_id in set(props["osm_node_ids"]):
        node_to_ways[node_id].add(way_id)

# A node used by multiple ways is treated as a junction.
shared_nodes = {
    node_id
    for node_id, ways in node_to_ways.items()
    if len(ways) > 1
}

print(f"Unique OSM nodes: {len(node_to_ways):,}")
print(f"Shared nodes: {len(shared_nodes):,}")


# --------------------------------------------------
# SPLIT WAYS INTO PHYSICAL ROAD SEGMENTS
# --------------------------------------------------

print("\nSplitting roads at intersections...")

physical_segments = []
intersection_coordinates = {}

for feature in features:

    props = feature["properties"]

    way_id = str(props["osm_way_id"])
    node_ids = props["osm_node_ids"]
    coordinates = feature["geometry"]["coordinates"]

    if len(node_ids) != len(coordinates):
        print(f"Skipping mismatched way: {way_id}")
        continue

    if len(node_ids) < 2:
        continue

    # Split at:
    # 1. First node
    # 2. Last node
    # 3. Every node shared by multiple ways

    split_indices = [0]

    for i in range(1, len(node_ids) - 1):
        if node_ids[i] in shared_nodes:
            split_indices.append(i)

    split_indices.append(len(node_ids) - 1)

    # Remove duplicate indices while preserving order.
    split_indices = list(dict.fromkeys(split_indices))

    for part_number in range(len(split_indices) - 1):

        start_idx = split_indices[part_number]
        end_idx = split_indices[part_number + 1]

        segment_coords = coordinates[start_idx:end_idx + 1]
        segment_node_ids = node_ids[start_idx:end_idx + 1]

        if len(segment_coords) < 2:
            continue

        start_node = segment_node_ids[0]
        end_node = segment_node_ids[-1]

        segment_id = f"{way_id}_{part_number}"

        line = LineString(segment_coords)

        if line.is_empty or line.length == 0:
            continue

        length_m = haversine_distance(segment_coords)

        # Store intersection coordinates.
        intersection_coordinates[start_node] = segment_coords[0]
        intersection_coordinates[end_node] = segment_coords[-1]

        physical_segments.append({
            "segment_id": segment_id,
            "osm_way_id": way_id,
            "part_number": part_number,
            "start_node": start_node,
            "end_node": end_node,
            "length_m": length_m,
            "highway": props.get("highway"),
            "name": props.get("name"),
            "ref": props.get("ref"),
            "lanes": props.get("lanes"),
            "maxspeed": props.get("maxspeed"),
            "oneway": props.get("oneway"),
            "bridge": props.get("bridge"),
            "tunnel": props.get("tunnel"),
            "surface": props.get("surface"),
            "access": props.get("access"),
            "service": props.get("service"),
            "junction": props.get("junction"),
            "geometry": line
        })


print(f"Created {len(physical_segments):,} physical road segments.")


# --------------------------------------------------
# CREATE INTERSECTION NODES
# --------------------------------------------------

print("\nCreating intersection node table...")

node_degree = Counter()

for segment in physical_segments:
    node_degree[segment["start_node"]] += 1
    node_degree[segment["end_node"]] += 1

road_nodes = []

for node_id, degree in node_degree.items():

    lon, lat = intersection_coordinates[node_id]

    road_nodes.append({
        "node_id": node_id,
        "longitude": lon,
        "latitude": lat,
        "degree": degree
    })

nodes_df = pd.DataFrame(road_nodes)

nodes_df.to_csv(
    OUTPUT_DIR / "intersection_nodes.csv",
    index=False
)


# --------------------------------------------------
# CREATE DIRECTED ROAD SEGMENTS
# --------------------------------------------------

print("\nCreating directed road segments...")

directed_segments = []

for segment in physical_segments:

    direction = is_oneway(segment)

    forward_id = segment["segment_id"] + "_F"
    reverse_id = segment["segment_id"] + "_R"

    common = {
        "physical_segment_id": segment["segment_id"],
        "osm_way_id": segment["osm_way_id"],
        "highway": segment["highway"],
        "name": segment["name"],
        "lanes": segment["lanes"],
        "maxspeed": segment["maxspeed"],
        "length_m": segment["length_m"],
        "bridge": segment["bridge"],
        "tunnel": segment["tunnel"],
        "surface": segment["surface"]
    }

    if direction in {"forward", "both"}:
        directed_segments.append({
            **common,
            "directed_segment_id": forward_id,
            "from_node": segment["start_node"],
            "to_node": segment["end_node"],
            "direction": "forward"
        })

    if direction in {"reverse", "both"}:
        directed_segments.append({
            **common,
            "directed_segment_id": reverse_id,
            "from_node": segment["end_node"],
            "to_node": segment["start_node"],
            "direction": "reverse"
        })


directed_df = pd.DataFrame(directed_segments)

directed_df.to_csv(
    OUTPUT_DIR / "directed_segments.csv",
    index=False
)


# --------------------------------------------------
# CREATE ROAD-SEGMENT GRAPH
# --------------------------------------------------

print("\nBuilding road-segment graph...")

# Each directed road segment becomes a graph node.
# A graph edge exists when one directed segment can
# be followed by another at a shared intersection.

incoming = defaultdict(list)
outgoing = defaultdict(list)

for segment in directed_segments:
    incoming[segment["to_node"]].append(segment)
    outgoing[segment["from_node"]].append(segment)

graph_edges = []

for intersection in set(incoming) & set(outgoing):

    arriving_segments = incoming[intersection]
    departing_segments = outgoing[intersection]

    for arriving in arriving_segments:
        for departing in departing_segments:

            # Do not create an immediate U-turn onto
            # the opposite direction of the same road.
            if (
                arriving["physical_segment_id"]
                == departing["physical_segment_id"]
            ):
                continue

            graph_edges.append({
                "from_segment": arriving["directed_segment_id"],
                "to_segment": departing["directed_segment_id"],
                "intersection_node": intersection
            })


graph_edges_df = pd.DataFrame(
    graph_edges,
    columns=[
        "from_segment",
        "to_segment",
        "intersection_node"
    ]
)

graph_edges_df.to_csv(
    OUTPUT_DIR / "road_segment_edges.csv",
    index=False
)


# --------------------------------------------------
# SAVE PHYSICAL SEGMENT GEOMETRIES
# --------------------------------------------------

print("\nSaving physical segment geometries...")

geojson_features = []

for segment in physical_segments:

    properties = {
        key: value
        for key, value in segment.items()
        if key != "geometry"
    }

    geojson_features.append({
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(segment["geometry"])
    })

with open(
    OUTPUT_DIR / "physical_segments.geojson",
    "w",
    encoding="utf-8"
) as f:

    json.dump({
        "type": "FeatureCollection",
        "features": geojson_features
    }, f)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

summary = {
    "input_road_ways": len(features),
    "shared_osm_nodes": len(shared_nodes),
    "intersection_nodes": len(road_nodes),
    "physical_road_segments": len(physical_segments),
    "directed_road_segments": len(directed_segments),
    "road_segment_graph_edges": len(graph_edges),
    "isolated_intersection_nodes": int(
        (nodes_df["degree"] == 1).sum()
    ),
    "road_type_counts": dict(
        Counter(
            str(s["highway"])
            for s in physical_segments
        )
    )
}

with open(
    OUTPUT_DIR / "topology_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)

print("\n" + "=" * 50)
print("ROAD TOPOLOGY BUILD COMPLETE")
print("=" * 50)

for key, value in summary.items():
    print(f"{key}: {value}")

print(f"\nOutput directory: {OUTPUT_DIR}")