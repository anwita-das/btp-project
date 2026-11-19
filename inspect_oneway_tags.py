
import json
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

road_file = Path("data/road_network/road_ways.geojson")
directed_file = Path("data/road_network/topology/directed_segments.csv")

print("Loading original OSM road tags...")

with open(road_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Original OSM tag distribution
tag_counts = defaultdict(Counter)

for feature in data["features"]:
    p = feature["properties"]
    highway = p.get("highway", "unknown")
    oneway = str(p.get("oneway", "MISSING")).lower()
    tag_counts[highway][oneway] += 1

print("\n1. ORIGINAL OSM ONEWAY TAGS BY ROAD TYPE")

for highway, counts in sorted(tag_counts.items()):
    print(f"\n{highway}:")
    for tag, count in counts.most_common():
        print(f"  {tag}: {count}")

# Direction distribution in generated graph
print("\n\n2. GENERATED DIRECTION COUNTS BY ROAD TYPE")

df = pd.read_csv(directed_file, low_memory=False)

direction_table = pd.crosstab(
    df["highway"],
    df["direction"]
)

print(direction_table.to_string())

# Specifically identify motorways missing an explicit one-way tag
print("\n\n3. MOTORWAYS WITH MISSING ONEWAY TAG")

for feature in data["features"]:
    p = feature["properties"]

    if (
        p.get("highway") in {"motorway", "motorway_link"}
        and not p.get("oneway")
    ):
        print({
            "way_id": p.get("osm_way_id"),
            "highway": p.get("highway"),
            "oneway": p.get("oneway"),
            "junction": p.get("junction"),
            "name": p.get("name")
        })

print("\nDiagnostic complete.")