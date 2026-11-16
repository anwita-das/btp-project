
from pathlib import Path
from collections import Counter
import json
import osmium


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OSM_FILE = BASE_DIR / "data" / "raw" / "austin_road_network.osm"
OUTPUT_DIR = BASE_DIR / "data" / "road_network"

OUTPUT_FILE = OUTPUT_DIR / "road_ways.geojson"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "busway",
}

EXCLUDED_SERVICES = {
    "parking_aisle",
    "driveway",
    "emergency_access",
}


# ============================================================
# ROAD FILTER
# ============================================================

def is_drivable(tags):

    highway = tags.get("highway", "")

    if highway not in ALLOWED_HIGHWAYS:
        return False

    # Exclude explicitly inaccessible roads.
    for key in ("access", "vehicle", "motor_vehicle", "motorcar"):

        value = tags.get(key, "").lower()

        if value in {"no", "private"}:
            return False

    service = tags.get("service", "").lower()

    if service in EXCLUDED_SERVICES:
        return False

    return True


# ============================================================
# PYOSMIUM HANDLER
# ============================================================

class RoadExtractor(osmium.SimpleHandler):

    def __init__(self, output_file):

        super().__init__()

        self.output_file = output_file
        self.road_count = 0
        self.skipped_count = 0
        self.highway_counts = Counter()

        self.file = open(
            output_file,
            "w",
            encoding="utf-8"
        )

        self.file.write('{"type":"FeatureCollection","features":[\n')

    def way(self, way):

        tags = way.tags

        if not is_drivable(tags):
            return

        # Obtain coordinates from the OSM node references.
        coordinates = []

        node_ids = []

        for node in way.nodes:

            if not node.location.valid():
                self.skipped_count += 1
                return

            coordinates.append([
                node.lon,
                node.lat
            ])

            node_ids.append(str(node.ref))

        if len(coordinates) < 2:
            self.skipped_count += 1
            return

        properties = {
            "osm_way_id": str(way.id),
            "osm_node_ids": node_ids,
            "highway": tags.get("highway"),
            "name": tags.get("name"),
            "ref": tags.get("ref"),
            "lanes": tags.get("lanes"),
            "maxspeed": tags.get("maxspeed"),
            "oneway": tags.get("oneway"),
            "bridge": tags.get("bridge"),
            "tunnel": tags.get("tunnel"),
            "surface": tags.get("surface"),
            "access": tags.get("access"),
            "service": tags.get("service"),
            "junction": tags.get("junction"),
            "width": tags.get("width"),
            "motor_vehicle": tags.get("motor_vehicle"),
            "motorcar": tags.get("motorcar"),
        }

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": properties
        }

        if self.road_count > 0:
            self.file.write(",\n")

        json.dump(feature, self.file)

        self.road_count += 1
        self.highway_counts[tags.get("highway")] += 1

        if self.road_count % 10000 == 0:
            print(f"Extracted {self.road_count:,} road ways")

    def close(self):

        self.file.write("\n]}")
        self.file.close()


# ============================================================
# MAIN
# ============================================================

def main():

    if not OSM_FILE.exists():
        raise FileNotFoundError(OSM_FILE)

    print("Starting streaming OSM road extraction...")
    print(f"Input: {OSM_FILE}")
    print(f"Size: {OSM_FILE.stat().st_size / (1024**3):.2f} GB")

    extractor = RoadExtractor(OUTPUT_FILE)

    try:
        # Disk-backed location index avoids keeping all node
        # coordinates in ordinary Python memory.
        extractor.apply_file(
            str(OSM_FILE),
            locations=True,
            idx="sparse_file_array"
        )

    finally:
        extractor.close()

    summary = {
        "source": "OpenStreetMap",
        "input_file": OSM_FILE.name,
        "output_file": OUTPUT_FILE.name,
        "extracted_road_ways": extractor.road_count,
        "skipped_ways": extractor.skipped_count,
        "highway_counts": dict(extractor.highway_counts),
    }

    with open(
        OUTPUT_DIR / "extraction_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Road ways:", f"{extractor.road_count:,}")
    print("Skipped ways:", f"{extractor.skipped_count:,}")
    print("\nRoad classifications:")

    for highway, count in extractor.highway_counts.most_common():
        print(f"{highway:<20} {count:,}")

    print("\nOutput:", OUTPUT_FILE)
    print("Extraction complete.")


if __name__ == "__main__":
    main()