
from pathlib import Path
from collections import Counter
import osmium

OSM_FILE = Path("data/raw/austin_road_network.osm")


class OSMInspector(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()
        self.nodes = 0
        self.ways = 0
        self.relations = 0
        self.highway_counts = Counter()

    def node(self, n):
        self.nodes += 1

    def way(self, w):
        self.ways += 1

        highway = w.tags.get("highway")

        if highway:
            self.highway_counts[highway] += 1

    def relation(self, r):
        self.relations += 1


def main():

    if not OSM_FILE.exists():
        raise FileNotFoundError(OSM_FILE)

    print("Inspecting OSM file...")
    print(f"File size: {OSM_FILE.stat().st_size / (1024**3):.2f} GB")

    inspector = OSMInspector()

    inspector.apply_file(str(OSM_FILE))

    print("\n========== OSM SUMMARY ==========")
    print("Nodes:", f"{inspector.nodes:,}")
    print("Ways:", f"{inspector.ways:,}")
    print("Relations:", f"{inspector.relations:,}")

    print("\n========== HIGHWAY TYPES ==========")

    for highway, count in inspector.highway_counts.most_common():
        print(f"{highway:<25} {count:,}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()