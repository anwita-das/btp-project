
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter

BASE = Path("data/road_network/topology")

nodes = pd.read_csv(BASE / "intersection_nodes.csv")
segments = pd.read_csv(BASE / "directed_segments.csv")
edges = pd.read_csv(BASE / "road_segment_edges.csv")

print("=" * 55)
print("ROAD TOPOLOGY VALIDATION")
print("=" * 55)

# 1. Basic counts
print("\n1. BASIC COUNTS")
print("Intersection nodes:", len(nodes))
print("Directed segments:", len(segments))
print("Segment connections:", len(edges))

# 2. Degree distribution
print("\n2. INTERSECTION DEGREE DISTRIBUTION")
print(nodes["degree"].describe())

print("\nDegree counts:")
print(nodes["degree"].value_counts().sort_index().head(15))

# 3. Directed segment graph
G = nx.DiGraph()

G.add_nodes_from(segments["directed_segment_id"])
G.add_edges_from(
    zip(edges["from_segment"], edges["to_segment"])
)

print("\n3. DIRECTED ROAD-SEGMENT GRAPH")
print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

print("Weakly connected components:",
      nx.number_weakly_connected_components(G))

print("Strongly connected components:",
      nx.number_strongly_connected_components(G))

# Largest weak components
components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

print("\nLargest 10 weakly connected components:")
for i, component in enumerate(components[:10], 1):
    print(i, len(component))

# 4. Check edge references
segment_ids = set(segments["directed_segment_id"])

invalid_from = set(edges["from_segment"]) - segment_ids
invalid_to = set(edges["to_segment"]) - segment_ids

print("\n4. EDGE REFERENCE CHECK")
print("Invalid from-segment references:", len(invalid_from))
print("Invalid to-segment references:", len(invalid_to))

# 5. Duplicate edges
duplicate_edges = edges.duplicated(
    subset=["from_segment", "to_segment"]
).sum()

print("\n5. DUPLICATE CONNECTIONS")
print("Duplicate directed connections:", duplicate_edges)

# 6. Self loops
self_loops = (
    edges["from_segment"] == edges["to_segment"]
).sum()

print("\n6. SELF LOOPS")
print("Self-loop connections:", self_loops)

# 7. Road type distribution
print("\n7. ROAD TYPE DISTRIBUTION")
print(segments["highway"].value_counts())

# 8. Segment length checks
print("\n8. SEGMENT LENGTH CHECK")
print(segments["length_m"].describe())

print("Zero or negative lengths:",
      (segments["length_m"] <= 0).sum())

print("Segments shorter than 5 metres:",
      (segments["length_m"] < 5).sum())

print("Segments longer than 5 km:",
      (segments["length_m"] > 5000).sum())

# 9. Direction distribution
print("\n9. DIRECTION DISTRIBUTION")
print(segments["direction"].value_counts())

# 10. Save report
report = {
    "intersection_nodes": len(nodes),
    "directed_segments": len(segments),
    "graph_edges": len(edges),
    "weak_components": len(components),
    "largest_component_size": len(components[0]) if components else 0,
    "duplicate_edges": int(duplicate_edges),
    "self_loops": int(self_loops),
    "invalid_from_references": len(invalid_from),
    "invalid_to_references": len(invalid_to),
    "short_segments_under_5m": int(
        (segments["length_m"] < 5).sum()
    ),
    "long_segments_over_5km": int(
        (segments["length_m"] > 5000).sum()
    )
}

pd.Series(report).to_json(
    BASE / "validation_report.json",
    indent=4
)

print("\nValidation report saved.")
print("\nVALIDATION COMPLETE")