
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path("data/road_network/topology")

segments = pd.read_csv(
    BASE / "directed_segments.csv",
    low_memory=False
)

edges = pd.read_csv(BASE / "road_segment_edges.csv")

G = nx.DiGraph()

G.add_nodes_from(
    segments["directed_segment_id"]
)

G.add_edges_from(
    zip(edges["from_segment"], edges["to_segment"])
)

# Map segment IDs to road attributes
segment_info = segments.set_index(
    "directed_segment_id"
)

components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

print("Total components:", len(components))

print("\nLARGEST 15 COMPONENTS")
print("-" * 65)

for i, component in enumerate(components[:15], 1):

    subset = segment_info.loc[list(component)]

    road_types = subset["highway"].value_counts()

    print(f"\nComponent {i}")
    print("Segments:", len(component))
    print("Road types:")
    print(road_types.head(8).to_dict())

# Examine components outside the main component
small_components = components[1:]

small_rows = []

for i, component in enumerate(small_components, 2):

    subset = segment_info.loc[list(component)]

    small_rows.append({
        "component_id": i,
        "segment_count": len(component),
        "road_types": ", ".join(
            subset["highway"].value_counts().index[:5]
        ),
        "dominant_road_type": subset["highway"].mode().iloc[0]
    })

small_df = pd.DataFrame(small_rows)

print("\n\nSMALL COMPONENT SUMMARY")
print("-" * 65)

print("\nComponent size distribution:")
print(small_df["segment_count"].describe())

print("\nDominant road types:")
print(
    small_df["dominant_road_type"]
    .value_counts()
)

print("\nComponents with at least 10 segments:")
print(
    small_df[small_df["segment_count"] >= 10]
    .sort_values("segment_count", ascending=False)
    .head(20)
    .to_string(index=False)
)

small_df.to_csv(
    BASE / "component_analysis.csv",
    index=False
)

print("\nSaved component_analysis.csv")