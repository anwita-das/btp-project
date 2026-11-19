
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path("data/road_network/topology")

segments = pd.read_csv(
    BASE / "directed_segments.csv",
    low_memory=False
)

edges = pd.read_csv(BASE / "road_segment_edges.csv")
nodes = pd.read_csv(BASE / "intersection_nodes.csv")

# Build directed graph
G = nx.DiGraph()

G.add_nodes_from(segments["directed_segment_id"])

G.add_edges_from(
    zip(edges["from_segment"], edges["to_segment"])
)

components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

# Intersection coordinate lookup
node_coords = nodes.set_index("node_id")[
    ["longitude", "latitude"]
]

# Segment lookup
segment_lookup = segments.set_index(
    "directed_segment_id"
)

results = []

# Inspect the 30 largest components, excluding the main one
for component_id, component in enumerate(
    components[1:31], start=2
):

    subset = segment_lookup.loc[list(component)]

    # Get the endpoint intersection IDs
    endpoint_ids = set(subset["from_node"]) | set(subset["to_node"])

    coords = node_coords.loc[
        list(endpoint_ids)
    ]

    results.append({
        "component_id": component_id,
        "segment_count": len(component),
        "dominant_road_type": subset["highway"].mode().iloc[0],
        "road_types": ", ".join(
            subset["highway"].value_counts().index
        ),
        "min_lon": coords["longitude"].min(),
        "max_lon": coords["longitude"].max(),
        "min_lat": coords["latitude"].min(),
        "max_lat": coords["latitude"].max(),
        "center_lon": coords["longitude"].mean(),
        "center_lat": coords["latitude"].mean()
    })

result_df = pd.DataFrame(results)

result_df.to_csv(
    BASE / "component_locations.csv",
    index=False
)

print("\nLARGEST DISCONNECTED COMPONENTS")
print(result_df.to_string(index=False))

print("\nSaved component_locations.csv")