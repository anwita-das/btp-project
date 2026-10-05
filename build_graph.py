
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import BallTree
from pathlib import Path

# Paths
INPUT_PATH = Path("data/wzdx_features.csv")
OUTPUT_DIR = Path("data/graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initial spatial threshold: 500 metres
DISTANCE_THRESHOLD_METERS = 500

df = pd.read_csv(INPUT_PATH)

# Validate the required columns
required = [
    "event_id",
    "centroid_lat",
    "centroid_lon",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df["event_id"].duplicated().any():
    raise ValueError("Duplicate event IDs found.")

if df[["centroid_lat", "centroid_lon"]].isna().any().any():
    raise ValueError("Missing centroid coordinates found.")

# BallTree with haversine distance expects radians.
coords_degrees = df[["centroid_lat", "centroid_lon"]].to_numpy()
coords_radians = np.radians(coords_degrees)

earth_radius_meters = 6_371_000
radius_radians = DISTANCE_THRESHOLD_METERS / earth_radius_meters

tree = BallTree(coords_radians, metric="haversine")

# Find all nearby work-zone pairs, including each node itself.
neighbors = tree.query_radius(coords_radians, r=radius_radians)

# Build an undirected graph.
G = nx.Graph()

# Each work zone becomes a node.
for _, row in df.iterrows():
    G.add_node(
        str(row["event_id"]),
        **row.drop(labels=["event_id"]).to_dict()
    )

# Connect nearby nodes. Exclude self-loops and duplicate pairs.
for i, nearby_indices in enumerate(neighbors):
    for j in nearby_indices:
        if j <= i:
            continue

        distance_radians = np.arccos(
            np.clip(
                np.sin(coords_radians[i, 0])
                * np.sin(coords_radians[j, 0])
                + np.cos(coords_radians[i, 0])
                * np.cos(coords_radians[j, 0])
                * np.cos(
                    coords_radians[i, 1] - coords_radians[j, 1]
                ),
                -1.0,
                1.0,
            )
        )

        distance_meters = distance_radians * earth_radius_meters

        G.add_edge(
            str(df.iloc[i]["event_id"]),
            str(df.iloc[j]["event_id"]),
            distance_meters=float(distance_meters),
        )

# Save graph nodes and edges as CSV files.
nodes = pd.DataFrame.from_dict(
    dict(G.nodes(data=True)),
    orient="index"
)
nodes.index.name = "event_id"
nodes.reset_index().to_csv(
    OUTPUT_DIR / "nodes.csv", index=False
)

edges = pd.DataFrame(
    [
        {
            "source": u,
            "target": v,
            **attributes,
        }
        for u, v, attributes in G.edges(data=True)
    ]
)
edges.to_csv(OUTPUT_DIR / "edges.csv", index=False)

# Save graph statistics.
degrees = dict(G.degree())

print("Graph construction complete.")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Isolated nodes:", sum(d == 0 for d in degrees.values()))

if degrees:
    print("Average degree:", round(np.mean(list(degrees.values())), 2))
    print("Maximum degree:", max(degrees.values()))

print("Output directory:", OUTPUT_DIR)
