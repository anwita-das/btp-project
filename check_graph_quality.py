
import pandas as pd

nodes = pd.read_csv("data/graph/nodes.csv")
edges = pd.read_csv("data/graph/edges.csv")

# Add event dates to both ends of every edge.
cols = ["event_id", "start_time", "end_time", "event_name", "road_names"]

source = nodes[cols].rename(columns={
    "event_id": "source",
    "start_time": "source_start",
    "end_time": "source_end",
    "event_name": "source_name",
    "road_names": "source_road",
})

target = nodes[cols].rename(columns={
    "event_id": "target",
    "start_time": "target_start",
    "end_time": "target_end",
    "event_name": "target_name",
    "road_names": "target_road",
})

edges = edges.merge(source, on="source", how="left")
edges = edges.merge(target, on="target", how="left")

for col in ["source_start", "source_end", "target_start", "target_end"]:
    edges[col] = pd.to_datetime(edges[col], utc=True, errors="coerce")

# Two events overlap if each starts before the other ends.
edges["temporal_overlap"] = (
    (edges["source_start"] <= edges["target_end"])
    & (edges["target_start"] <= edges["source_end"])
)

print("Total edges:", len(edges))
print("Edges with overlapping time periods:",
      int(edges["temporal_overlap"].sum()))
print("Edges without overlapping time periods:",
      int((~edges["temporal_overlap"]).sum()))

print("\nSample edges without temporal overlap:")
print(
    edges.loc[
        ~edges["temporal_overlap"],
        ["source_road", "target_road", "distance_meters"]
    ].head(10).to_string(index=False)
)