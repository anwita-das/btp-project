
import pandas as pd

df = pd.read_csv("data/wzdx_features.csv")

print("WZDx event coordinate ranges:")
print("Latitude:", df["centroid_lat"].min(), "to", df["centroid_lat"].max())
print("Longitude:", df["centroid_lon"].min(), "to", df["centroid_lon"].max())

south, west = 30.108, -97.930
north, east = 30.497, -97.571

inside = (
    df["centroid_lat"].between(south, north)
    & df["centroid_lon"].between(west, east)
)

print("\nTotal events:", len(df))
print("Inside map bounds:", inside.sum())
print("Outside map bounds:", (~inside).sum())