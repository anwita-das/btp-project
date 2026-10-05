import pandas as pd

df = pd.read_csv("data/wzdx_analysis.csv")

print("Shape:", df.shape)

print("\nColumns and data types:")
print(df.dtypes.to_string())

print("\nSample records:")
print(df.head(3).to_string())

print("\nVerification flag distributions:")
verification_cols = [
    col for col in df.columns
    if "verified" in col.lower()
]

for col in verification_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

print("\nMissing values:")
print(df.isna().sum().to_string())