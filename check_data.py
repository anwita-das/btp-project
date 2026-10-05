import pandas as pd
from pathlib import Path

INPUT_CSV = Path("data/wzdx_clean.csv")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. Run build_data.py first "
            "and execute this script from the project root."
        )

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print("=" * 60)
    print("WZDx DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # 1. Missing values
    print("\n[1] MISSING VALUES")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        for column, count in missing.items():
            print(f"{column}: {count:,} ({count / len(df):.1%})")

    # 2. Duplicate records
    print("\n[2] DUPLICATES")

    if "event_id" in df.columns:
        duplicates = df["event_id"].duplicated().sum()
        print(f"Duplicate event_id rows: {duplicates:,}")

    candidate_cols = [
        col for col in [
            "road_names",
            "start_time",
            "end_time",
            "centroid_lat",
            "centroid_lon",
        ]
        if col in df.columns
    ]

    if candidate_cols:
        possible_duplicates = df.duplicated(
            subset=candidate_cols
        ).sum()

        print(
            "Possible duplicates based on road/date/location: "
            f"{possible_duplicates:,}"
        )
        print("These are candidates for inspection, not automatic deletions.")

    # 3. Verification flags
    print("\n[3] VERIFICATION FLAGS")

    flag_cols = [
        "start_date_verified",
        "end_date_verified",
        "start_position_verified",
        "end_position_verified",
    ]

    for col in flag_cols:
        if col in df.columns:
            values = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            print(f"\n{col}:")
            print(values.value_counts(dropna=False).to_string())

    # 4. Duration checks
    print("\n[4] DURATION CHECK")

    if "duration_hours" in df.columns:
        duration = pd.to_numeric(
            df["duration_hours"], errors="coerce"
        )

        print("\nDuration statistics (hours):")
        print(
            duration.describe(
                percentiles=[0.01, 0.10, 0.50, 0.90, 0.99]
            ).to_string()
        )

        print(f"\nDuration <= 0 hours: {(duration <= 0).sum():,}")
        print(
            "Duration > 30 days: "
            f"{(duration > 30 * 24).sum():,}"
        )
        print(
            "Duration > 180 days: "
            f"{(duration > 180 * 24).sum():,}"
        )
        print(
            "Duration > 365 days: "
            f"{(duration > 365 * 24).sum():,}"
        )

        long_cols = [
            col for col in [
                "event_id",
                "event_name",
                "road_names",
                "start_time",
                "end_time",
                "duration_days",
            ]
            if col in df.columns
        ]

        long_events = df.loc[
            duration > 180 * 24, long_cols
        ]

        if not long_events.empty:
            print("\nExamples lasting more than 180 days:")
            print(long_events.head(10).to_string(index=False))
            print("Review these records; do not delete them automatically.")

    # 5. Vehicle impact
    print("\n[5] VEHICLE IMPACT CATEGORIES")

    if "vehicle_impact" in df.columns:
        print(
            df["vehicle_impact"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "vehicle_impact_score" in df.columns:
        print("\nVehicle impact score distribution:")

        scores = pd.to_numeric(
            df["vehicle_impact_score"], errors="coerce"
        )

        print(
            scores.value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    # 6. Coordinate validity
    print("\n[6] COORDINATE CHECK")

    longitude_cols = [
        "start_lon",
        "end_lon",
        "centroid_lon",
    ]

    latitude_cols = [
        "start_lat",
        "end_lat",
        "centroid_lat",
    ]

    for col in longitude_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = values.notna() & ~values.between(-180, 180)
            print(f"{col} outside valid range: {invalid.sum():,}")

    for col in latitude_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = values.notna() & ~values.between(-90, 90)
            print(f"{col} outside valid range: {invalid.sum():,}")

    # Rough geographic sanity check for the Austin dataset
    if {"centroid_lat", "centroid_lon"}.issubset(df.columns):
        lat = pd.to_numeric(df["centroid_lat"], errors="coerce")
        lon = pd.to_numeric(df["centroid_lon"], errors="coerce")

        outside_austin = ~(
            lat.between(29.0, 31.0)
            & lon.between(-98.5, -97.0)
        )

        print(
            "\nCentroids outside rough Austin-area bounds: "
            f"{outside_austin.sum():,}"
        )
        print("These bounds are approximate, not an exact city boundary.")

    # 7. Timestamp parsing
    print("\n[7] TIMESTAMP CHECK")

    for col in ["start_time", "end_time"]:
        if col in df.columns:
            parsed = pd.to_datetime(
                df[col], errors="coerce", utc=True
            )

            print(
                f"{col} values that fail parsing: "
                f"{parsed.isna().sum():,}"
            )

    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("Inspect unusual records before changing or deleting them.")
    print("=" * 60)


if __name__ == "__main__":
    main()