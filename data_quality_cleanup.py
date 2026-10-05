import pandas as pd
from pathlib import Path

INPUT_CSV = Path("data/wzdx_clean.csv")
OUTPUT_CSV = Path("data/wzdx_analysis.csv")
AUDIT_CSV = Path("data/data_quality_removed.csv")


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Run from your project root."
        )

    original = pd.read_csv(INPUT_CSV, low_memory=False)
    df = original.copy()
    removed_parts = []

    print("=" * 55)
    print("WZDx DATA-QUALITY CLEANUP")
    print("=" * 55)
    print(f"Input rows: {len(df):,}")

    # 1. Remove zero or negative-duration records
    if "duration_hours" in df.columns:
        duration = pd.to_numeric(
            df["duration_hours"], errors="coerce"
        )
        bad_mask = duration.notna() & (duration <= 0)

        if bad_mask.any():
            removed = df.loc[bad_mask].copy()
            removed["removal_reason"] = "non_positive_duration"
            removed_parts.append(removed)

        df = df.loc[~bad_mask].copy()
        print(f"Removed non-positive durations: {bad_mask.sum():,}")

    # 2. Report unknown direction values and remove the column
    if "direction" in df.columns:
        values = (
            df["direction"].astype("string").str.strip().str.lower()
        )
        unknown = values.isin(
            ["unknown", "unspecified", "none", "null", ""]
        ).sum()

        print(f"Unknown/unspecified directions: {unknown:,}")
        df = df.drop(columns=["direction"])
        print("Removed direction column.")
    else:
        print("Direction column already absent.")

    # 3. Remove exact duplicates, ignoring event_id
    # All other remaining fields must match.
    duplicate_columns = [
        col for col in df.columns if col != "event_id"
    ]

    if duplicate_columns:
        duplicate_mask = df.duplicated(
            subset=duplicate_columns,
            keep="first",
        )

        if duplicate_mask.any():
            removed = df.loc[duplicate_mask].copy()
            removed["removal_reason"] = (
                "exact_duplicate_except_event_id"
            )
            removed_parts.append(removed)

        print(
            "Exact duplicate copies removed: "
            f"{duplicate_mask.sum():,}"
        )
        df = df.loc[~duplicate_mask].copy()

    # 4. Save the analysis dataset separately
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # 5. Save removed records with reasons
    if removed_parts:
        audit = pd.concat(
            removed_parts, ignore_index=True, sort=False
        )
    else:
        audit = pd.DataFrame(
            columns=list(original.columns) + ["removal_reason"]
        )

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)

    print("\n" + "=" * 55)
    print(f"Output rows: {len(df):,}")
    print(f"Total removed rows: {len(audit):,}")
    print(f"Analysis dataset: {OUTPUT_CSV}")
    print(f"Removal audit: {AUDIT_CSV}")
    print("Original wzdx_clean.csv was NOT modified.")
    print("=" * 55)


if __name__ == "__main__":
    main()