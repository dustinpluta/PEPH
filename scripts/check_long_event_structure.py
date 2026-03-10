from __future__ import annotations

import argparse
import pandas as pd


def detect_interval_column(df: pd.DataFrame) -> str | None:
    candidates = ["k", "interval", "interval_id", "k_interval"]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose event structure in PE long data.")
    parser.add_argument("--long-csv", required=True, help="Path to long-form CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.long_csv)

    print("\n==============================")
    print("LONG DATA EVENT DIAGNOSTICS")
    print("==============================")

    print("\nRows:", len(df))
    print("Total events:", int(df["event"].sum()))

    interval_col = detect_interval_column(df)

    if interval_col is None:
        print("\n⚠ No interval column detected.")
        print("Available columns:")
        print(list(df.columns))
    else:
        print(f"\n--- Events by interval ({interval_col}) ---")
        print(df.groupby(interval_col)["event"].agg(["sum", "count"]))

        if "treated_td" in df.columns:
            print("\n--- Events by interval × treated_td ---")
            print(df.groupby([interval_col, "treated_td"])["event"].agg(["sum", "count"]))

    if "treated_td" in df.columns:
        print("\n--- Events by treated_td ---")
        print(df.groupby("treated_td")["event"].agg(["sum", "count"]))

    for col in ["stage", "sex"]:
        if col in df.columns:
            print(f"\n--- Events by {col} ---")
            print(df.groupby(col)["event"].agg(["sum", "count"]))

    if "zip" in df.columns:
        print("\n--- Lowest event ZIPs ---")
        print(
            df.groupby("zip")["event"]
            .agg(["sum", "count"])
            .sort_values("sum")
            .head(10)
        )

    print("\n==============================")
    print("Potential separation flags")
    print("==============================")

    if interval_col is not None:
        interval_stats = df.groupby(interval_col)["event"].sum()
        if (interval_stats == 0).any():
            print("⚠ Some intervals have ZERO events.")

    if "treated_td" in df.columns:
        treated_stats = df.groupby("treated_td")["event"].sum()
        if (treated_stats == 0).any():
            print("⚠ treated_td level with ZERO events.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()