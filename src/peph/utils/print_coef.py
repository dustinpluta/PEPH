from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd


def print_coef_table(
    parquet_path: str | Path,
    *,
    section: Literal["all", "baseline", "covariates"] = "all",
    decimals: int = 4,
    sort_by: Optional[str] = None,
) -> None:
    """
    Pretty-print coefficient table from coef_table.parquet.

    Parameters
    ----------
    parquet_path : path to coef_table.parquet
    section : which rows to print
        - "all"
        - "baseline"   (interval log hazards)
        - "covariates" (non-baseline terms)
    decimals : number of decimals for numeric formatting
    sort_by : optional column to sort by (e.g. "p_value")
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    if "term" not in df.columns:
        raise ValueError("Invalid coefficient table format (missing 'term' column)")

    # Identify baseline terms by naming convention: first K terms are baseline
    # They usually look like "log_nu[0]", etc.
    is_baseline = df["term"].str.contains(r"log_nu|\[", regex=True)

    if section == "baseline":
        df = df[is_baseline].copy()
    elif section == "covariates":
        df = df[~is_baseline].copy()

    if sort_by is not None:
        if sort_by not in df.columns:
            raise ValueError(f"Column '{sort_by}' not found in table")
        df = df.sort_values(sort_by)

    # Format numeric columns
    num_cols = df.select_dtypes(include=["float", "int"]).columns
    df_fmt = df.copy()
    for c in num_cols:
        df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.{decimals}f}")

    # Column rename for readability
    rename_map = {
        "estimate": "Estimate",
        "std_error": "SE",
        "z": "z",
        "p_value": "p",
        "ci_lower": "CI Lower",
        "ci_upper": "CI Upper",
    }
    df_fmt = df_fmt.rename(columns=rename_map)

    print("\nCoefficient Summary")
    print("=" * 72)
    print(df_fmt.to_string(index=False))
    print("=" * 72)
    print(f"Rows shown: {len(df_fmt)}\n")