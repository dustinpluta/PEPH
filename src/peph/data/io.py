from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def read_table(path: str | Path, fmt: Literal["csv", "parquet"]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if fmt == "csv":
        return pd.read_csv(p)
    if fmt == "parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported format: {fmt}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf == ".csv":
        df.to_csv(p, index=False)
    elif suf == ".parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {suf}")