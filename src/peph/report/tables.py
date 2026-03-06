from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from peph.report.discover import RunArtifacts
from peph.utils.json import read_json


def load_metrics(art: RunArtifacts) -> Dict[str, Any]:
    if art.metrics_json is None:
        raise FileNotFoundError(f"metrics.json not found under: {art.run_dir}")
    return read_json(str(art.metrics_json))


def load_coef_table(art: RunArtifacts) -> pd.DataFrame:
    if art.coef_table is None:
        raise FileNotFoundError(f"coef_table.parquet not found under: {art.run_dir}")
    return pd.read_parquet(art.coef_table)


def load_baseline_table(art: RunArtifacts) -> pd.DataFrame:
    if art.baseline_table is None:
        raise FileNotFoundError(f"baseline_table.parquet not found under: {art.run_dir}")
    return pd.read_parquet(art.baseline_table)


def load_frailty_table(art: RunArtifacts) -> pd.DataFrame:
    if art.frailty_table is None:
        raise FileNotFoundError(f"frailty_table.parquet not found under: {art.run_dir}")
    return pd.read_parquet(art.frailty_table)


def coef_with_hr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add HR columns to a coefficient table. Expects at least:
      - beta column OR estimate column
      - ci_lo / ci_hi (or conf_low/conf_high)
    This is defensive: if CI columns are missing, HR CI is omitted.
    """
    out = df.copy()

    beta_col = None
    for c in ("beta", "estimate"):
        if c in out.columns:
            beta_col = c
            break
    if beta_col is None:
        return out

    out["hr"] = np.exp(out[beta_col].astype(float))

    lo_col = None
    hi_col = None
    for lo, hi in (("ci_lo", "ci_hi"), ("conf_low", "conf_high"), ("lower", "upper")):
        if lo in out.columns and hi in out.columns:
            lo_col, hi_col = lo, hi
            break

    if lo_col is not None and hi_col is not None:
        out["hr_lo"] = np.exp(out[lo_col].astype(float))
        out["hr_hi"] = np.exp(out[hi_col].astype(float))

    return out


def top_terms(
    df: pd.DataFrame,
    top: int = 20,
    sort: str = "abs_z",
) -> pd.DataFrame:
    """
    sort in {'abs_z','p','abs_beta'}.
    """
    out = df.copy()

    if sort == "p":
        if "p" in out.columns:
            out = out.sort_values("p", ascending=True)
        elif "p_value" in out.columns:
            out = out.sort_values("p_value", ascending=True)
    elif sort == "abs_beta":
        b = "beta" if "beta" in out.columns else ("estimate" if "estimate" in out.columns else None)
        if b:
            out["_abs_beta"] = out[b].astype(float).abs()
            out = out.sort_values("_abs_beta", ascending=False).drop(columns=["_abs_beta"])
    else:
        # abs_z default
        z = "z" if "z" in out.columns else ("z_value" if "z_value" in out.columns else None)
        if z:
            out["_abs_z"] = out[z].astype(float).abs()
            out = out.sort_values("_abs_z", ascending=False).drop(columns=["_abs_z"])

    return out.head(int(top))