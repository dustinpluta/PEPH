from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from peph.spatial.graph import SpatialGraph


@dataclass(frozen=True)
class FrailtyOutputs:
    table: pd.DataFrame
    summary: Dict[str, Any]


def build_frailty_outputs(
    *,
    graph: SpatialGraph,
    u_hat: np.ndarray,
    n_train_by_zip: Dict[str, int],
) -> FrailtyOutputs:
    zips = graph.zips
    u = np.asarray(u_hat, dtype=float).ravel()
    if u.size != len(zips):
        raise ValueError("u_hat length must equal number of graph nodes")

    comp = graph.component_ids()
    df = pd.DataFrame(
        {
            "zip": zips,
            "u_hat": u,
            "component": comp,
            "n_train": [int(n_train_by_zip.get(z, 0)) for z in zips],
        }
    )

    # rank (global)
    df = df.sort_values("u_hat").reset_index(drop=True)
    df["rank"] = np.arange(len(df), dtype=int)

    # component-wise means for validation/debug
    comp_means = (
        df.groupby("component")
        .apply(lambda g: float(np.average(g["u_hat"], weights=np.maximum(g["n_train"], 1))))
        .to_dict()
    )

    summary = {
        "n_areas": int(len(df)),
        "u_mean": float(df["u_hat"].mean()),
        "u_sd": float(df["u_hat"].std(ddof=1)) if len(df) > 1 else 0.0,
        "u_min": float(df["u_hat"].min()),
        "u_max": float(df["u_hat"].max()),
        "u_quantiles": {q: float(df["u_hat"].quantile(q)) for q in [0.01,0.05,0.25,0.5,0.75,0.95,0.99]},
        "component_weighted_means": comp_means,
        "n_train_total": int(df["n_train"].sum()),
        "n_train_nonzero_areas": int((df["n_train"] > 0).sum()),
    }
    return FrailtyOutputs(table=df, summary=summary)


def n_train_by_area_from_wide(train_wide: pd.DataFrame, area_col: str) -> Dict[str, int]:
    vc = train_wide[area_col].astype(str).value_counts()
    return {str(k): int(v) for k, v in vc.items()}