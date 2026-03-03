from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from peph.model.design import _encode_fixed_effects  # see note below


@dataclass(frozen=True)
class LongComponentsInfo:
    K: int
    x_col_names: List[str]
    categorical_levels_seen: Dict[str, List[str]]


def build_long_components(
    long_df: pd.DataFrame,
    *,
    K: int,
    y_col: str = "event",
    exposure_col: str = "exposure",
    interval_col: str = "k",
    id_col: str = "id",
    area_col: Optional[str] = None,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    zip_to_index: Optional[Dict[str, int]] = None,
    allow_unseen_area: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], LongComponentsInfo]:
    """
    Returns:
      y        shape (n,)
      exposure shape (n,)
      k        shape (n,) interval index in [0,K-1]
      X        shape (n,p) fixed effects design (no baseline dummies)
      area_idx shape (n,) integer in [0,G-1] or None if area_col None
      info
    """
    for col in [y_col, exposure_col, interval_col, id_col]:
        if col not in long_df.columns:
            raise ValueError(f"Missing required long_df column: '{col}'")

    y = long_df[y_col].to_numpy(dtype=float)
    exposure = long_df[exposure_col].to_numpy(dtype=float)
    k = long_df[interval_col].to_numpy(dtype=int)
    if np.any(k < 0) or np.any(k >= K):
        raise ValueError("Interval index k out of range")

    # Build fixed effects design (no baseline)
    X, x_col_names, levels_seen = _encode_fixed_effects(
        long_df=long_df,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
    )

    area_idx: Optional[np.ndarray] = None
    if area_col is not None:
        if area_col not in long_df.columns:
            raise ValueError(f"area_col='{area_col}' not present in long_df")
        if zip_to_index is None:
            raise ValueError("zip_to_index must be provided when area_col is set")

        z = long_df[area_col].astype(str).to_numpy()
        idx = np.empty(z.shape[0], dtype=int)
        unseen = []
        for i, zz in enumerate(z):
            if zz in zip_to_index:
                idx[i] = int(zip_to_index[zz])
            else:
                unseen.append(zz)
                idx[i] = -1

        if unseen and not allow_unseen_area:
            unseen_u = sorted(set(unseen))[:10]
            raise ValueError(
                f"Unseen area values at prediction/transform time (showing up to 10): {unseen_u}"
            )
        area_idx = idx

    info = LongComponentsInfo(
        K=int(K),
        x_col_names=x_col_names,
        categorical_levels_seen=levels_seen,
    )
    return y, exposure, k, X, area_idx, info