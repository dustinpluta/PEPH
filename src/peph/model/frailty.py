from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from peph.model.result import FittedPEPHModel


FrailtyMode = Literal["auto", "none", "conditional", "marginal"]


def get_frailty_vector_for_wide(
    wide_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
) -> np.ndarray:
    """
    Return a per-row frailty vector u_i to be added to the linear predictor.

    Modes:
      - auto: if model has spatial frailty -> conditional, else none
      - none: zeros
      - conditional: use fitted u[zip] (hard-fail on unseen unless allow_unseen_area=True)
      - marginal: currently returns zeros (E[u]=0); later can be Laplace-marginal

    If allow_unseen_area is None, defaults to False.
    """
    if allow_unseen_area is None:
        allow_unseen_area = False

    spatial = getattr(model, "spatial", None)
    # PR5 attached via __dict__; getattr won't find it unless result dataclass has field.
    if spatial is None:
        spatial = model.__dict__.get("spatial", None)

    if mode == "auto":
        mode = "conditional" if spatial is not None else "none"

    n = len(wide_df)
    if mode == "none":
        return np.zeros(n, dtype=float)

    if mode == "marginal":
        # E[u]=0 under centered prior; marginal prediction will be refined later.
        return np.zeros(n, dtype=float)

    if spatial is None:
        raise ValueError("Frailty mode requested but model has no spatial frailty attached")

    area_col = spatial.get("area_col")
    if area_col is None:
        raise ValueError("Model spatial metadata missing 'area_col'")
    if area_col not in wide_df.columns:
        raise ValueError(f"wide_df missing area_col='{area_col}' required for conditional frailty")

    zips = spatial.get("zips")
    u = spatial.get("u")
    if zips is None or u is None:
        raise ValueError("Model spatial metadata missing 'zips' or 'u'")

    zip_to_idx = {str(z): i for i, z in enumerate(zips)}
    u_arr = np.asarray(u, dtype=float)

    vals = wide_df[area_col].astype(str).to_numpy()
    out = np.empty(n, dtype=float)

    unseen = []
    for i, zz in enumerate(vals):
        if zz in zip_to_idx:
            out[i] = float(u_arr[zip_to_idx[zz]])
        else:
            unseen.append(zz)
            out[i] = 0.0

    if unseen and not allow_unseen_area:
        unseen_u = sorted(set(unseen))[:10]
        raise ValueError(f"Unseen area values at prediction time (showing up to 10): {unseen_u}")

    return out