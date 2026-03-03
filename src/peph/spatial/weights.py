from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def zip_weights_from_train_wide(
    train_wide: pd.DataFrame,
    *,
    area_col: str,
    zip_to_index: Dict[str, int],
    allow_unseen_area: bool = False,
) -> np.ndarray:
    """
    Compute weights n_g = number of subjects in ZIP g, aligned with graph ordering.

    Uses SUBJECT COUNTS (wide rows / unique id), not long rows.
    """
    if area_col not in train_wide.columns:
        raise ValueError(f"area_col='{area_col}' not found in train_wide")

    z = train_wide[area_col].astype(str).to_numpy()
    G = len(zip_to_index)
    w = np.zeros(G, dtype=float)

    unseen = []
    for zz in z:
        if zz in zip_to_index:
            w[zip_to_index[zz]] += 1.0
        else:
            unseen.append(zz)

    if unseen and not allow_unseen_area:
        unseen_u = sorted(set(unseen))[:10]
        raise ValueError(f"Unseen area values in training data (showing up to 10): {unseen_u}")

    # ensure no zero-weight issues in centering (they're fine; they just don't contribute)
    return w