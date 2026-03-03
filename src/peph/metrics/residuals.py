from __future__ import annotations

import numpy as np
import pandas as pd

from peph.model.predict import predict_cumhaz
from peph.model.result import FittedPEPHModel


def cox_snell_residuals(model: FittedPEPHModel, wide_df: pd.DataFrame, time_col: str) -> np.ndarray:
    """
    r_i = H(T_i | x_i)
    Uses model breaks (truncates at max break).
    """
    t = wide_df[time_col].to_numpy(dtype=float)
    H = predict_cumhaz(model, wide_df, times=t)  # returns (n, n) if times iterable is t
    # predict_cumhaz expects times iterable; if we pass t, we get (n,len(t)).
    # We want diagonal.
    if H.shape[1] != len(t):
        raise RuntimeError("Unexpected shape for cumhaz matrix in cox_snell_residuals")
    r = np.diag(H)
    return r