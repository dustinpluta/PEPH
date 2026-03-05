from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def known_status_mask(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    """
    Keep subjects with known event status at horizon:
      - time > horizon  (alive past horizon, regardless of later event)
      - OR event==1 and time <= horizon (event before/at horizon)
    Drop event==0 and time <= horizon (censored before horizon).
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    return (time > horizon) | ((event == 1) & (time <= horizon))


def calibration_by_frailty_decile(
    *,
    df_wide: pd.DataFrame,
    time_col: str,
    event_col: str,
    risk_col: str,
    zip_col: str,
    frailty_table: pd.DataFrame,  # columns zip,u_hat
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin ZIPs by frailty decile (based on u_hat), then compute within-bin:
      - n subjects used (known status at horizon)
      - mean predicted risk
      - observed event rate at horizon
      - calibration-in-the-large (obs - pred)
    """
    ft = frailty_table[["zip", "u_hat"]].copy()
    ft["zip"] = ft["zip"].astype(str)

    d = df_wide.copy()
    d[zip_col] = d[zip_col].astype(str)
    d = d.merge(ft, how="left", left_on=zip_col, right_on="zip", validate="many_to_one")

    if d["u_hat"].isna().any():
        # if allow_unseen_area=True, user may see NaNs here; those should be handled upstream
        raise ValueError("Missing frailty values for some rows (unseen area?)")

    # Compute bin edges on ZIP-level u_hat, not subject-level
    # Use qcut over unique zips to avoid overweighting big zips
    zip_bins = (
        ft.drop_duplicates("zip")
        .assign(bin=lambda x: pd.qcut(x["u_hat"], q=n_bins, labels=False, duplicates="drop"))
    )
    d = d.merge(zip_bins[["zip", "bin"]], on="zip", how="left", validate="many_to_one")

    if d["bin"].isna().any():
        raise ValueError("Failed to assign frailty bins (check u_hat distribution)")

    time = d[time_col].to_numpy(dtype=float)
    event = d[event_col].to_numpy(dtype=int)

    # Horizon is encoded in the risk_col name; caller supplies correct risk_col per horizon.
    # We need horizon separately in the pipeline; keep it there.
    # Here compute observed at horizon using provided horizon stored on df? Not present.
    # So caller should precompute y and pass? To keep API minimal, we infer horizon from risk_col is not robust.
    # => Caller should pass y_col; but for PR9 keep simple: require df contains y_horizon column.
    raise NotImplementedError("Use the pipeline wrapper that supplies y_horizon explicitly.")