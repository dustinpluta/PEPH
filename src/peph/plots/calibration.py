from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peph.metrics.calibration import observed_risk_ipcw


def plot_calibration_risk_by_quantile(
    df_pred: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    risk_col: str,
    tau: float,
    n_bins: int,
    out_path: str | Path,
) -> None:
    """
    Bin by predicted risk quantiles; compare mean predicted vs observed (IPCW) risk within bin.
    Observed risk within bin estimated by IPCW mean of I(T<=tau, event=1).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = df_pred[risk_col].to_numpy(dtype=float)
    # quantile bins; drop duplicate edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size < 3:
        plt.figure()
        plt.text(0.5, 0.5, "Not enough unique risk values for calibration bins", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    bins = pd.cut(df_pred[risk_col], bins=edges, include_lowest=True, duplicates="drop")
    groups = df_pred.groupby(bins, observed=True)

    pred_means = []
    obs_means = []

    for _, g in groups:
        pred_means.append(float(g[risk_col].mean()))
        obs_means.append(
            observed_risk_ipcw(
                g[time_col].to_numpy(dtype=float),
                g[event_col].to_numpy(dtype=int),
                float(tau),
            )
        )

    plt.figure()
    plt.plot(pred_means, obs_means, marker="o")
    lo = min(pred_means + obs_means)
    hi = max(pred_means + obs_means)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed risk (IPCW)")
    plt.title(f"Calibration risk plot at t={int(tau)}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()