from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_frailty_caterpillar(
    frailty_df: pd.DataFrame,  # columns: zip,u_hat (sorted or not)
    out_path: str | Path,
    top_k: int = 30,
) -> None:
    """
    Plot top/bottom K frailties (by u_hat) as a caterpillar-style dot plot.
    """
    df = frailty_df[["zip", "u_hat"]].copy().sort_values("u_hat")
    low = df.head(top_k)
    high = df.tail(top_k)
    sub = pd.concat([low, high], axis=0)

    labels = sub["zip"].astype(str).to_list()
    y = sub["u_hat"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hlines(np.arange(len(y)), 0, y, linewidth=1)
    ax.plot(y, np.arange(len(y)), "o")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Frailty estimate (û)")
    ax.set_title(f"Frailty extremes (bottom/top {top_k})")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_morans_scatter(
    x: np.ndarray,
    Wx: np.ndarray,
    out_path: str | Path,
    title: str = "Moran scatterplot",
) -> None:
    """
    Moran scatterplot: x vs spatial lag Wx.
    Caller supplies Wx = W_row_norm @ x or (W @ x)/row_sum.
    """
    x = np.asarray(x, float).ravel()
    Wx = np.asarray(Wx, float).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, Wx, "o", markersize=3)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("x (centered)")
    ax.set_ylabel("Spatial lag W x")
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_calibration_by_bin(
    cal_df: pd.DataFrame,  # columns: bin, mean_pred, obs_rate
    out_path: str | Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cal_df["mean_pred"].to_numpy(), cal_df["obs_rate"].to_numpy(), "o-")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)