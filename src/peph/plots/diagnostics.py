from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from peph.metrics.kaplan_meier import fit_km


def plot_cox_snell(residual: np.ndarray, event: np.ndarray, out_path: str | Path) -> None:
    """
    Cox–Snell plot:
      KM estimate of S_r(r) vs theoretical exp(-r).
    """
    r = np.asarray(residual, dtype=float)
    e = np.asarray(event, dtype=int)
    km = fit_km(r, e)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    if km.times.size > 0:
        plt.step(km.times, km.surv, where="post", label="KM(residual)")
        plt.plot(km.times, np.exp(-km.times), label="exp(-r)")
    else:
        plt.text(0.5, 0.5, "No events for Cox–Snell plot", ha="center", va="center")
    plt.xlabel("Cox–Snell residual r")
    plt.ylabel("Survival")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()