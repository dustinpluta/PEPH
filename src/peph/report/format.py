from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "NA"
    except Exception:
        pass
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def format_metrics_summary(metrics: Dict[str, Any], horizons: Iterable[int] = (365, 730, 1825)) -> pd.DataFrame:
    """
    Convert flat metrics dict into a compact table.
    """
    rows = []
    rows.append(("c_index", _fmt(metrics.get("c_index"))))

    for h in horizons:
        rows.append((f"auc_t{h}", _fmt(metrics.get(f"auc_t{h}"))))
    for h in horizons:
        rows.append((f"brier_t{h}", _fmt(metrics.get(f"brier_t{h}"))))
    for h in horizons:
        rows.append((f"cal_int_t{h}", _fmt(metrics.get(f"cal_int_t{h}"))))
    for h in horizons:
        rows.append((f"cal_slope_t{h}", _fmt(metrics.get(f"cal_slope_t{h}"))))

    # residual summary
    if "cox_snell_mean" in metrics or "cox_snell_var" in metrics:
        rows.append(("cox_snell_mean", _fmt(metrics.get("cox_snell_mean"))))
        rows.append(("cox_snell_var", _fmt(metrics.get("cox_snell_var"))))

    # spatial summary (optional)
    if any(k in metrics for k in ("leroux_rho_hat", "leroux_tau_hat", "morans_I_u", "morans_I_u_z")):
        rows.append(("leroux_rho_hat", _fmt(metrics.get("leroux_rho_hat"))))
        rows.append(("leroux_tau_hat", _fmt(metrics.get("leroux_tau_hat"))))
        rows.append(("morans_I_u", _fmt(metrics.get("morans_I_u"))))
        rows.append(("morans_I_u_z", _fmt(metrics.get("morans_I_u_z"))))

    return pd.DataFrame(rows, columns=["metric", "value"])


def print_df_pretty(df: pd.DataFrame, max_rows: int = 40, max_cols: int = 50) -> None:
    with pd.option_context(
        "display.max_rows", int(max_rows),
        "display.max_columns", int(max_cols),
        "display.width", 120,
        "display.expand_frame_repr", False,
    ):
        print(df.to_string(index=False))