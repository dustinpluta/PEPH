from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignInfo:
    K: int
    baseline_col_names: List[str]
    x_col_names: List[str]  # expanded X names (numeric + one-hot)
    param_names: List[str]  # baseline + x
    categorical_levels_seen: Dict[str, List[str]]


def _sorted_levels(series: pd.Series) -> List[str]:
    # treat values as strings for stable JSON + comparisons
    levels = pd.Series(series.astype("string")).dropna().unique().tolist()
    levels = sorted([str(x) for x in levels])
    return levels


def _one_hot_with_reference(
    s: pd.Series,
    *,
    var_name: str,
    levels_seen: List[str],
    reference: str,
    hard_fail: bool,
) -> Tuple[np.ndarray, List[str]]:
    s_str = s.astype("string").fillna(pd.NA)
    present = set([str(x) for x in pd.Series(s_str).dropna().unique().tolist()])
    allowed = set(levels_seen)

    unseen = sorted(list(present - allowed))
    if unseen and hard_fail:
        raise ValueError(
            f"Unseen categorical levels for '{var_name}': {unseen}. "
            f"Allowed levels from training: {levels_seen}"
        )

    # columns for all non-reference levels (in training order)
    cols = [lvl for lvl in levels_seen if lvl != reference]
    X = np.zeros((len(s_str), len(cols)), dtype=float)

    # encode: 1 if equals lvl, 0 otherwise; unseen levels remain 0s if not hard-failing
    s_vals = s_str.astype("string").to_numpy()
    for j, lvl in enumerate(cols):
        X[:, j] = (s_vals == lvl).astype(float)

    col_names = [f"{var_name}{lvl}" for lvl in cols]
    return X, col_names


def build_design_long_train(
    long_df: pd.DataFrame,
    *,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    K: int,
    eps_offset: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DesignInfo]:
    """
    Build (y, X, offset) for Poisson GLM on long-form TRAINING data.

    X includes:
      - baseline interval indicators: K columns, no intercept
      - expanded covariates: numeric + one-hot categoricals (drop reference)

    offset = log(max(exposure, eps_offset))
    """
    if "k" not in long_df.columns:
        raise ValueError("long_df must contain interval index column 'k'")
    if "exposure" not in long_df.columns or "event" not in long_df.columns:
        raise ValueError("long_df must contain 'exposure' and 'event'")

    n = len(long_df)

    # Baseline interval indicators
    k = long_df["k"].astype(int).to_numpy()
    if k.min(initial=0) < 0 or k.max(initial=-1) >= K:
        raise ValueError(f"Interval index k out of bounds for K={K}")

    B = np.zeros((n, K), dtype=float)
    B[np.arange(n), k] = 1.0
    baseline_col_names = [f"log_nu[{j}]" for j in range(K)]

    # Covariates
    parts = []
    x_col_names: List[str] = []

    if x_numeric:
        Xn = long_df[x_numeric].to_numpy(dtype=float)
        parts.append(Xn)
        x_col_names.extend(list(x_numeric))

    categorical_levels_seen: Dict[str, List[str]] = {}
    for c in x_categorical:
        ref = categorical_reference_levels[c]
        levels = _sorted_levels(long_df[c])
        if ref not in levels:
            raise ValueError(
                f"Reference level '{ref}' for '{c}' not present in training data. "
                f"Observed levels: {levels}"
            )
        categorical_levels_seen[c] = levels

        Xc, names = _one_hot_with_reference(
            long_df[c],
            var_name=c,
            levels_seen=levels,
            reference=ref,
            hard_fail=False,  # train: no need to hard-fail; levels_seen derived here
        )
        parts.append(Xc)
        x_col_names.extend(names)

    Xx = np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=float)

    # full X: [B | Xx]
    X = np.concatenate([B, Xx], axis=1)

    y = long_df["event"].astype(int).to_numpy()

    exposure = long_df["exposure"].to_numpy(dtype=float)
    offset = np.log(np.maximum(exposure, eps_offset))

    param_names = baseline_col_names + x_col_names
    info = DesignInfo(
        K=K,
        baseline_col_names=baseline_col_names,
        x_col_names=x_col_names,
        param_names=param_names,
        categorical_levels_seen=categorical_levels_seen,
    )
    return y, X, offset, info


def build_x_wide_for_prediction(
    wide_df: pd.DataFrame,
    *,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    categorical_levels_seen: Dict[str, List[str]],
    x_col_names: List[str],
    hard_fail: bool = True,
) -> np.ndarray:
    """
    Build expanded X (numeric + one-hot) for WIDE prediction data, matching training columns exactly.
    """
    parts = []
    col_names: List[str] = []

    if x_numeric:
        missing = [c for c in x_numeric if c not in wide_df.columns]
        if missing:
            raise ValueError(f"Missing numeric columns at prediction: {missing}")
        parts.append(wide_df[x_numeric].to_numpy(dtype=float))
        col_names.extend(list(x_numeric))

    for c in x_categorical:
        if c not in wide_df.columns:
            raise ValueError(f"Missing categorical column at prediction: '{c}'")
        ref = categorical_reference_levels[c]
        levels = categorical_levels_seen[c]
        Xc, names = _one_hot_with_reference(
            wide_df[c],
            var_name=c,
            levels_seen=levels,
            reference=ref,
            hard_fail=hard_fail,
        )
        parts.append(Xc)
        col_names.extend(names)

    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(wide_df), 0), dtype=float)

    if col_names != x_col_names:
        raise RuntimeError(
            "Expanded prediction columns do not match training columns.\n"
            f"Expected: {x_col_names}\n"
            f"Got:      {col_names}"
        )
    return X