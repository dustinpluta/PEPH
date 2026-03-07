from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignInfo:
    """
    Metadata describing the constructed design.

    baseline_col_names:
        Names for baseline (interval) parameters, length K.
    x_col_names:
        Names for expanded fixed-effect covariates (numeric + one-hot), length p.
    param_names:
        Full parameter names: baseline_col_names + x_col_names, length K+p.
    categorical_levels_seen:
        Dict mapping each categorical column -> sorted unique levels observed in training data.
    """
    baseline_col_names: List[str]
    x_col_names: List[str]
    param_names: List[str]
    categorical_levels_seen: Dict[str, List[str]]


def _validate_unique_covariate_names(
    *,
    x_numeric: List[str],
    x_td_numeric: List[str],
    x_categorical: List[str],
) -> None:
    """
    Fail if a covariate name is repeated across groups.
    """
    all_names = list(x_numeric) + list(x_td_numeric) + list(x_categorical)
    dupes = sorted({c for c in all_names if all_names.count(c) > 1})
    if dupes:
        raise ValueError(
            "Covariate names must be unique across x_numeric, x_td_numeric, and "
            f"x_categorical. Duplicates: {dupes}"
        )


def _encode_fixed_effects(
    *,
    long_df: pd.DataFrame,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    x_td_numeric: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """
    Encode fixed effects from LONG data.

    Numeric:
        - x_numeric: baseline numeric covariates carried into long form
        - x_td_numeric: time-dependent numeric covariates already present in long_df

    Categorical:
        One-hot using sorted levels observed in long_df; exclude reference.

    Column naming:
        f"{col}{level}" for categorical indicators (e.g. sexM, stageIII)
    """
    if x_td_numeric is None:
        x_td_numeric = []

    n = len(long_df)

    _validate_unique_covariate_names(
        x_numeric=x_numeric,
        x_td_numeric=x_td_numeric,
        x_categorical=x_categorical,
    )

    for c in x_numeric:
        if c not in long_df.columns:
            raise ValueError(f"Numeric covariate column not found in long_df: '{c}'")

    for c in x_td_numeric:
        if c not in long_df.columns:
            raise ValueError(
                f"Time-dependent numeric covariate column not found in long_df: '{c}'"
            )

    for c in x_categorical:
        if c not in long_df.columns:
            raise ValueError(f"Categorical covariate column not found in long_df: '{c}'")
        if c not in categorical_reference_levels:
            raise ValueError(f"Missing reference level for categorical '{c}'")

    blocks: List[np.ndarray] = []
    names: List[str] = []
    levels_seen: Dict[str, List[str]] = {}

    if x_numeric:
        X_num = long_df[x_numeric].to_numpy(dtype=float, copy=False)
        blocks.append(X_num)
        names.extend(list(x_numeric))

    if x_td_numeric:
        X_td = long_df[x_td_numeric].to_numpy(dtype=float, copy=False)
        blocks.append(X_td)
        names.extend(list(x_td_numeric))

    for col in x_categorical:
        ref = str(categorical_reference_levels[col])
        vals = long_df[col].astype(str)

        lvls = sorted(pd.unique(vals))
        levels_seen[col] = lvls
        nonref = [lvl for lvl in lvls if lvl != ref]

        if nonref:
            X_cat = np.zeros((n, len(nonref)), dtype=float)
            v = vals.to_numpy()
            for j, lvl in enumerate(nonref):
                X_cat[:, j] = (v == lvl).astype(float)
                names.append(f"{col}{lvl}")
            blocks.append(X_cat)

    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0), dtype=float)
    return X, names, levels_seen


def _encode_fixed_effects_from_wide(
    *,
    wide_df: pd.DataFrame,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    categorical_levels_seen: Dict[str, List[str]],
    hard_fail_on_unseen: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode fixed effects from WIDE data for prediction using levels from training.

    Notes
    -----
    This path remains baseline-only. Time-dependent covariates such as treated_td
    are handled in LONG training design and will require a separate prediction path.
    """
    n = len(wide_df)

    for c in x_numeric:
        if c not in wide_df.columns:
            raise ValueError(f"Numeric covariate column not found in wide_df: '{c}'")

    for c in x_categorical:
        if c not in wide_df.columns:
            raise ValueError(f"Categorical covariate column not found in wide_df: '{c}'")
        if c not in categorical_reference_levels:
            raise ValueError(f"Missing reference level for categorical '{c}'")
        if c not in categorical_levels_seen:
            raise ValueError(f"Missing categorical_levels_seen for '{c}'")

    blocks: List[np.ndarray] = []
    names: List[str] = []

    if x_numeric:
        X_num = wide_df[x_numeric].to_numpy(dtype=float, copy=False)
        blocks.append(X_num)
        names.extend(list(x_numeric))

    for col in x_categorical:
        ref = str(categorical_reference_levels[col])
        train_lvls = list(map(str, categorical_levels_seen[col]))
        train_set = set(train_lvls)

        vals = wide_df[col].astype(str).to_numpy()
        if hard_fail_on_unseen:
            unseen = sorted(set(vals) - train_set)
            if unseen:
                raise ValueError(f"Unseen categorical level(s) for '{col}': {unseen[:10]}")

        nonref = [lvl for lvl in train_lvls if lvl != ref]
        if nonref:
            X_cat = np.zeros((n, len(nonref)), dtype=float)
            for j, lvl in enumerate(nonref):
                X_cat[:, j] = (vals == lvl).astype(float)
                names.append(f"{col}{lvl}")
            blocks.append(X_cat)

    X = np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0), dtype=float)
    return X, names


def build_design_long_train(
    long_df: pd.DataFrame,
    *,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    K: int,
    x_td_numeric: Optional[List[str]] = None,
    y_col: str = "event",
    interval_col: str = "k",
    exposure_col: str = "exposure",
    eps_offset: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DesignInfo]:
    """
    Build Poisson-trick GLM design for PE-PH training on long-form data.

    log(mu_i) = log(exposure_i) + alpha_{k_i} + X_i beta
    """
    if x_td_numeric is None:
        x_td_numeric = []

    if y_col not in long_df.columns:
        raise ValueError(f"Missing y_col '{y_col}' in long_df")
    if interval_col not in long_df.columns:
        raise ValueError(f"Missing interval_col '{interval_col}' in long_df")
    if exposure_col not in long_df.columns:
        raise ValueError(f"Missing exposure_col '{exposure_col}' in long_df")

    n = len(long_df)
    if n == 0:
        raise ValueError("long_df is empty")

    y = long_df[y_col].to_numpy(dtype=float)
    k = long_df[interval_col].to_numpy(dtype=int)
    exposure = long_df[exposure_col].to_numpy(dtype=float)

    if np.any(k < 0) or np.any(k >= K):
        raise ValueError(f"Interval index '{interval_col}' out of range for K={K}")

    if np.any(exposure < 0):
        raise ValueError("exposure must be non-negative")

    offset = np.log(np.maximum(exposure, float(eps_offset)))

    B = np.zeros((n, K), dtype=float)
    B[np.arange(n), k] = 1.0
    baseline_col_names = [f"log_nu[{j}]" for j in range(K)]

    X_fixed, x_col_names, levels_seen = _encode_fixed_effects(
        long_df=long_df,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        x_td_numeric=x_td_numeric,
    )

    X_full = np.concatenate([B, X_fixed], axis=1)
    param_names = baseline_col_names + x_col_names

    info = DesignInfo(
        baseline_col_names=baseline_col_names,
        x_col_names=x_col_names,
        param_names=param_names,
        categorical_levels_seen=levels_seen,
    )
    return y, X_full, offset, info


def build_x_wide_for_prediction(
    wide_df: pd.DataFrame,
    *,
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    categorical_levels_seen: Dict[str, List[str]],
    x_col_names: Optional[List[str]] = None,
    hard_fail_on_unseen: bool = True,
    hard_fail: Optional[bool] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build fixed-effect design matrix for prediction on WIDE data.

    Backward-compatible kwargs:
      - hard_fail: alias for hard_fail_on_unseen
      - x_col_names: enforce exact column ordering
    """
    if hard_fail is not None:
        hard_fail_on_unseen = bool(hard_fail)

    X_raw, names_raw = _encode_fixed_effects_from_wide(
        wide_df=wide_df,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        categorical_levels_seen=categorical_levels_seen,
        hard_fail_on_unseen=hard_fail_on_unseen,
    )

    if x_col_names is None:
        return X_raw, names_raw

    expected = list(x_col_names)
    got = list(names_raw)

    if set(got) != set(expected):
        missing = [c for c in expected if c not in got]
        extra = [c for c in got if c not in expected]
        raise ValueError(
            "Prediction design columns do not match model encoding.\n"
            f"Missing: {missing[:20]}\n"
            f"Extra: {extra[:20]}"
        )

    idx = [got.index(c) for c in expected]
    X = X_raw[:, idx]
    return X, expected