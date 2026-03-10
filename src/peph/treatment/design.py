from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from peph.treatment.result import TreatmentFeatureEncoding


def _validate_reference_levels(
    df: pd.DataFrame,
    *,
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
) -> None:
    missing = [c for c in x_categorical if c not in categorical_reference_levels]
    if missing:
        raise ValueError(
            "Missing categorical reference levels for columns: "
            f"{missing}"
        )

    for c in x_categorical:
        vals = df[c].astype(str)
        ref = str(categorical_reference_levels[c])
        if ref not in set(vals.unique()):
            raise ValueError(
                f"Reference level '{ref}' for categorical column '{c}' "
                "was not observed in the fitting data."
            )


def _categorical_levels_seen_from_fit(
    df: pd.DataFrame,
    *,
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
) -> dict[str, list[str]]:
    """
    Deterministically record the levels seen in fitting data.

    Convention:
    - cast to string
    - sort unique levels lexicographically
    - require chosen reference level to be present
    """
    _validate_reference_levels(
        df,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
    )

    out: dict[str, list[str]] = {}
    for c in x_categorical:
        levels = sorted(df[c].astype(str).unique().tolist())
        out[c] = levels
    return out


def _expanded_column_names(
    *,
    x_numeric: list[str],
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
    categorical_levels_seen: dict[str, list[str]],
) -> list[str]:
    cols: list[str] = list(x_numeric)

    for c in x_categorical:
        ref = str(categorical_reference_levels[c])
        levels = list(categorical_levels_seen[c])
        for lvl in levels:
            if lvl == ref:
                continue
            cols.append(f"{c}{lvl}")

    return cols


def build_x_treatment_fit(
    wide_df: pd.DataFrame,
    *,
    x_numeric: list[str],
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
) -> tuple[np.ndarray, TreatmentFeatureEncoding]:
    """
    Build the treatment-model design matrix from fitting data.

    Returns
    -------
    X : np.ndarray
        Expanded design matrix of shape (n_subjects, p).
    encoding : TreatmentFeatureEncoding
        Encoding metadata needed for prediction-time reconstruction.
    """
    required = list(x_numeric) + list(x_categorical)
    missing = [c for c in required if c not in wide_df.columns]
    if missing:
        raise ValueError(f"Missing required treatment covariate columns: {missing}")

    categorical_levels_seen = _categorical_levels_seen_from_fit(
        wide_df,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
    )

    x_col_names = _expanded_column_names(
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        categorical_levels_seen=categorical_levels_seen,
    )

    X = pd.DataFrame(index=wide_df.index)

    for c in x_numeric:
        X[c] = pd.to_numeric(wide_df[c], errors="raise").astype(float)

    for c in x_categorical:
        vals = wide_df[c].astype(str)
        ref = str(categorical_reference_levels[c])
        for lvl in categorical_levels_seen[c]:
            if lvl == ref:
                continue
            X[f"{c}{lvl}"] = (vals == lvl).astype(float)

    X = X.loc[:, x_col_names]

    encoding = TreatmentFeatureEncoding(
        x_numeric=list(x_numeric),
        x_categorical=list(x_categorical),
        categorical_reference_levels=dict(categorical_reference_levels),
        categorical_levels_seen=categorical_levels_seen,
        x_expanded_cols=list(x_col_names),
    )

    return X.to_numpy(dtype=float), encoding


def build_x_treatment_prediction(
    wide_df: pd.DataFrame,
    *,
    x_numeric: list[str],
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
    categorical_levels_seen: dict[str, list[str]],
    x_col_names: list[str],
    hard_fail: bool = True,
) -> tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Build the treatment-model design matrix for prediction data using
    encoding learned during fitting.

    Parameters
    ----------
    hard_fail
        If True, unseen categorical levels raise an error.
        If False, unseen levels are encoded as all-zero dummy columns
        for that categorical variable, and the unseen levels are reported.

    Returns
    -------
    X : np.ndarray
        Expanded design matrix of shape (n_subjects, p).
    unseen : dict[str, list[str]] or None
        Mapping of categorical columns to unseen levels encountered, or None.
    """
    required = list(x_numeric) + list(x_categorical)
    missing = [c for c in required if c not in wide_df.columns]
    if missing:
        raise ValueError(f"Missing required treatment covariate columns: {missing}")

    X = pd.DataFrame(index=wide_df.index)

    for c in x_numeric:
        X[c] = pd.to_numeric(wide_df[c], errors="raise").astype(float)

    unseen: dict[str, list[str]] = {}

    for c in x_categorical:
        vals = wide_df[c].astype(str)
        seen = list(categorical_levels_seen[c])
        ref = str(categorical_reference_levels[c])

        unseen_levels = sorted(set(vals.unique()) - set(seen))
        if unseen_levels:
            unseen[c] = unseen_levels
            if hard_fail:
                raise ValueError(
                    f"Unseen categorical levels for '{c}' at prediction time: "
                    f"{unseen_levels}"
                )

        for lvl in seen:
            if lvl == ref:
                continue
            X[f"{c}{lvl}"] = (vals == lvl).astype(float)

    # Ensure all expected columns exist, in the fitted order
    for c in x_col_names:
        if c not in X.columns:
            X[c] = 0.0

    X = X.loc[:, x_col_names]

    return X.to_numpy(dtype=float), (None if len(unseen) == 0 else unseen)