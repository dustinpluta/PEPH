from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train_ids: np.ndarray
    test_ids: np.ndarray


def train_test_split_subject(
    df: pd.DataFrame,
    id_col: str,
    test_size: float,
    seed: int,
) -> SplitResult:
    ids = df[id_col].dropna().unique()
    ids = np.asarray(ids)
    if ids.size == 0:
        raise ValueError("No subject ids found for split")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids.size)
    n_test = int(np.floor(test_size * ids.size))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return SplitResult(train_ids=ids[train_idx], test_ids=ids[test_idx])


def apply_split(
    df: pd.DataFrame,
    id_col: str,
    split: SplitResult,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[id_col].isin(split.train_ids)].copy()
    test = df[df[id_col].isin(split.test_ids)].copy()

    # safety: no leakage
    overlap = set(train[id_col].unique()).intersection(set(test[id_col].unique()))
    if overlap:
        raise RuntimeError(f"Split leakage detected for ids: {sorted(list(overlap))[:10]}")
    return train, test