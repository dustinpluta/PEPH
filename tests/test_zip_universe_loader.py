from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from peph.pipeline.run import _load_zip_universe


def test_zip_universe_loader_accepts_zip_column(tmp_path: Path) -> None:
    p = tmp_path / "zips_with_zip_col.csv"
    pd.DataFrame({"zip": ["10001", "10002", "10003"]}).to_csv(p, index=False)

    out = _load_zip_universe(str(p))
    assert out == ["10001", "10002", "10003"]


def test_zip_universe_loader_accepts_single_column_no_zip_name(tmp_path: Path) -> None:
    p = tmp_path / "zips_single_col.csv"
    # Single column but not named "zip"
    pd.DataFrame({"Z": ["20001", "20002"]}).to_csv(p, index=False)

    out = _load_zip_universe(str(p))
    assert out == ["20001", "20002"]


def test_zip_universe_loader_rejects_multi_column_without_zip(tmp_path: Path) -> None:
    p = tmp_path / "bad_multi_col.csv"
    pd.DataFrame({"a": ["1"], "b": ["2"]}).to_csv(p, index=False)

    with pytest.raises(ValueError, match=r"ZIP universe file must contain a 'zip' column or be single-column"):
        _load_zip_universe(str(p))