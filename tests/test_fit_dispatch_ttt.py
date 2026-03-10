from __future__ import annotations

import pandas as pd
import pytest

import peph.model.fit_dispatch as fd


def test_fit_model_dispatch_forwards_x_td_numeric_to_ph(monkeypatch) -> None:
    captured = {}

    def _fake_fit_peph(**kwargs):
        captured.update(kwargs)
        return {"backend": "ph"}

    monkeypatch.setattr(fd, "fit_peph", _fake_fit_peph)

    long_train = pd.DataFrame(
        {
            "id": [1, 1],
            "k": [0, 1],
            "event": [0, 1],
            "exposure": [30.0, 20.0],
            "x": [0.1, 0.1],
            "treated_td": [0, 1],
        }
    )
    train_wide = pd.DataFrame({"id": [1]})

    out = fd.fit_model_dispatch(
        backend="statsmodels_glm_poisson",
        long_train=long_train,
        train_wide=train_wide,
        breaks=[0, 30, 90],
        x_numeric=["x"],
        x_td_numeric=["treated_td"],
        x_categorical=[],
        categorical_reference_levels={},
        n_train_subjects=1,
        covariance="classical",
    )

    assert out == {"backend": "ph"}
    assert captured["x_numeric"] == ["x"]
    assert captured["x_td_numeric"] == ["treated_td"]
    assert captured["x_categorical"] == []
    assert captured["long_train"].equals(long_train)


def test_fit_model_dispatch_forwards_x_td_numeric_to_leroux(monkeypatch) -> None:
    captured = {}

    def _fake_fit_pe_leroux_map(**kwargs):
        captured.update(kwargs)
        return {"backend": "leroux"}

    monkeypatch.setattr(fd, "fit_pe_leroux_map", _fake_fit_pe_leroux_map)

    long_train = pd.DataFrame(
        {
            "id": [1, 1],
            "k": [0, 1],
            "event": [0, 1],
            "exposure": [30.0, 20.0],
            "x": [0.1, 0.1],
            "treated_td": [0, 1],
            "zip": ["30303", "30303"],
        }
    )
    train_wide = pd.DataFrame({"id": [1], "zip": ["30303"]})

    out = fd.fit_model_dispatch(
        backend="map_leroux",
        long_train=long_train,
        train_wide=train_wide,
        breaks=[0, 30, 90],
        x_numeric=["x"],
        x_td_numeric=["treated_td"],
        x_categorical=[],
        categorical_reference_levels={},
        n_train_subjects=1,
        covariance="classical",
        spatial_area_col="zip",
        spatial_zips_path="data/zips.csv",
        spatial_edges_path="data/zip_adjacency.csv",
        spatial_edges_u_col="zip_u",
        spatial_edges_v_col="zip_v",
        allow_unseen_area=False,
        leroux_max_iter=25,
        leroux_ftol=1e-6,
        rho_clip=1e-6,
        q_jitter=1e-8,
        prior_logtau_sd=10.0,
        prior_rho_a=1.0,
        prior_rho_b=1.0,
    )

    assert out == {"backend": "leroux"}
    assert captured["x_numeric"] == ["x"]
    assert captured["x_td_numeric"] == ["treated_td"]
    assert captured["x_categorical"] == []
    assert captured["area_col"] == "zip"


def test_fit_model_dispatch_defaults_x_td_numeric_to_empty(monkeypatch) -> None:
    captured = {}

    def _fake_fit_peph(**kwargs):
        captured.update(kwargs)
        return {"backend": "ph"}

    monkeypatch.setattr(fd, "fit_peph", _fake_fit_peph)

    long_train = pd.DataFrame(
        {
            "id": [1],
            "k": [0],
            "event": [0],
            "exposure": [10.0],
            "x": [0.1],
        }
    )
    train_wide = pd.DataFrame({"id": [1]})

    fd.fit_model_dispatch(
        backend="statsmodels_glm_poisson",
        long_train=long_train,
        train_wide=train_wide,
        breaks=[0, 30],
        x_numeric=["x"],
        x_categorical=[],
        categorical_reference_levels={},
        n_train_subjects=1,
    )

    assert captured["x_td_numeric"] == []


def test_fit_model_dispatch_leroux_requires_spatial_inputs() -> None:
    long_train = pd.DataFrame(
        {
            "id": [1],
            "k": [0],
            "event": [0],
            "exposure": [10.0],
            "x": [0.1],
            "zip": ["30303"],
        }
    )
    train_wide = pd.DataFrame({"id": [1], "zip": ["30303"]})

    with pytest.raises(ValueError, match="requires spatial_area_col"):
        fd.fit_model_dispatch(
            backend="map_leroux",
            long_train=long_train,
            train_wide=train_wide,
            breaks=[0, 30],
            x_numeric=["x"],
            x_categorical=[],
            categorical_reference_levels={},
            n_train_subjects=1,
        )