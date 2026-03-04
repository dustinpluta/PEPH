import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from peph.model.predict import predict_risk
from peph.model.result import FeatureEncoding, FittedPEPHModel


def _encoding_to_jsonable(enc: FeatureEncoding) -> dict:
    # pydantic v2
    if hasattr(enc, "model_dump"):
        return enc.model_dump()
    # pydantic v1
    if hasattr(enc, "dict"):
        return enc.dict()
    # dataclass
    if is_dataclass(enc):
        return asdict(enc)
    # fallback
    return dict(enc.__dict__)


def _tiny_model_with_spatial() -> FittedPEPHModel:
    enc = FeatureEncoding(
        x_numeric=["x"],
        x_categorical=[],
        categorical_reference_levels={},
        categorical_levels_seen={},
        x_expanded_cols=["x"],
    )
    m = FittedPEPHModel(
        breaks=[0.0, 10.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]"],
        x_col_names=["x"],
        param_names=["log_nu[0]", "x"],
        params=[float(np.log(0.1)), 1.0],
        cov=[[1.0, 0.0], [0.0, 1.0]],
        nu=[0.1],
        fit_backend="map_leroux",
        n_train_subjects=2,
        n_train_long_rows=2,
        converged=True,
        aic=None,
        deviance=None,
        llf=None,
    )
    m.__dict__["spatial"] = {
        "type": "leroux",
        "area_col": "zip",
        "zips": ["A", "B"],
        "u": [0.5, -0.5],
        "tau": 1.0,
        "rho": 0.5,
        "optimizer": {"success": True},
        "graph": {"G": 2, "n_components": 1},
    }
    return m


def test_model_artifact_json_roundtrip_preserves_spatial(tmp_path: Path) -> None:
    m = _tiny_model_with_spatial()

    df = pd.DataFrame({"x": [0.0, 0.0], "zip": ["A", "B"]})
    r0 = predict_risk(df, m, times=[10.0], frailty_mode="conditional").ravel()

    # ---- serialize: build JSON-safe payload ----
    payload = m.to_dict() if hasattr(m, "to_dict") else dict(m.__dict__)

    # Ensure encoding is JSON-able
    if isinstance(payload.get("encoding"), FeatureEncoding):
        payload["encoding"] = _encoding_to_jsonable(payload["encoding"])
    elif "encoding" in payload and not isinstance(payload["encoding"], dict):
        # handle case where encoding is stored elsewhere
        payload["encoding"] = _encoding_to_jsonable(m.encoding)

    # Ensure spatial is included
    if "spatial" not in payload and "spatial" in m.__dict__:
        payload["spatial"] = m.__dict__["spatial"]

    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload, indent=2))

    loaded = json.loads(path.read_text())

    # ---- deserialize ----
    if hasattr(FittedPEPHModel, "from_dict"):
        m2 = FittedPEPHModel.from_dict(loaded)
    else:
        m2 = FittedPEPHModel(
            breaks=loaded["breaks"],
            interval_convention=loaded["interval_convention"],
            encoding=FeatureEncoding(**loaded["encoding"]),
            baseline_col_names=loaded["baseline_col_names"],
            x_col_names=loaded["x_col_names"],
            param_names=loaded["param_names"],
            params=loaded["params"],
            cov=loaded["cov"],
            nu=loaded["nu"],
            fit_backend=loaded["fit_backend"],
            n_train_subjects=loaded["n_train_subjects"],
            n_train_long_rows=loaded["n_train_long_rows"],
            converged=loaded.get("converged"),
            aic=loaded.get("aic"),
            deviance=loaded.get("deviance"),
            llf=loaded.get("llf"),
        )
        if "spatial" in loaded:
            m2.__dict__["spatial"] = loaded["spatial"]

    r1 = predict_risk(df, m2, times=[10.0], frailty_mode="conditional").ravel()
    assert np.allclose(r0, r1)
    assert "spatial" in m2.__dict__