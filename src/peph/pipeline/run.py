from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

from peph.config.schema import RunConfig
from peph.data.io import read_table, write_table
from peph.data.long import expand_long
from peph.data.split import apply_split, train_test_split_subject
from peph.metrics.calibration import brier_ipcw, calibration_logistic_ipcw
from peph.metrics.discrimination import c_index_harrell, time_dependent_auc_ipcw
from peph.metrics.residuals import cox_snell_residuals
from peph.model.fit import fit_peph
from peph.model.predict import make_test_prediction_frame
from peph.plots.calibration import plot_calibration_risk_by_quantile
from peph.plots.diagnostics import plot_cox_snell
from peph.utils.json import write_json


def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_pipeline(cfg: RunConfig) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(cfg.output.root_dir) / f"{cfg.run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(out_dir / "config_resolved.yml", cfg.model_dump())

    # Read wide data
    wide = read_table(cfg.data.path, cfg.data.format)

    # Validate columns exist
    required = [cfg.schema.id_col, cfg.schema.time_col, cfg.schema.event_col] + cfg.schema.x_numeric + cfg.schema.x_categorical
    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    # Split
    split = train_test_split_subject(
        wide,
        id_col=cfg.schema.id_col,
        test_size=cfg.split.test_size,
        seed=cfg.split.seed,
    )
    train_wide, test_wide = apply_split(wide, id_col=cfg.schema.id_col, split=split)

    write_json(out_dir / "split_ids.json", {"train_ids": split.train_ids.tolist(), "test_ids": split.test_ids.tolist()})

    # Write wide splits (helpful for debugging)
    write_table(train_wide, out_dir / "train_wide.parquet")
    write_table(test_wide, out_dir / "test_wide.parquet")

    # Expand to long
    x_cols_all = cfg.schema.x_numeric + cfg.schema.x_categorical
    long_train = expand_long(
        train_wide,
        id_col=cfg.schema.id_col,
        time_col=cfg.schema.time_col,
        event_col=cfg.schema.event_col,
        x_cols=x_cols_all,
        breaks=cfg.time.breaks,
    )
    long_test = expand_long(
        test_wide,
        id_col=cfg.schema.id_col,
        time_col=cfg.schema.time_col,
        event_col=cfg.schema.event_col,
        x_cols=x_cols_all,
        breaks=cfg.time.breaks,
    )

    write_table(long_train, out_dir / "long_train.parquet")
    write_table(long_test, out_dir / "long_test.parquet")

    # Fit
    fitted = fit_peph(
        long_train,
        breaks=cfg.time.breaks,
        x_numeric=cfg.schema.x_numeric,
        x_categorical=cfg.schema.x_categorical,
        categorical_reference_levels=cfg.schema.categorical_reference_levels,
        max_iter=cfg.fit.max_iter,
        tol=cfg.fit.tol,
        n_train_subjects=int(train_wide[cfg.schema.id_col].nunique()),
    )
    fitted.save(str(out_dir / "model.json"))

    # Predict on test
    horizons = list(map(float, cfg.predict.horizons_days or []))
    pred_df = make_test_prediction_frame(
        fitted,
        test_wide,
        id_col=cfg.schema.id_col,
        time_col=cfg.schema.time_col,
        event_col=cfg.schema.event_col,
        horizons=horizons,
    )

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_table(pred_df, pred_dir / "test_predictions.parquet")

    # Metrics (test)
    metrics: Dict[str, Any] = {}
    time = pred_df[cfg.schema.time_col].to_numpy(dtype=float)
    event = pred_df[cfg.schema.event_col].to_numpy(dtype=int)
    score = pred_df["eta"].to_numpy(dtype=float)

    if cfg.metrics.discrimination.get("c_index", True):
        metrics["c_index"] = c_index_harrell(time, event, score)

    if cfg.metrics.discrimination.get("time_dependent_auc", True) and horizons:
        metrics.update(time_dependent_auc_ipcw(time, event, score, horizons))

    # Calibration + Brier by horizon (requires predicted risk columns)
    if horizons:
        for tau in horizons:
            rc = f"risk_t{int(tau)}"
            if rc not in pred_df.columns:
                raise RuntimeError(f"Missing required prediction column '{rc}' for calibration metrics")
        # stack risks for functions
        # compute per-horizon using each rc
        if cfg.metrics.calibration.get("brier_score", True):
            for tau in horizons:
                bs = brier_ipcw(time, event, pred_df[f"risk_t{int(tau)}"].to_numpy(dtype=float), [tau])
                metrics.update(bs)

        if cfg.metrics.calibration.get("calibration_in_the_large", True) or cfg.metrics.calibration.get("calibration_slope", True):
            for tau in horizons:
                cal = calibration_logistic_ipcw(
                    time, event, pred_df[f"risk_t{int(tau)}"].to_numpy(dtype=float), [tau]
                )
                metrics.update(cal)

    # Cox–Snell residuals (test)
    if cfg.metrics.residuals.get("cox_snell", True):
        r = cox_snell_residuals(fitted, test_wide, time_col=cfg.schema.time_col)
        # store summary only; full residuals can go to file if desired
        metrics["cox_snell_mean"] = float(np.mean(r))
        metrics["cox_snell_var"] = float(np.var(r))
        # write residuals
        resid_df = pd.DataFrame(
            {
                cfg.schema.id_col: test_wide[cfg.schema.id_col].values,
                "cox_snell": r,
                cfg.schema.event_col: test_wide[cfg.schema.event_col].values,
            }
        )
        write_table(resid_df, out_dir / "cox_snell_residuals.parquet")

    write_json(out_dir / "metrics.json", metrics)

    # Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.plots.cox_snell and cfg.metrics.residuals.get("cox_snell", True):
        resid_df = pd.read_parquet(out_dir / "cox_snell_residuals.parquet")
        plot_cox_snell(
            resid_df["cox_snell"].to_numpy(dtype=float),
            resid_df[cfg.schema.event_col].to_numpy(dtype=int),
            plots_dir / "cox_snell.png",
        )

    if cfg.plots.calibration_risk and horizons:
        for tau in horizons:
            rc = f"risk_t{int(tau)}"
            plot_calibration_risk_by_quantile(
                pred_df,
                time_col=cfg.schema.time_col,
                event_col=cfg.schema.event_col,
                risk_col=rc,
                tau=float(tau),
                n_bins=10,
                out_path=plots_dir / f"calibration_risk_t{int(tau)}.png",
            )

    return out_dir