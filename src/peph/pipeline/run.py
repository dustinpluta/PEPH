from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

from peph.model.fit_dispatch import fit_model_dispatch
from peph.model.inference import baseline_table, coef_table, inference_summary
from peph.config.schema import RunConfig
from peph.data.io import read_table, write_table
from peph.data.long import expand_long
from peph.data.split import apply_split, train_test_split_subject
from peph.metrics.calibration import brier_ipcw, calibration_logistic_ipcw
from peph.metrics.discrimination import c_index_harrell, time_dependent_auc_ipcw
from peph.metrics.residuals import cox_snell_residuals
from peph.model.predict import (
    predict_linear_predictor,
    predict_risk,
    predict_survival,
    predict_cumhaz,
)
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
    required = (
        [cfg.schema.id_col, cfg.schema.time_col, cfg.schema.event_col]
        + cfg.schema.x_numeric
        + cfg.schema.x_categorical
    )

    # If spatial is enabled, area_col must exist in WIDE and must be carried to LONG.
    if cfg.spatial is not None:
        required.append(cfg.spatial.area_col)

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

    write_json(
        out_dir / "split_ids.json",
        {"train_ids": split.train_ids.tolist(), "test_ids": split.test_ids.tolist()},
    )

    # Write wide splits (helpful for debugging)
    write_table(train_wide, out_dir / "train_wide.parquet")
    write_table(test_wide, out_dir / "test_wide.parquet")

    # Expand to long
    x_cols_all = cfg.schema.x_numeric + cfg.schema.x_categorical
    if cfg.spatial is not None:
        # critical for Leroux: area_col must be present in long rows
        x_cols_all = x_cols_all + [cfg.spatial.area_col]

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
    fitted = fit_model_dispatch(
        backend=cfg.fit.backend,
        long_train=long_train,
        train_wide=train_wide,
        breaks=cfg.time.breaks,
        x_numeric=cfg.schema.x_numeric,
        x_categorical=cfg.schema.x_categorical,
        categorical_reference_levels=cfg.schema.categorical_reference_levels,
        n_train_subjects=int(train_wide[cfg.schema.id_col].nunique()),
        covariance=cfg.fit.covariance,
        spatial_area_col=(cfg.spatial.area_col if cfg.spatial else None),
        spatial_zips_path=(cfg.spatial.zips_path if cfg.spatial else None),
        spatial_edges_path=(cfg.spatial.edges_path if cfg.spatial else None),
        spatial_edges_u_col=(cfg.spatial.edges_u_col if cfg.spatial else "zip_u"),
        spatial_edges_v_col=(cfg.spatial.edges_v_col if cfg.spatial else "zip_v"),
        allow_unseen_area=(cfg.spatial.allow_unseen_area if cfg.spatial else False),
        leroux_max_iter=cfg.fit.leroux_max_iter,
        leroux_ftol=cfg.fit.leroux_ftol,
        rho_clip=cfg.fit.rho_clip,
        q_jitter=cfg.fit.q_jitter,
        prior_logtau_sd=cfg.fit.prior_logtau_sd,
        prior_rho_a=cfg.fit.prior_rho_a,
        prior_rho_b=cfg.fit.prior_rho_b,
    )
    fitted.save(str(out_dir / "model.json"))

    # Inference artifacts
    coef_df = coef_table(fitted)
    base_df = baseline_table(fitted)

    write_table(coef_df, out_dir / "coef_table.parquet")
    write_table(base_df, out_dir / "baseline_table.parquet")

    inf = inference_summary(
        fitted,
        train_wide_time=train_wide[cfg.schema.time_col].to_numpy(dtype=float),
        train_wide_event=train_wide[cfg.schema.event_col].to_numpy(dtype=int),
    )
    write_json(out_dir / "inference.json", inf)

    # Predict on test
    horizons = list(map(float, cfg.predict.horizons_days or []))

    # Frailty-aware predictions:
    # - "auto": Leroux -> conditional, PH -> none
    # If cfg.predict.frailty_mode isn't defined in your schema yet, default to "auto".
    frailty_mode = getattr(cfg.predict, "frailty_mode", "auto")

    eta = predict_linear_predictor(
        test_wide,
        fitted,
        frailty_mode=frailty_mode,
        # hard_fail categorical handling is enforced in design builder
        hard_fail=True,
        # unseen ZIP behavior is controlled by allow_unseen_area at prediction time
        allow_unseen_area=(cfg.spatial.allow_unseen_area if cfg.spatial else False),
    )

    pred_df = pd.DataFrame(
        {
            cfg.schema.id_col: test_wide[cfg.schema.id_col].to_numpy(),
            cfg.schema.time_col: test_wide[cfg.schema.time_col].to_numpy(dtype=float),
            cfg.schema.event_col: test_wide[cfg.schema.event_col].to_numpy(dtype=int),
            "eta": eta.astype(float),
        }
    )

    if horizons:
        S = predict_survival(
            test_wide,
            fitted,
            times=horizons,
            frailty_mode=frailty_mode,
            hard_fail=True,
            allow_unseen_area=(cfg.spatial.allow_unseen_area if cfg.spatial else False),
        )
        R = predict_risk(
            test_wide,
            fitted,
            times=horizons,
            frailty_mode=frailty_mode,
            hard_fail=True,
            allow_unseen_area=(cfg.spatial.allow_unseen_area if cfg.spatial else False),
        )
        CH = predict_cumhaz(
            test_wide,
            fitted,
            times=horizons,
            frailty_mode=frailty_mode,
            hard_fail=True,
            allow_unseen_area=(cfg.spatial.allow_unseen_area if cfg.spatial else False),
        )

        for j, tau in enumerate(horizons):
            it = int(tau)
            pred_df[f"surv_t{it}"] = S[:, j]
            pred_df[f"risk_t{it}"] = R[:, j]
            pred_df[f"cumhaz_t{it}"] = CH[:, j]

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

        if cfg.metrics.calibration.get("brier_score", True):
            for tau in horizons:
                bs = brier_ipcw(
                    time,
                    event,
                    pred_df[f"risk_t{int(tau)}"].to_numpy(dtype=float),
                    [tau],
                )
                metrics.update(bs)

        if cfg.metrics.calibration.get("calibration_in_the_large", True) or cfg.metrics.calibration.get(
            "calibration_slope", True
        ):
            for tau in horizons:
                cal = calibration_logistic_ipcw(
                    time,
                    event,
                    pred_df[f"risk_t{int(tau)}"].to_numpy(dtype=float),
                    [tau],
                )
                metrics.update(cal)

    # Cox–Snell residuals (test)
    if cfg.metrics.residuals.get("cox_snell", True):
        r = cox_snell_residuals(fitted, test_wide, time_col=cfg.schema.time_col)
        metrics["cox_snell_mean"] = float(np.mean(r))
        metrics["cox_snell_var"] = float(np.var(r))
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