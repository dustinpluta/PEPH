from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

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
from peph.spatial.graph import build_graph_from_edge_list
from peph.utils.json import write_json


def _write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# =========================
# PR9 helpers (self-contained)
# =========================

def _extract_spatial(fitted: Any) -> Optional[Dict[str, Any]]:
    """
    Leroux fit contract (in-memory):

      fitted.spatial is a dict with keys:
        - "type": "leroux"
        - "area_col": str
        - "zips": list[str]
        - "u": np.ndarray shape (G,)
        - "tau": float
        - "rho": float
        - "optimizer": dict
        - "graph": (may be dict-like; DO NOT rely on it here)

    Returns None if not present (PH model).
    """
    sp = getattr(fitted, "spatial", None)
    if sp is None or not isinstance(sp, dict):
        return None
    required = {"u", "rho", "tau", "zips", "area_col"}
    if not required.issubset(set(sp.keys())):
        return None
    return sp


def _n_train_by_area(train_wide: pd.DataFrame, area_col: str) -> Dict[str, int]:
    vc = train_wide[area_col].astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def _frailty_table_and_summary(
    *,
    zips: list[str],
    components: np.ndarray,
    u_hat: np.ndarray,
    n_train_by_zip: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    u = np.asarray(u_hat, dtype=float).ravel()
    if u.size != len(zips):
        raise ValueError("u_hat length must match number of zips")

    comp = np.asarray(components, dtype=int).ravel()
    if comp.size != len(zips):
        raise ValueError("components length must match number of zips")

    df = pd.DataFrame(
        {
            "zip": [str(z) for z in zips],
            "u_hat": u,
            "component": comp,
            "n_train": [int(n_train_by_zip.get(str(z), 0)) for z in zips],
        }
    ).sort_values("u_hat").reset_index(drop=True)

    df["rank"] = np.arange(len(df), dtype=int)

    comp_means: Dict[str, float] = {}
    for c in np.unique(comp):
        sub = df.loc[df["component"] == c]
        w = np.maximum(sub["n_train"].to_numpy(dtype=float), 1.0)
        comp_means[str(int(c))] = float(np.sum(w * sub["u_hat"].to_numpy(dtype=float)) / np.sum(w))

    summary = {
        "n_areas": int(len(df)),
        "n_train_total": int(df["n_train"].sum()),
        "n_train_nonzero_areas": int((df["n_train"] > 0).sum()),
        "u_mean": float(df["u_hat"].mean()),
        "u_sd": float(df["u_hat"].std(ddof=1)) if len(df) > 1 else 0.0,
        "u_min": float(df["u_hat"].min()),
        "u_max": float(df["u_hat"].max()),
        "u_quantiles": {
            q: float(df["u_hat"].quantile(q))
            for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        },
        "component_weighted_means": comp_means,
    }
    return df, summary


def _morans_I(x: np.ndarray, W_csr: Any) -> Dict[str, float]:
    """
    Moran's I diagnostic with a standard normal approximation (variance under randomization).

    W_csr: symmetric sparse adjacency with zero diagonal.
    """
    import scipy.sparse as sp  # local import

    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        return {"I": float("nan"), "expected": float("nan"), "variance": float("nan"), "z": float("nan")}

    xc = x - float(x.mean())
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return {"I": 0.0, "expected": -1.0 / (n - 1), "variance": float("nan"), "z": float("nan")}

    W = W_csr.tocsr()
    S0 = float(W.sum())
    if S0 <= 0:
        return {"I": 0.0, "expected": -1.0 / (n - 1), "variance": float("nan"), "z": float("nan")}

    num = float(xc @ (W @ xc))
    I = (n / S0) * (num / denom)
    EI = -1.0 / (n - 1)

    if n <= 3:
        return {"I": float(I), "expected": float(EI), "variance": float("nan"), "z": float("nan")}

    # Cliff-Ord style variance under randomization (best-effort)
    W_plus = W + W.T
    S1 = 0.5 * float((W_plus.multiply(W_plus)).sum())
    rs = np.asarray(W.sum(axis=1)).ravel()
    cs = np.asarray(W.sum(axis=0)).ravel()
    S2 = float(np.sum((rs + cs) ** 2))

    x2 = xc**2
    x4 = xc**4
    s2 = float(np.sum(x2))
    b2 = float(n * np.sum(x4) / (s2**2)) if s2 > 0 else 0.0

    num_var = (
        n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * (S0**2))
        - b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * (S0**2))
    )
    den_var = (n - 1) * (n - 2) * (n - 3) * (S0**2)
    varI = float(num_var / den_var) - EI**2
    z = float((I - EI) / np.sqrt(varI)) if varI > 0 else float("nan")

    return {"I": float(I), "expected": float(EI), "variance": float(varI), "z": float(z)}


def _plot_frailty_caterpillar(frailty_df: pd.DataFrame, out_path: Path, top_k: int = 30) -> None:
    import matplotlib.pyplot as plt  # local import

    df = frailty_df[["zip", "u_hat"]].copy().sort_values("u_hat")
    low = df.head(top_k)
    high = df.tail(top_k)
    sub = pd.concat([low, high], axis=0)

    labels = sub["zip"].astype(str).to_list()
    y = sub["u_hat"].to_numpy(dtype=float)
    idx = np.arange(len(y), dtype=int)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hlines(idx, 0.0, y, linewidth=1)
    ax.plot(y, idx, "o")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(idx)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Frailty estimate (û)")
    ax.set_title(f"Frailty extremes (bottom/top {top_k})")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_morans_scatter(x_centered: np.ndarray, spatial_lag: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt  # local import

    x_centered = np.asarray(x_centered, dtype=float).ravel()
    spatial_lag = np.asarray(spatial_lag, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_centered, spatial_lag, "o", markersize=3)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("û (centered)")
    ax.set_ylabel("Row-normalized spatial lag W û")
    ax.set_title(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_calibration_by_bin(cal_df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt  # local import

    df = cal_df.sort_values("bin").copy()
    x = df["mean_pred"].to_numpy(dtype=float)
    y = df["obs_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "o-")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _known_status_mask(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    return (time > horizon) | ((event == 1) & (time <= horizon))


def _load_zip_universe(zips_path: str) -> List[str]:
    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        return zdf["zip"].astype(str).tolist()
    # allow single-column CSV
    if zdf.shape[1] == 1:
        return zdf.iloc[:, 0].astype(str).tolist()
    raise ValueError(f"ZIP universe file must contain a 'zip' column or be single-column: {zips_path}")


def run_pipeline(cfg: RunConfig) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(cfg.output.root_dir) / f"{cfg.run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(out_dir / "config_resolved.yml", cfg.model_dump())

    # Read wide data
    wide = read_table(cfg.data.path, cfg.data.format)

    # ---------------------------------
    # Resolve TTT / TD-covariate config
    # ---------------------------------
    x_td_numeric = list(getattr(cfg.data_schema, "x_td_numeric", []) or [])

    ttt_enabled = bool(getattr(cfg, "ttt", None) is not None and getattr(cfg.ttt, "enabled", False))
    cut_times_col: Optional[str] = None
    td_treatment_col: Optional[str] = None
    treated_td_col: Optional[str] = None

    if ttt_enabled:
        cut_times_col = str(cfg.ttt.treatment_time_col)
        td_treatment_col = str(cfg.ttt.treatment_time_col)
        treated_td_col = str(cfg.ttt.treated_td_col)

        if treated_td_col not in x_td_numeric:
            raise ValueError(
                "TTT is enabled but data_schema.x_td_numeric does not include "
                f"treated_td_col='{treated_td_col}'."
            )

    # Validate columns exist in WIDE data
    required = (
        [cfg.data_schema.id_col, cfg.data_schema.time_col, cfg.data_schema.event_col]
        + cfg.data_schema.x_numeric
        + cfg.data_schema.x_categorical
    )

    if cut_times_col is not None:
        required.append(cut_times_col)

    # If spatial is enabled, area_col must exist in WIDE and must be carried to LONG.
    if cfg.spatial is not None:
        required.append(cfg.spatial.area_col)

    missing = [c for c in required if c not in wide.columns]
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    # Split
    split = train_test_split_subject(
        wide,
        id_col=cfg.data_schema.id_col,
        test_size=cfg.split.test_size,
        seed=cfg.split.seed,
    )
    train_wide, test_wide = apply_split(wide, id_col=cfg.data_schema.id_col, split=split)

    write_json(
        out_dir / "split_ids.json",
        {"train_ids": split.train_ids.tolist(), "test_ids": split.test_ids.tolist()},
    )

    # Write wide splits (helpful for debugging)
    write_table(train_wide, out_dir / "train_wide.parquet")
    write_table(test_wide, out_dir / "test_wide.parquet")

    # Expand to long
    x_cols_all = cfg.data_schema.x_numeric + cfg.data_schema.x_categorical
    if cfg.spatial is not None:
        # critical for Leroux: area_col must be present in long rows
        x_cols_all = x_cols_all + [cfg.spatial.area_col]

    long_train = expand_long(
        train_wide,
        id_col=cfg.data_schema.id_col,
        time_col=cfg.data_schema.time_col,
        event_col=cfg.data_schema.event_col,
        x_cols=x_cols_all,
        breaks=cfg.time.breaks,
        cut_times_col=cut_times_col,
        td_treatment_col=td_treatment_col,
        treated_td_col=(treated_td_col if treated_td_col is not None else "treated_td"),
    )
    long_test = expand_long(
        test_wide,
        id_col=cfg.data_schema.id_col,
        time_col=cfg.data_schema.time_col,
        event_col=cfg.data_schema.event_col,
        x_cols=x_cols_all,
        breaks=cfg.time.breaks,
        cut_times_col=cut_times_col,
        td_treatment_col=td_treatment_col,
        treated_td_col=(treated_td_col if treated_td_col is not None else "treated_td"),
    )

    write_table(long_train, out_dir / "long_train.parquet")
    write_table(long_test, out_dir / "long_test.parquet")

    # Fit
    fitted = fit_model_dispatch(
        backend=cfg.fit.backend,
        long_train=long_train,
        train_wide=train_wide,
        breaks=cfg.time.breaks,
        x_numeric=cfg.data_schema.x_numeric,
        x_td_numeric=x_td_numeric,
        x_categorical=cfg.data_schema.x_categorical,
        categorical_reference_levels=cfg.data_schema.categorical_reference_levels,
        n_train_subjects=int(train_wide[cfg.data_schema.id_col].nunique()),
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
        train_wide_time=train_wide[cfg.data_schema.time_col].to_numpy(dtype=float),
        train_wide_event=train_wide[cfg.data_schema.event_col].to_numpy(dtype=int),
    )
    write_json(out_dir / "inference.json", inf)

    # ---------------------------------
    # Prediction: not yet supported for TD covariates
    # ---------------------------------
    if x_td_numeric:
        raise NotImplementedError(
            "Pipeline fitting with long-form time-dependent covariates is enabled, "
            f"but prediction is not yet implemented for x_td_numeric={x_td_numeric}. "
            "Current predict.py only supports baseline wide-data covariates."
        )

    # Predict on test
    horizons = list(map(float, cfg.predict.horizons_days or []))

    # Frailty-aware predictions:
    # - "auto": Leroux -> conditional, PH -> none
    frailty_mode = getattr(cfg.predict, "frailty_mode", "auto")

    eta = predict_linear_predictor(
        test_wide,
        fitted,
        frailty_mode=frailty_mode,
        hard_fail=True,
        allow_unseen_area=(cfg.spatial.allow_unseen_area if cfg.spatial else False),
    )

    pred_df = pd.DataFrame(
        {
            cfg.data_schema.id_col: test_wide[cfg.data_schema.id_col].to_numpy(),
            cfg.data_schema.time_col: test_wide[cfg.data_schema.time_col].to_numpy(dtype=float),
            cfg.data_schema.event_col: test_wide[cfg.data_schema.event_col].to_numpy(dtype=int),
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
    time = pred_df[cfg.data_schema.time_col].to_numpy(dtype=float)
    event = pred_df[cfg.data_schema.event_col].to_numpy(dtype=int)
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
    resid_df: Optional[pd.DataFrame] = None
    if cfg.metrics.residuals.get("cox_snell", True):
        r = cox_snell_residuals(fitted, test_wide, time_col=cfg.data_schema.time_col)
        metrics["cox_snell_mean"] = float(np.mean(r))
        metrics["cox_snell_var"] = float(np.var(r))
        resid_df = pd.DataFrame(
            {
                cfg.data_schema.id_col: test_wide[cfg.data_schema.id_col].values,
                "cox_snell": r,
                cfg.data_schema.event_col: test_wide[cfg.data_schema.event_col].values,
            }
        )
        write_table(resid_df, out_dir / "cox_snell_residuals.parquet")

    # =========================
    # PR9: Spatial frailty diagnostics + grouped calibration
    # =========================
    sp = _extract_spatial(fitted)
    if sp is not None:
        if cfg.spatial is None:
            raise RuntimeError("Fitted model includes spatial frailty but cfg.spatial is None")

        # Build graph from config files (authoritative)
        zips_universe = _load_zip_universe(cfg.spatial.zips_path)
        edges_df = pd.read_csv(cfg.spatial.edges_path)
        graph = build_graph_from_edge_list(
            zips_universe,
            edges_df,
            col_u=cfg.spatial.edges_u_col,
            col_v=cfg.spatial.edges_v_col,
        )

        # Align u to graph ordering using fitted.spatial["zips"]
        u_hat = np.asarray(sp["u"], dtype=float).ravel()
        z_u = [str(z) for z in sp["zips"]]
        if len(z_u) != u_hat.size:
            raise ValueError(f"Mismatch: len(fitted.spatial['zips'])={len(z_u)} vs len(u)={u_hat.size}")

        u_map = {z: u_hat[i] for i, z in enumerate(z_u)}
        missing_u = [z for z in graph.zips if z not in u_map]
        if missing_u:
            raise ValueError(
                f"Graph contains ZIPs not present in fitted spatial.zips (n_missing={len(missing_u)}). "
                f"Example: {missing_u[:5]}"
            )
        u_graph = np.array([u_map[z] for z in graph.zips], dtype=float)

        rho_hat = float(sp["rho"])
        tau_hat = float(sp["tau"])
        area_col = str(sp["area_col"])

        # Frailty outputs
        n_train_map = _n_train_by_area(train_wide, area_col=area_col)
        components = np.asarray(graph.component_ids(), dtype=int)
        zips_graph = [str(z) for z in graph.zips]

        frailty_df, frailty_summary = _frailty_table_and_summary(
            zips=zips_graph,
            components=components,
            u_hat=u_graph,
            n_train_by_zip=n_train_map,
        )
        frailty_summary["rho_hat"] = rho_hat
        frailty_summary["tau_hat"] = tau_hat

        write_table(frailty_df, out_dir / "frailty_table.parquet")
        write_json(out_dir / "frailty_summary.json", frailty_summary)

        # Moran's I on u (graph-aligned)
        W = graph.W()
        mi_u = _morans_I(u_graph, W)
        write_json(out_dir / "spatial_autocorr.json", {"morans_I_u": mi_u})

        metrics["morans_I_u"] = float(mi_u.get("I", np.nan))
        metrics["morans_I_u_z"] = float(mi_u.get("z", np.nan))
        metrics["leroux_rho_hat"] = rho_hat
        metrics["leroux_tau_hat"] = tau_hat

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Moran scatterplot: row-normalized spatial lag of centered u
        rs = np.asarray(W.sum(axis=1)).ravel().astype(float)
        rs[rs == 0.0] = 1.0
        uc = u_graph - float(np.mean(u_graph))
        lag = (W @ uc) / rs
        _plot_morans_scatter(uc, lag, plots_dir / "morans_scatter_u.png", title="Moran scatterplot of û")

        _plot_frailty_caterpillar(frailty_df, plots_dir / "frailty_caterpillar.png", top_k=30)

        # Calibration by frailty decile (known-status subset at each horizon)
        if horizons:
            tables_dir = out_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)

            ft = frailty_df[["zip", "u_hat"]].drop_duplicates("zip").copy()
            ft["bin"] = pd.qcut(ft["u_hat"], q=10, labels=False, duplicates="drop")

            test_aug = test_wide.copy()
            test_aug[area_col] = test_aug[area_col].astype(str)
            test_aug = test_aug.merge(
                ft[["zip", "bin"]],
                how="left",
                left_on=area_col,
                right_on="zip",
                validate="many_to_one",
            )

            if test_aug["bin"].isna().any():
                raise ValueError(
                    "Some test rows could not be assigned a frailty bin (unseen area?). "
                    "Set spatial.allow_unseen_area=true to allow unseen ZIPs."
                )

            risk_cols = [f"risk_t{int(t)}" for t in horizons]
            join_cols = [cfg.data_schema.id_col] + [c for c in risk_cols if c in pred_df.columns]
            test_aug = test_aug.merge(
                pred_df[join_cols],
                on=cfg.data_schema.id_col,
                how="left",
                validate="one_to_one",
            )

            t_arr = test_aug[cfg.data_schema.time_col].to_numpy(dtype=float)
            e_arr = test_aug[cfg.data_schema.event_col].to_numpy(dtype=int)

            for tau in horizons:
                it = int(tau)
                rc = f"risk_t{it}"
                if rc not in test_aug.columns:
                    continue

                known = _known_status_mask(t_arr, e_arr, float(tau))
                sub = test_aug.loc[known, ["bin", rc, cfg.data_schema.time_col, cfg.data_schema.event_col]].copy()

                y = (
                    (sub[cfg.data_schema.event_col].to_numpy(dtype=int) == 1)
                    & (sub[cfg.data_schema.time_col].to_numpy(dtype=float) <= float(tau))
                ).astype(float)
                p = sub[rc].to_numpy(dtype=float)

                cal = (
                    pd.DataFrame({"bin": sub["bin"].to_numpy(dtype=int), "y": y, "p": p})
                    .groupby("bin", as_index=False)
                    .agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
                )
                cal["cal_in_the_large"] = cal["obs_rate"] - cal["mean_pred"]

                write_table(cal, tables_dir / f"calibration_by_frailty_decile_t{it}.parquet")
                _plot_calibration_by_bin(
                    cal,
                    plots_dir / f"calibration_by_frailty_decile_t{it}.png",
                    title=f"Calibration by frailty decile (t={it} days)",
                )

    # Write metrics after spatial additions
    write_json(out_dir / "metrics.json", metrics)

    # Plots (existing)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.plots.cox_snell and cfg.metrics.residuals.get("cox_snell", True):
        if resid_df is None:
            resid_df = pd.read_parquet(out_dir / "cox_snell_residuals.parquet")
        plot_cox_snell(
            resid_df["cox_snell"].to_numpy(dtype=float),
            resid_df[cfg.data_schema.event_col].to_numpy(dtype=int),
            plots_dir / "cox_snell.png",
        )

    if cfg.plots.calibration_risk and horizons:
        for tau in horizons:
            rc = f"risk_t{int(tau)}"
            plot_calibration_risk_by_quantile(
                pred_df,
                time_col=cfg.data_schema.time_col,
                event_col=cfg.data_schema.event_col,
                risk_col=rc,
                tau=float(tau),
                n_bins=10,
                out_path=plots_dir / f"calibration_risk_t{int(tau)}.png",
            )

    return out_dir