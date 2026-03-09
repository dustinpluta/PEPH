from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peph.model.predict import predict_risk, predict_survival
from peph.model.predict_bootstrap import (
    predict_risk_bootstrap,
    predict_risk_contrast_bootstrap,
    predict_survival_bootstrap,
)
from peph.model.result import FittedPEPHModel


def _parse_float_list(text: str | None) -> list[float]:
    if text is None or text.strip() == "":
        return []
    vals = []
    for part in text.split(","):
        p = part.strip()
        if p:
            vals.append(float(p))
    return vals


def _make_reference_row(
    args: argparse.Namespace,
    model: FittedPEPHModel,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "id": 1,
        "age_per10_centered": float(args.age_per10_centered),
        "cci": int(args.cci),
        "tumor_size_log": float(args.tumor_size_log),
        "ses": float(args.ses),
        "sex": str(args.sex),
        "stage": str(args.stage),
        "treatment_time": np.nan,
    }

    spatial = getattr(model, "spatial", None)
    if spatial is not None:
        area_col = spatial.get("area_col", "zip")
        if args.zip is None:
            raise ValueError(
                f"This model includes spatial frailty with area_col='{area_col}'. "
                "Please provide --zip for the reference patient."
            )
        row[area_col] = str(args.zip)
    elif args.zip is not None:
        row["zip"] = str(args.zip)

    return pd.DataFrame([row])


def _predict_fixed_schedule(
    ref_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    times: list[float],
    fixed_treatment_time: float,
    n_boot: int = 0,
    alpha: float = 0.05,
    bootstrap_seed: int = 123,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if n_boot > 0:
        surv = predict_survival_bootstrap(
            ref_df,
            model,
            times=times,
            treatment_time_col="treatment_time",
            counterfactual_mode="fixed",
            fixed_treatment_time=float(fixed_treatment_time),
            n_boot=n_boot,
            alpha=alpha,
            seed=bootstrap_seed,
            hard_fail=True,
        )
        risk = predict_risk_bootstrap(
            ref_df,
            model,
            times=times,
            treatment_time_col="treatment_time",
            counterfactual_mode="fixed",
            fixed_treatment_time=float(fixed_treatment_time),
            n_boot=n_boot,
            alpha=alpha,
            seed=bootstrap_seed,
            hard_fail=True,
        )
        return surv, risk

    surv_point = predict_survival(
        ref_df,
        model,
        times=times,
        treatment_time_col="treatment_time",
        counterfactual_mode="fixed",
        fixed_treatment_time=float(fixed_treatment_time),
        hard_fail=True,
    )
    risk_point = predict_risk(
        ref_df,
        model,
        times=times,
        treatment_time_col="treatment_time",
        counterfactual_mode="fixed",
        fixed_treatment_time=float(fixed_treatment_time),
        hard_fail=True,
    )

    surv = {
        "point": surv_point,
        "mean": surv_point,
        "sd": np.zeros_like(surv_point),
        "lower": surv_point,
        "upper": surv_point,
    }
    risk = {
        "point": risk_point,
        "mean": risk_point,
        "sd": np.zeros_like(risk_point),
        "lower": risk_point,
        "upper": risk_point,
    }
    return surv, risk


def _predict_never(
    ref_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    times: list[float],
    n_boot: int = 0,
    alpha: float = 0.05,
    bootstrap_seed: int = 123,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if n_boot > 0:
        surv = predict_survival_bootstrap(
            ref_df,
            model,
            times=times,
            treatment_time_col="treatment_time",
            counterfactual_mode="never",
            n_boot=n_boot,
            alpha=alpha,
            seed=bootstrap_seed,
            hard_fail=True,
        )
        risk = predict_risk_bootstrap(
            ref_df,
            model,
            times=times,
            treatment_time_col="treatment_time",
            counterfactual_mode="never",
            n_boot=n_boot,
            alpha=alpha,
            seed=bootstrap_seed,
            hard_fail=True,
        )
        return surv, risk

    surv_point = predict_survival(
        ref_df,
        model,
        times=times,
        treatment_time_col="treatment_time",
        counterfactual_mode="never",
        hard_fail=True,
    )
    risk_point = predict_risk(
        ref_df,
        model,
        times=times,
        treatment_time_col="treatment_time",
        counterfactual_mode="never",
        hard_fail=True,
    )

    surv = {
        "point": surv_point,
        "mean": surv_point,
        "sd": np.zeros_like(surv_point),
        "lower": surv_point,
        "upper": surv_point,
    }
    risk = {
        "point": risk_point,
        "mean": risk_point,
        "sd": np.zeros_like(risk_point),
        "lower": risk_point,
        "upper": risk_point,
    }
    return surv, risk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute direct delay contrasts for a reference patient under fixed treatment schedules."
    )

    parser.add_argument("--model", required=True, help="Path to fitted model.json")
    parser.add_argument("--out-dir", required=True, help="Directory to write outputs")

    parser.add_argument(
        "--times",
        default="365,730,1825",
        help="Comma-separated prediction horizons in days, e.g. '365,730,1825'",
    )
    parser.add_argument(
        "--base-treat-times",
        default="30,60,90",
        help="Comma-separated base treatment times s; contrasts compare s vs s+delay_days",
    )
    parser.add_argument(
        "--delay-days",
        type=float,
        default=30.0,
        help="Delay amount in days; compares treat_at_s vs treat_at_(s+delay_days)",
    )

    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=123)

    parser.add_argument("--age-per10-centered", type=float, default=0.0)
    parser.add_argument("--cci", type=int, default=1)
    parser.add_argument("--tumor-size-log", type=float, default=3.5)
    parser.add_argument("--ses", type=float, default=0.0)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--stage", type=str, default="II")
    parser.add_argument("--zip", type=str, default=None)

    args = parser.parse_args()

    if args.n_boot < 0:
        raise ValueError("--n-boot must be nonnegative")
    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("--alpha must be in (0, 1)")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = FittedPEPHModel.load(str(model_path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = sorted(_parse_float_list(args.times))
    if not times:
        raise ValueError("--times must contain at least one horizon")

    base_treat_times = sorted(_parse_float_list(args.base_treat_times))
    if not base_treat_times:
        raise ValueError("--base-treat-times must contain at least one value")

    delay_days = float(args.delay_days)
    if delay_days < 0:
        raise ValueError("--delay-days must be nonnegative")

    ref_df = _make_reference_row(args, model)

    never_surv_out, never_risk_out = _predict_never(
        ref_df,
        model,
        times=times,
        n_boot=args.n_boot,
        alpha=args.alpha,
        bootstrap_seed=args.bootstrap_seed,
    )
    never_surv = never_surv_out["point"][0, :]
    never_risk = never_risk_out["point"][0, :]
    never_surv_lower = never_surv_out["lower"][0, :]
    never_surv_upper = never_surv_out["upper"][0, :]
    never_risk_lower = never_risk_out["lower"][0, :]
    never_risk_upper = never_risk_out["upper"][0, :]

    rows: list[dict[str, Any]] = []

    for t, s, s_lo, s_hi, r, r_lo, r_hi in zip(
        times,
        never_surv,
        never_surv_lower,
        never_surv_upper,
        never_risk,
        never_risk_lower,
        never_risk_upper,
    ):
        rows.append(
            {
                "contrast": "never_treated",
                "base_treat_time": None,
                "delayed_treat_time": None,
                "delay_days": None,
                "horizon_days": float(t),
                "survival_base": None,
                "survival_base_lower": None,
                "survival_base_upper": None,
                "survival_delayed": None,
                "survival_delayed_lower": None,
                "survival_delayed_upper": None,
                "risk_base": None,
                "risk_base_lower": None,
                "risk_base_upper": None,
                "risk_delayed": None,
                "risk_delayed_lower": None,
                "risk_delayed_upper": None,
                "survival_diff_base_minus_delayed": None,
                "risk_diff_delayed_minus_base": None,
                "risk_diff_delayed_minus_base_lower": None,
                "risk_diff_delayed_minus_base_upper": None,
                "risk_ratio_delayed_over_base": None,
                "never_treated_survival": float(s),
                "never_treated_survival_lower": float(s_lo),
                "never_treated_survival_upper": float(s_hi),
                "never_treated_risk": float(r),
                "never_treated_risk_lower": float(r_lo),
                "never_treated_risk_upper": float(r_hi),
            }
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    for base_time in base_treat_times:
        delayed_time = float(base_time + delay_days)

        surv_base_out, risk_base_out = _predict_fixed_schedule(
            ref_df,
            model,
            times=times,
            fixed_treatment_time=float(base_time),
            n_boot=args.n_boot,
            alpha=args.alpha,
            bootstrap_seed=args.bootstrap_seed,
        )
        surv_delayed_out, risk_delayed_out = _predict_fixed_schedule(
            ref_df,
            model,
            times=times,
            fixed_treatment_time=delayed_time,
            n_boot=args.n_boot,
            alpha=args.alpha,
            bootstrap_seed=args.bootstrap_seed,
        )

        surv_base = surv_base_out["point"][0, :]
        surv_base_lower = surv_base_out["lower"][0, :]
        surv_base_upper = surv_base_out["upper"][0, :]

        surv_delayed = surv_delayed_out["point"][0, :]
        surv_delayed_lower = surv_delayed_out["lower"][0, :]
        surv_delayed_upper = surv_delayed_out["upper"][0, :]

        risk_base = risk_base_out["point"][0, :]
        risk_base_lower = risk_base_out["lower"][0, :]
        risk_base_upper = risk_base_out["upper"][0, :]

        risk_delayed = risk_delayed_out["point"][0, :]
        risk_delayed_lower = risk_delayed_out["lower"][0, :]
        risk_delayed_upper = risk_delayed_out["upper"][0, :]

        if args.n_boot > 0:
            contrast_out = predict_risk_contrast_bootstrap(
                ref_df,
                model,
                times=times,
                scenario_a={
                    "counterfactual_mode": "fixed",
                    "fixed_treatment_time": float(base_time),
                },
                scenario_b={
                    "counterfactual_mode": "fixed",
                    "fixed_treatment_time": delayed_time,
                },
                n_boot=args.n_boot,
                alpha=args.alpha,
                seed=args.bootstrap_seed,
                treatment_time_col="treatment_time",
                hard_fail=True,
            )
            contrast_point = contrast_out["point"][0, :]
            contrast_lower = contrast_out["lower"][0, :]
            contrast_upper = contrast_out["upper"][0, :]
        else:
            contrast_point = risk_delayed - risk_base
            contrast_lower = contrast_point.copy()
            contrast_upper = contrast_point.copy()

        label = f"delay_{int(delay_days)}d_from_{int(base_time)}"
        ax.plot(
            times,
            contrast_point,
            marker="o",
            label=label,
        )
        if args.n_boot > 0:
            ax.fill_between(times, contrast_lower, contrast_upper, alpha=0.2)

        for (
            t,
            sb,
            sb_lo,
            sb_hi,
            sd,
            sd_lo,
            sd_hi,
            rb,
            rb_lo,
            rb_hi,
            rd,
            rd_lo,
            rd_hi,
            cp,
            cl,
            cu,
            ns,
            nr,
        ) in zip(
            times,
            surv_base,
            surv_base_lower,
            surv_base_upper,
            surv_delayed,
            surv_delayed_lower,
            surv_delayed_upper,
            risk_base,
            risk_base_lower,
            risk_base_upper,
            risk_delayed,
            risk_delayed_lower,
            risk_delayed_upper,
            contrast_point,
            contrast_lower,
            contrast_upper,
            never_surv,
            never_risk,
        ):
            rows.append(
                {
                    "contrast": label,
                    "base_treat_time": float(base_time),
                    "delayed_treat_time": delayed_time,
                    "delay_days": delay_days,
                    "horizon_days": float(t),
                    "survival_base": float(sb),
                    "survival_base_lower": float(sb_lo),
                    "survival_base_upper": float(sb_hi),
                    "survival_delayed": float(sd),
                    "survival_delayed_lower": float(sd_lo),
                    "survival_delayed_upper": float(sd_hi),
                    "risk_base": float(rb),
                    "risk_base_lower": float(rb_lo),
                    "risk_base_upper": float(rb_hi),
                    "risk_delayed": float(rd),
                    "risk_delayed_lower": float(rd_lo),
                    "risk_delayed_upper": float(rd_hi),
                    "survival_diff_base_minus_delayed": float(sb - sd),
                    "risk_diff_delayed_minus_base": float(cp),
                    "risk_diff_delayed_minus_base_lower": float(cl),
                    "risk_diff_delayed_minus_base_upper": float(cu),
                    "risk_ratio_delayed_over_base": (float(rd / rb) if rb > 0 else np.nan),
                    "never_treated_survival": float(ns),
                    "never_treated_survival_lower": np.nan,
                    "never_treated_survival_upper": np.nan,
                    "never_treated_risk": float(nr),
                    "never_treated_risk_lower": np.nan,
                    "never_treated_risk_upper": np.nan,
                }
            )

    df = pd.DataFrame(rows)

    csv_long = out_dir / "reference_delay_contrasts.csv"
    df.to_csv(csv_long, index=False)

    df_wide = (
        df.loc[df["contrast"] != "never_treated"]
        .pivot_table(
            index=["contrast", "base_treat_time", "delayed_treat_time", "delay_days"],
            columns="horizon_days",
            values=[
                "survival_base",
                "survival_base_lower",
                "survival_base_upper",
                "survival_delayed",
                "survival_delayed_lower",
                "survival_delayed_upper",
                "risk_base",
                "risk_base_lower",
                "risk_base_upper",
                "risk_delayed",
                "risk_delayed_lower",
                "risk_delayed_upper",
                "survival_diff_base_minus_delayed",
                "risk_diff_delayed_minus_base",
                "risk_diff_delayed_minus_base_lower",
                "risk_diff_delayed_minus_base_upper",
                "risk_ratio_delayed_over_base",
            ],
        )
        .sort_index(axis=1)
    )
    df_wide.columns = [
        f"{metric}_t{int(h)}" for metric, h in df_wide.columns.to_flat_index()
    ]
    df_wide = df_wide.reset_index()

    csv_wide = out_dir / "reference_delay_contrasts_wide.csv"
    df_wide.to_csv(csv_wide, index=False)

    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Absolute risk increase from delay")
    ax.set_title("Reference-patient delay contrasts")
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / "reference_delay_contrast_plot.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    ref_df.to_csv(out_dir / "reference_patient.csv", index=False)

    print(f"Wrote: {csv_long}")
    print(f"Wrote: {csv_wide}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {out_dir / 'reference_patient.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        raise