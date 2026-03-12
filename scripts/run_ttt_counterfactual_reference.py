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
    predict_survival_bootstrap,
)
from peph.model.result import FittedPEPHModel


def _parse_float_list(text: str | None) -> list[float]:
    if text is None or text.strip() == "":
        return []
    vals: list[float] = []
    for part in text.split(","):
        p = part.strip()
        if p == "":
            continue
        vals.append(float(p))
    return vals


def _maybe_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = x.strip()
    if s == "":
        return None
    if s.lower() in {"na", "nan", "none"}:
        return None
    return float(s)


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

    if args.observed_treatment_time is not None:
        row["treatment_time"] = float(args.observed_treatment_time)
    else:
        row["treatment_time"] = np.nan

    return pd.DataFrame([row])


def _scenario_survival_and_risk(
    ref_df: pd.DataFrame,
    model: FittedPEPHModel,
    times: list[float],
    *,
    counterfactual_mode: str,
    fixed_treatment_time: float | None = None,
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
            counterfactual_mode=counterfactual_mode,
            fixed_treatment_time=fixed_treatment_time,
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
            counterfactual_mode=counterfactual_mode,
            fixed_treatment_time=fixed_treatment_time,
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
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
        hard_fail=True,
    )
    risk_point = predict_risk(
        ref_df,
        model,
        times=times,
        treatment_time_col="treatment_time",
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
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
        description="Run counterfactual TTT predictions for a single reference patient."
    )

    parser.add_argument("--model", required=True, help="Path to fitted model.json")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write output tables and plots",
    )

    parser.add_argument(
        "--times",
        default="365,730,1825",
        help="Comma-separated prediction horizons in days, e.g. '365,730,1825'",
    )
    parser.add_argument(
        "--treat-at-times",
        default="30,60,90",
        help="Comma-separated fixed treatment times in days; never-treated is always included",
    )

    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-seed", type=int, default=123)

    # Reference patient fields
    parser.add_argument("--age-per10-centered", type=float, default=0.0)
    parser.add_argument("--cci", type=int, default=1)
    parser.add_argument("--tumor-size-log", type=float, default=3.5)
    parser.add_argument("--ses", type=float, default=0.0)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--stage", type=str, default="II")
    parser.add_argument("--zip", type=str, default=None)

    # Optional observed schedule
    parser.add_argument(
        "--observed-treatment-time",
        type=str,
        default=None,
        help="Optional observed treatment time in days; if provided, include observed-history prediction",
    )

    args = parser.parse_args()

    if args.n_boot < 0:
        raise ValueError("--n-boot must be nonnegative")
    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("--alpha must be in (0, 1)")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FittedPEPHModel.load(str(model_path))

    times = _parse_float_list(args.times)
    if len(times) == 0:
        raise ValueError("--times must contain at least one horizon")
    times = sorted(times)

    fixed_times = _parse_float_list(args.treat_at_times)
    fixed_times = sorted(set(fixed_times))

    args.observed_treatment_time = _maybe_float(args.observed_treatment_time)

    ref_df = _make_reference_row(args, model)

    scenarios: list[dict[str, Any]] = []

    # Always include never treated
    scenarios.append(
        {
            "scenario": "never_treated",
            "counterfactual_mode": "never",
            "fixed_treatment_time": None,
        }
    )

    # Optional observed-history scenario
    if args.observed_treatment_time is not None:
        scenarios.append(
            {
                "scenario": f"observed_treat_at_{int(args.observed_treatment_time)}",
                "counterfactual_mode": "observed",
                "fixed_treatment_time": None,
            }
        )

    # Fixed treatment schedules
    for ttx in fixed_times:
        scenarios.append(
            {
                "scenario": f"treat_at_{int(ttx)}",
                "counterfactual_mode": "fixed",
                "fixed_treatment_time": float(ttx),
            }
        )

    rows: list[dict[str, Any]] = []

    fig_s, ax_s = plt.subplots(figsize=(7, 5))
    fig_r, ax_r = plt.subplots(figsize=(7, 5))

    for sc in scenarios:
        surv_out, risk_out = _scenario_survival_and_risk(
            ref_df,
            model,
            times,
            counterfactual_mode=sc["counterfactual_mode"],
            fixed_treatment_time=sc["fixed_treatment_time"],
            n_boot=args.n_boot,
            alpha=args.alpha,
            bootstrap_seed=args.bootstrap_seed,
        )

        surv = surv_out["point"][0, :]
        risk = risk_out["point"][0, :]
        surv_lo = surv_out["lower"][0, :]
        surv_hi = surv_out["upper"][0, :]
        risk_lo = risk_out["lower"][0, :]
        risk_hi = risk_out["upper"][0, :]

        ax_s.plot(times, surv, marker="o", label=sc["scenario"])
        if args.n_boot > 0:
            ax_s.fill_between(times, surv_lo, surv_hi, alpha=0.2)

        ax_r.plot(times, risk, marker="o", label=sc["scenario"])
        if args.n_boot > 0:
            ax_r.fill_between(times, risk_lo, risk_hi, alpha=0.2)

        for t, s, s_lo, s_hi, r, r_lo, r_hi in zip(
            times, surv, surv_lo, surv_hi, risk, risk_lo, risk_hi
        ):
            rows.append(
                {
                    "scenario": sc["scenario"],
                    "counterfactual_mode": sc["counterfactual_mode"],
                    "fixed_treatment_time": sc["fixed_treatment_time"],
                    "horizon_days": float(t),
                    "survival": float(s),
                    "survival_lower": float(s_lo),
                    "survival_upper": float(s_hi),
                    "risk": float(r),
                    "risk_lower": float(r_lo),
                    "risk_upper": float(r_hi),
                }
            )

    summary_df = pd.DataFrame(rows)

    csv_path = out_dir / "reference_counterfactual_predictions.csv"
    summary_df.to_csv(csv_path, index=False)

    ax_s.set_xlabel("Horizon (days)")
    ax_s.set_ylabel("Predicted survival")
    ax_s.set_title("Reference-patient survival under treatment schedules")
    ax_s.legend()
    fig_s.tight_layout()
    surv_plot_path = out_dir / "reference_counterfactual_survival.png"
    fig_s.savefig(surv_plot_path, dpi=200)
    plt.close(fig_s)

    ax_r.set_xlabel("Horizon (days)")
    ax_r.set_ylabel("Predicted risk")
    ax_r.set_title("Reference-patient risk under treatment schedules")
    ax_r.legend()
    fig_r.tight_layout()
    risk_plot_path = out_dir / "reference_counterfactual_risk.png"
    fig_r.savefig(risk_plot_path, dpi=200)
    plt.close(fig_r)

    ref_df.to_csv(out_dir / "reference_patient.csv", index=False)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {surv_plot_path}")
    print(f"Wrote: {risk_plot_path}")
    print(f"Wrote: {out_dir / 'reference_patient.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        raise