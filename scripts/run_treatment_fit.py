from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from peph.treatment.fit import fit_treatment_lognormal_aft
from peph.treatment.report import (
    summarize_treatment_coefficients,
    summarize_treatment_model,
    summarize_treatment_reference_predictions,
)


def _parse_list_arg(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_reference_levels(text: str) -> dict[str, str]:
    """
    Parse strings like:
        "sex=F,stage=I"
    """
    out: dict[str, str] = {}
    for piece in _parse_list_arg(text):
        if "=" not in piece:
            raise ValueError(
                "Reference levels must be formatted like 'sex=F,stage=I'"
            )
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_float_list(text: str) -> list[float]:
    vals: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a log-normal AFT treatment-time model and write artifacts."
    )

    parser.add_argument("--data", required=True, help="Path to wide CSV input")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    parser.add_argument(
        "--time-col",
        default="treatment_time_obs",
        help="Observed treatment time column",
    )
    parser.add_argument(
        "--event-col",
        default="treatment_event",
        help="Treatment event indicator column (1=treatment observed, 0=censored)",
    )

    parser.add_argument(
        "--x-numeric",
        default="age_per10_centered,cci,ses",
        help="Comma-separated numeric covariate columns",
    )
    parser.add_argument(
        "--x-categorical",
        default="sex,stage",
        help="Comma-separated categorical covariate columns",
    )
    parser.add_argument(
        "--reference-levels",
        default="sex=F,stage=I",
        help="Comma-separated reference levels, e.g. 'sex=F,stage=I'",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Optimizer max iterations",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Optimizer tolerance",
    )
    parser.add_argument(
        "--optimizer-method",
        default="L-BFGS-B",
        help="scipy.optimize.minimize method",
    )

    parser.add_argument(
        "--write-reference-predictions",
        action="store_true",
        help="Also write reference predictions for the first few rows of the input data",
    )
    parser.add_argument(
        "--reference-n",
        type=int,
        default=5,
        help="Number of rows to use for optional reference predictions",
    )
    parser.add_argument(
        "--reference-horizons",
        default="30,60,90,180",
        help="Comma-separated horizons for optional reference predictions",
    )
    parser.add_argument(
        "--reference-quantiles",
        default="0.25,0.75",
        help="Comma-separated quantiles for optional reference predictions",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found: {data_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_numeric = _parse_list_arg(args.x_numeric)
    x_categorical = _parse_list_arg(args.x_categorical)
    reference_levels = _parse_reference_levels(args.reference_levels)

    df = pd.read_csv(data_path)

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col=args.time_col,
        treatment_event_col=args.event_col,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=reference_levels,
        max_iter=args.max_iter,
        tol=args.tol,
        optimizer_method=args.optimizer_method,
    )

    model_path = out_dir / "treatment_model.json"
    coef_path = out_dir / "treatment_coefficients.csv"
    summary_path = out_dir / "treatment_summary.json"

    fitted.save(model_path)

    coef_df = summarize_treatment_coefficients(fitted)
    coef_df.to_csv(coef_path, index=False)

    summary = summarize_treatment_model(fitted)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {coef_path}")
    print(f"Wrote: {summary_path}")

    if args.write_reference_predictions:
        n_ref = min(int(args.reference_n), len(df))
        ref_df = df.iloc[:n_ref].copy()

        horizons = _parse_float_list(args.reference_horizons)
        quantiles = _parse_float_list(args.reference_quantiles)

        ref_out = summarize_treatment_reference_predictions(
            ref_df,
            fitted,
            horizons=horizons,
            quantiles=quantiles,
            hard_fail=True,
        )

        ref_path = out_dir / "treatment_reference_predictions.csv"
        ref_out.to_csv(ref_path, index=False)
        print(f"Wrote: {ref_path}")


if __name__ == "__main__":
    main()