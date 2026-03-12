from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peph.model.result import FittedPEPHModel


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _summarize_coefficients(
    model: FittedPEPHModel,
    coef_df: pd.DataFrame | None,
    metadata: dict[str, Any] | None,
) -> None:
    _print_header("COEFFICIENTS")

    if coef_df is not None:
        show_cols = [c for c in ["term", "coef", "se", "z", "p_value", "hr", "ci_lower", "ci_upper"] if c in coef_df.columns]
        if show_cols:
            print(coef_df[show_cols].to_string(index=False))
        else:
            print("coef_table.parquet found but expected columns were not present.")
    else:
        print("coef_table.parquet not found. Falling back to model parameters.\n")
        for name, val in zip(model.param_names, model.params):
            print(f"{name:20s} {val: .4f}")

    if "treated_td" in model.param_names:
        j = model.param_names.index("treated_td")
        gamma_hat = float(model.params[j])
        hr_hat = float(np.exp(gamma_hat))
        print("\nTTT effect:")
        print(f"  gamma_hat (treated_td coef): {gamma_hat:.4f}")
        print(f"  HR_post_treatment          : {hr_hat:.4f}")

        if metadata is not None and "gamma_treated" in metadata:
            gamma_true = float(metadata["gamma_treated"])
            print(f"  gamma_true                : {gamma_true:.4f}")
            print(f"  error                     : {gamma_hat - gamma_true:+.4f}")

    if metadata is not None and "beta" in metadata:
        beta_true = dict(metadata["beta"])
        print("\nApproximate comparison to true baseline-effect parameters:")
        for term, val in zip(model.param_names, model.params):
            if term in beta_true:
                print(
                    f"  {term:20s} hat={float(val): .4f}  true={float(beta_true[term]): .4f}  "
                    f"err={float(val) - float(beta_true[term]):+.4f}"
                )


def _plot_baseline_hazard(
    model: FittedPEPHModel,
    baseline_df: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    _print_header("BASELINE HAZARD")

    if baseline_df is not None:
        print("baseline_table.parquet found.")
        print(baseline_df.head().to_string(index=False))

        if {"start", "end", "nu"}.issubset(baseline_df.columns):
            mids = 0.5 * (baseline_df["start"].to_numpy(dtype=float) + baseline_df["end"].to_numpy(dtype=float))
            nu = baseline_df["nu"].to_numpy(dtype=float)
        else:
            # fallback
            breaks = np.asarray(model.breaks, dtype=float)
            mids = 0.5 * (breaks[:-1] + breaks[1:])
            nu = np.asarray(model.nu, dtype=float)
    else:
        print("baseline_table.parquet not found. Using model.nu.")
        breaks = np.asarray(model.breaks, dtype=float)
        mids = 0.5 * (breaks[:-1] + breaks[1:])
        nu = np.asarray(model.nu, dtype=float)

    print("\nBaseline hazard by interval:")
    for i, val in enumerate(nu):
        print(f"  interval {i:2d}: nu = {float(val):.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mids, nu, marker="o")
    ax.set_xlabel("Interval midpoint (days)")
    ax.set_ylabel("Baseline hazard")
    ax.set_title("Piecewise baseline hazard")
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_hazard_check.png", dpi=200)
    plt.close(fig)


def _summarize_frailty(
    run_dir: Path,
    metadata: dict[str, Any] | None,
    out_dir: Path,
) -> None:
    frailty_df = _safe_read_parquet(run_dir / "frailty_table.parquet")
    frailty_summary_path = run_dir / "frailty_summary.json"

    if frailty_df is None and not frailty_summary_path.exists():
        return

    _print_header("SPATIAL FRAILTY")

    if frailty_summary_path.exists():
        frailty_summary = _load_json(frailty_summary_path)
        for k in ["n_areas", "u_mean", "u_sd", "u_min", "u_max", "rho_hat", "tau_hat"]:
            if k in frailty_summary:
                print(f"{k:20s}: {frailty_summary[k]}")

        if metadata is not None:
            if "rho_true" in metadata:
                print(f"{'rho_true':20s}: {metadata['rho_true']}")
            if "tau_true" in metadata:
                print(f"{'tau_true':20s}: {metadata['tau_true']}")

    if frailty_df is not None and "u_hat" in frailty_df.columns:
        print("\nFrailty table preview:")
        print(frailty_df.head().to_string(index=False))

        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(frailty_df["u_hat"].to_numpy(dtype=float), bins=20)
        ax.set_xlabel("u_hat")
        ax.set_ylabel("Count")
        ax.set_title("Frailty estimate distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "frailty_histogram_check.png", dpi=200)
        plt.close(fig)


def _summarize_ttt(run_dir: Path) -> None:
    ttt_summary_path = run_dir / "ttt_summary.json"
    if not ttt_summary_path.exists():
        return

    _print_header("TTT DIAGNOSTICS")

    ttt = _load_json(ttt_summary_path)

    tp = ttt.get("treatment_process", {})
    lr = ttt.get("long_risk_time", {})
    te = ttt.get("treatment_effect", {})

    if tp:
        print("Treatment process:")
        for k in ["n_subjects", "n_treated_observed", "prop_treated_observed"]:
            if k in tp:
                print(f"  {k:24s}: {tp[k]}")
        if "treatment_time_summary" in tp:
            s = tp["treatment_time_summary"]
            for k in ["mean", "median", "q25", "q75", "max"]:
                if k in s:
                    print(f"  {'treatment_time_' + k:24s}: {s[k]}")

    if lr:
        print("\nLong-form exposure:")
        for k in [
            "person_time_total",
            "person_time_untreated",
            "person_time_treated",
            "events_total",
            "events_untreated",
            "events_treated",
            "prop_person_time_treated",
        ]:
            if k in lr:
                print(f"  {k:24s}: {lr[k]}")

    if te:
        print("\nTreatment effect:")
        for k in ["coefficient", "hazard_ratio", "se", "ci_lower", "ci_upper", "hr_ci_lower", "hr_ci_upper"]:
            if k in te:
                print(f"  {k:24s}: {te[k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model sanity checks on a completed pipeline run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to timestamped pipeline output directory.")
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional metadata JSON with true simulation parameters for comparison.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_path = run_dir / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model.json not found in: {run_dir}")

    metadata = _load_json(Path(args.metadata_json)) if args.metadata_json else None

    model = FittedPEPHModel.load(str(model_path))
    coef_df = _safe_read_parquet(run_dir / "coef_table.parquet")
    baseline_df = _safe_read_parquet(run_dir / "baseline_table.parquet")

    out_dir = run_dir / "sanity_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_header("RUN")
    print(f"run_dir      : {run_dir}")
    print(f"fit_backend  : {model.fit_backend}")
    print(f"n_subjects   : {model.n_train_subjects}")
    print(f"n_long_rows  : {model.n_train_long_rows}")

    _summarize_coefficients(model, coef_df, metadata)
    _plot_baseline_hazard(model, baseline_df, out_dir)
    _summarize_frailty(run_dir, metadata, out_dir)
    _summarize_ttt(run_dir)

    print("\nSaved sanity-check plots to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()