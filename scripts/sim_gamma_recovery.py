from __future__ import annotations

import numpy as np
import pandas as pd

from peph.data.long import expand_long
from peph.model.fit import fit_peph
from peph.sim.ttt_effect import simulate_peph_ttt_effect_dataset


def run_ttt_recovery_seed_sweep(
    *,
    seeds=range(20),
    n=8000,
    gamma_true=-0.60,
):
    breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
    nu_true = np.array([0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010], dtype=float)

    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.16,
        "tumor_size_log": 0.22,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.30,
        "stageIII": 0.55,
    }

    rows = []

    for seed in seeds:
        wide = simulate_peph_ttt_effect_dataset(
            n=n,
            breaks=breaks,
            nu=nu_true,
            beta=beta_true,
            gamma_treated=gamma_true,
            admin_censor=float(breaks[-1]),
            random_censor_rate=0.0007,
            max_treatment_time=365.0,
            seed=int(seed),
            return_latent_truth=True,
        )

        long_df = expand_long(
            wide,
            id_col="id",
            time_col="time",
            event_col="event",
            x_cols=[
                "age_per10_centered",
                "cci",
                "tumor_size_log",
                "ses",
                "sex",
                "stage",
            ],
            breaks=breaks,
            cut_times_col="treatment_time",
            td_treatment_col="treatment_time",
            treated_td_col="treated_td",
        )

        fitted = fit_peph(
            long_train=long_df,
            breaks=breaks,
            x_numeric=["age_per10_centered", "cci", "tumor_size_log", "ses"],
            x_td_numeric=["treated_td"],
            x_categorical=["sex", "stage"],
            categorical_reference_levels={"sex": "F", "stage": "I"},
            n_train_subjects=int(wide["id"].nunique()),
            covariance="classical",
        )

        param_names = list(fitted.param_names)
        params = np.asarray(fitted.params, dtype=float)

        gamma_hat = float(params[param_names.index("treated_td")])

        rows.append(
            {
                "seed": int(seed),
                "gamma_true": float(gamma_true),
                "gamma_hat": gamma_hat,
                "bias": gamma_hat - float(gamma_true),
                "abs_error": abs(gamma_hat - float(gamma_true)),
                "treated_observed_prop": float(wide["treatment_time"].notna().mean()),
                "treated_row_prop": float(long_df["treated_td"].mean()),
                "event_rate": float(wide["event"].mean()),
                "n_long": int(len(long_df)),
            }
        )

    out = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

    summary = {
        "n_seeds": int(len(out)),
        "gamma_true": float(gamma_true),
        "gamma_hat_mean": float(out["gamma_hat"].mean()),
        "gamma_hat_sd": float(out["gamma_hat"].std(ddof=1)),
        "gamma_hat_min": float(out["gamma_hat"].min()),
        "gamma_hat_max": float(out["gamma_hat"].max()),
        "bias_mean": float(out["bias"].mean()),
        "abs_error_mean": float(out["abs_error"].mean()),
        "abs_error_median": float(out["abs_error"].median()),
        "p05": float(out["gamma_hat"].quantile(0.05)),
        "p25": float(out["gamma_hat"].quantile(0.25)),
        "p50": float(out["gamma_hat"].quantile(0.50)),
        "p75": float(out["gamma_hat"].quantile(0.75)),
        "p95": float(out["gamma_hat"].quantile(0.95)),
        "sign_failures": int((out["gamma_hat"] >= 0.0).sum()),
        "within_0p10": float((out["abs_error"] < 0.10).mean()),
        "within_0p15": float((out["abs_error"] < 0.15).mean()),
        "within_0p20": float((out["abs_error"] < 0.20).mean()),
        "treated_observed_prop_mean": float(out["treated_observed_prop"].mean()),
        "treated_row_prop_mean": float(out["treated_row_prop"].mean()),
        "event_rate_mean": float(out["event_rate"].mean()),
    }

    return out, summary


if __name__ == "__main__":
    df_res, summ = run_ttt_recovery_seed_sweep(seeds=range(20), n=8000, gamma_true=-0.60)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("\nPer-seed results:\n")
    print(df_res)

    print("\nSummary:\n")
    for k, v in summ.items():
        print(f"{k}: {v}")

    # optional save
    df_res.to_csv("data/dev/ttt_gamma_seed_sweep.csv", index=False)
    print("\nWrote: data/dev/ttt_gamma_seed_sweep.csv")