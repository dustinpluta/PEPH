from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from peph.sim.joint_ttt_survival import simulate_joint_ttt_survival_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
    nu = [0.0030, 0.0024, 0.0018, 0.0014, 0.0010, 0.0008]

    beta_survival = {
        "age_per10_centered": 0.05,
        "cci": 0.08,
        "tumor_size_log": 0.10,
        "ses": -0.04,
        "sexM": 0.02,
        "stageII": 0.15,
        "stageIII": 0.25,
    }

    beta_treatment = {
        "age_per10_centered": 0.04,
        "cci": 0.08,
        "tumor_size_log": 0.06,
        "ses": -0.04,
        "sexM": 0.02,
        "stageII": 0.10,
        "stageIII": 0.22,
    }

    treatment_intercept = float(np.log(120.0))
    sigma_treatment = 0.45

    gamma_treated = -0.25
    tau_true = 1.0
    rho_true = 0.85
    admin_censor = 1825.0
    random_censor_rate = 0.0005
    seed = 123
    n = 10000

    df = simulate_joint_ttt_survival_dataset(
        n=n,
        breaks=breaks,
        nu=nu,
        beta_survival=beta_survival,
        gamma_treated=gamma_treated,
        beta_treatment=beta_treatment,
        sigma_treatment=sigma_treatment,
        treatment_intercept=treatment_intercept,
        zips_path=str(DATA_DIR / "zips.csv"),
        edges_path=str(DATA_DIR / "zip_adjacency.csv"),
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=tau_true,
        rho_true=rho_true,
        admin_censor=admin_censor,
        random_censor_rate=random_censor_rate,
        seed=seed,
        return_latent_truth=False,
    )

    csv_path = DATA_DIR / "joint_ttt_survival_dataset.csv"
    json_path = DATA_DIR / "joint_ttt_survival_metadata.json"

    df.to_csv(csv_path, index=False)

    metadata = {
        "n_subjects": int(len(df)),
        "breaks": breaks,
        "nu": list(nu),
        "beta_survival": dict(beta_survival),
        "gamma_treated": float(gamma_treated),
        "beta_treatment": dict(beta_treatment),
        "sigma_treatment": float(sigma_treatment),
        "tau_true": float(tau_true),
        "rho_true": float(rho_true),
        "admin_censor": float(admin_censor),
        "random_censor_rate": float(random_censor_rate),
        "seed": int(seed),
        "event_rate": float(df["event"].mean()),
        "treatment_event_rate": float(df["treatment_event"].mean()),
        "observed_treatment_fraction_for_survival": float(df["treatment_time"].notna().mean()),
        "median_observed_survival_time": float(df["time"].median()),
        "median_observed_treatment_time": float(df.loc[df["treatment_event"] == 1, "treatment_time_obs"].median()),
        "columns": list(df.columns),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote dataset: {csv_path}")
    print(f"Wrote metadata: {json_path}")
    print(f"n_subjects: {metadata['n_subjects']}")
    print(f"event_rate: {metadata['event_rate']:.4f}")
    print(f"treatment_event_rate: {metadata['treatment_event_rate']:.4f}")
    print(
        "observed_treatment_fraction_for_survival: "
        f"{metadata['observed_treatment_fraction_for_survival']:.4f}"
    )


if __name__ == "__main__":
    main()