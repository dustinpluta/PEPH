import numpy as np

from peph.sim import PHSimSpec, simulate_ph_wide


def test_simulated_schema_and_admin_censor_mass() -> None:
    spec = PHSimSpec(
        breaks=[0, 30, 90, 180, 365, 730, 1825],
        nu=[0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010],
        beta={
            "age_per10_centered": 0.10,
            "cci": 0.18,
            "tumor_size_log": 0.25,
            "ses": -0.08,
            "sexM": 0.05,
            "stageII": 0.35,
            "stageIII": 0.70,
            "stageIV": 1.20,
        },
        seed=7,
        censoring_enabled=True,
        censoring_rate=0.0008,
    )

    df = simulate_ph_wide(2000, spec, include_debug_cols=True)

    # required columns
    for c in ["id", "time", "event", "age_per10_centered", "cci", "tumor_size_log", "ses", "sex", "stage"]:
        assert c in df.columns

    # event indicator valid
    assert set(df["event"].unique()).issubset({0, 1})
    assert (df["time"] >= 0).all()

    # should have at least some administrative censoring at the max horizon with these defaults
    admin = spec.breaks[-1]
    frac_admin_cens = float(((df["time"] == admin) & (df["event"] == 0)).mean())
    # not a strict guarantee, but should generally be >0 with these settings
    assert frac_admin_cens >= 0.001