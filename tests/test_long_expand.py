import numpy as np
import pandas as pd

from peph.data.long import expand_long


def test_expand_long_invariants() -> None:
    breaks = [0, 30, 90, 180, 365, 730, 1825]
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "time": [10, 30, 31, 2000],   # includes exact break (30) and beyond window
            "event": [0, 1, 1, 1],
            "x": [0.1, 0.2, 0.3, -0.1],
        }
    )

    long = expand_long(
        df,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=["x"],
        breaks=breaks,
    )

    # exposure sums: min(time, tmax)
    tmax = breaks[-1]
    for rid, t in zip(df["id"], df["time"]):
        exp_sum = long.loc[long["id"] == rid, "exposure"].sum()
        assert np.isclose(exp_sum, min(float(t), float(tmax)))

    # event row counts:
    # id=1 censored => 0
    assert long.loc[long["id"] == 1, "event"].sum() == 0

    # id=2 event at t=30 => assigned to interval starting at 30, i.e. k=1 ([30,90))
    sub2 = long.loc[long["id"] == 2]
    assert sub2["event"].sum() == 1
    assert int(sub2.loc[sub2["event"] == 1, "k"].iloc[0]) == 1

    # id=4 event time beyond tmax => treated as censored at tmax
    assert long.loc[long["id"] == 4, "event"].sum() == 0


def test_expand_long_with_subject_specific_cut_time() -> None:
    breaks = [0, 30, 90, 180, 365]
    df = pd.DataFrame(
        {
            "id": [1],
            "time": [200],
            "event": [0],
            "treatment_time": [80],
            "x": [1.25],
        }
    )

    long = expand_long(
        df,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=["x"],
        breaks=breaks,
        cut_times_col="treatment_time",
    ).reset_index(drop=True)

    assert long[["t0", "t1"]].to_dict("records") == [
        {"t0": 0.0, "t1": 30.0},
        {"t0": 30.0, "t1": 80.0},
        {"t0": 80.0, "t1": 90.0},
        {"t0": 90.0, "t1": 180.0},
        {"t0": 180.0, "t1": 200.0},
    ]

    assert long["k"].tolist() == [0, 1, 1, 2, 3]
    assert np.allclose(long["exposure"].to_numpy(), [30.0, 50.0, 10.0, 90.0, 20.0])
    assert long["event"].sum() == 0
    assert np.isclose(long["exposure"].sum(), 200.0)


def test_expand_long_ignores_cut_time_after_followup_or_missing() -> None:
    breaks = [0, 30, 90, 180, 365]
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [70, 70, 70],
            "event": [0, 0, 0],
            "cut": [80, np.nan, None],
            "x": [0.0, 0.0, 0.0],
        }
    )

    long = expand_long(
        df,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=["x"],
        breaks=breaks,
        cut_times_col="cut",
    )

    for rid in [1, 2, 3]:
        sub = long.loc[long["id"] == rid].reset_index(drop=True)
        assert sub[["t0", "t1"]].to_dict("records") == [
            {"t0": 0.0, "t1": 30.0},
            {"t0": 30.0, "t1": 70.0},
        ]
        assert np.isclose(sub["exposure"].sum(), 70.0)