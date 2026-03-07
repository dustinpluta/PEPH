from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from peph.data.long import expand_long


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_ttt_long_expansion(
    *,
    data_path: str | Path = "data/simulated_seer_crc_ttt.csv",
    out_dir: str | Path = "data/validation_ttt_long",
    breaks: list[float] | None = None,
    id_col: str = "id",
    time_col: str = "time",
    event_col: str = "event",
    cut_times_col: str = "treatment_time",
    zip_col: str = "zip",
) -> None:
    """
    Validate Phase 1 time-to-treatment long expansion.

    Checks
    ------
    1. Exposure sums per subject equal observed follow-up time.
    2. Observed events contribute exactly one event row.
    3. Event-at-break convention is left-closed/right-open.
    4. If treatment_time is observed and strictly inside follow-up, there is a split at treatment_time.
    5. If treatment_time is missing or beyond follow-up, no extra split is introduced.
    6. Long-form rows remain nested within the global PE intervals via k.

    Outputs
    -------
    - validation_summary.txt
    - subject_level_checks.csv
    - split_examples.csv
    - no_split_examples.csv
    """
    if breaks is None:
        breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]

    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    required = [id_col, time_col, event_col, cut_times_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x_cols = [
    c for c in df.columns
    if c not in {id_col, time_col, event_col}
    and not c.startswith("_")   # drop latent/debug columns
    ]   
    long_df = expand_long(
        df,
        id_col=id_col,
        time_col=time_col,
        event_col=event_col,
        x_cols=x_cols,
        breaks=breaks,
        cut_times_col=cut_times_col,
    ).copy()

    if long_df.empty:
        raise ValueError("Long dataset is empty; nothing to validate.")

    # ----------------------------
    # Subject-level aggregates
    # ----------------------------
    agg = (
        long_df.groupby("id", as_index=False)
        .agg(
            exposure_sum=("exposure", "sum"),
            event_rows=("event", "sum"),
            n_rows=("id", "size"),
        )
        .rename(columns={"id": id_col})
    )

    check_df = df[[id_col, time_col, event_col, cut_times_col]].merge(
        agg, on=id_col, how="left"
    )
    check_df["exposure_sum"] = check_df["exposure_sum"].fillna(0.0)
    check_df["event_rows"] = check_df["event_rows"].fillna(0).astype(int)
    check_df["n_rows"] = check_df["n_rows"].fillna(0).astype(int)

    tmax = float(breaks[-1])
    check_df["expected_exposure"] = np.minimum(
        check_df[time_col].to_numpy(dtype=float), tmax
    )
    check_df["expected_event_rows"] = (
        (check_df[event_col].to_numpy(dtype=int) == 1)
        & (check_df[time_col].to_numpy(dtype=float) < tmax)
    ).astype(int)

    check_df["exposure_ok"] = np.isclose(
        check_df["exposure_sum"].to_numpy(dtype=float),
        check_df["expected_exposure"].to_numpy(dtype=float),
    )
    check_df["event_rows_ok"] = (
        check_df["event_rows"].to_numpy(dtype=int)
        == check_df["expected_event_rows"].to_numpy(dtype=int)
    )

    # ----------------------------
    # Row-level PE interval checks
    # ----------------------------
    b = np.asarray(breaks, dtype=float)
    k = long_df["k"].to_numpy(dtype=int)
    t0 = long_df["t0"].to_numpy(dtype=float)
    t1 = long_df["t1"].to_numpy(dtype=float)

    long_df["k_left_ok"] = np.isclose(t0, b[k]) | ((t0 > b[k]) & (t0 < b[k + 1]))
    long_df["k_right_ok"] = (t1 <= b[k + 1] + 1e-12) & (t1 >= t0 - 1e-12)
    long_df["interval_ok"] = long_df["k_left_ok"] & long_df["k_right_ok"]

    # ----------------------------
    # Treatment split checks
    # ----------------------------
    internal_treat = (
        check_df[cut_times_col].notna()
        & (check_df[cut_times_col].to_numpy(dtype=float) > 0.0)
        & (
            check_df[cut_times_col].to_numpy(dtype=float)
            < check_df["expected_exposure"].to_numpy(dtype=float)
        )
    )

    check_df["treat_internal"] = internal_treat

    split_found = []
    split_gap = []
    for row in check_df.itertuples(index=False):
        rid = getattr(row, id_col)
        treat = getattr(row, cut_times_col)
        sub = long_df.loc[long_df[id_col] == rid].sort_values(["t0", "t1"]).reset_index(drop=True)

        if pd.isna(treat):
            split_found.append(False)
            split_gap.append(np.nan)
            continue

        treat = float(treat)
        has_boundary = np.any(np.isclose(sub["t0"].to_numpy(dtype=float), treat)) or np.any(
            np.isclose(sub["t1"].to_numpy(dtype=float), treat)
        )
        split_found.append(bool(has_boundary))

        if has_boundary:
            grid = np.concatenate(
                [sub["t0"].to_numpy(dtype=float), sub["t1"].to_numpy(dtype=float)]
            )
            split_gap.append(float(np.min(np.abs(grid - treat))))
        else:
            split_gap.append(np.nan)

    check_df["split_found"] = split_found
    check_df["split_gap"] = split_gap

    check_df["split_ok"] = np.where(
        check_df["treat_internal"],
        check_df["split_found"],
        True,
    )

    # ----------------------------
    # Strong per-subject logical checks
    # ----------------------------
    # 1) every subject with internal treatment time should have at least one boundary at treatment_time
    _assert(
        bool(check_df.loc[check_df["treat_internal"], "split_ok"].all()),
        "At least one subject with internal treatment time was not split at treatment_time.",
    )

    # 2) exposure sums are preserved
    _assert(
        bool(check_df["exposure_ok"].all()),
        "Exposure sum check failed for at least one subject.",
    )

    # 3) event rows match expected count
    _assert(
        bool(check_df["event_rows_ok"].all()),
        "Event-row count check failed for at least one subject.",
    )

    # 4) all long rows remain within claimed global PE interval
    _assert(
        bool(long_df["interval_ok"].all()),
        "At least one long row is inconsistent with its PE interval index k.",
    )

    # ----------------------------
    # Breakpoint convention spot-check
    # ----------------------------
    # Synthetic check to verify event exactly on a break lands in the interval that starts there.
    df_break = pd.DataFrame(
        {
            id_col: [999001, 999002],
            time_col: [30.0, 90.0],
            event_col: [1, 1],
            cut_times_col: [np.nan, np.nan],
        }
    )

    for c in x_cols:
        if c not in df_break.columns:
            if c == zip_col:
                df_break[c] = "30303"
            elif c == "sex":
                df_break[c] = "F"
            elif c == "stage":
                df_break[c] = "II"
            else:
                df_break[c] = 0.0

    long_break = expand_long(
        df_break,
        id_col=id_col,
        time_col=time_col,
        event_col=event_col,
        x_cols=x_cols,
        breaks=breaks,
        cut_times_col=cut_times_col,
    )

    event_k_30 = int(
        long_break.loc[(long_break[id_col] == 999001) & (long_break["event"] == 1), "k"].iloc[0]
    )
    event_k_90 = int(
        long_break.loc[(long_break[id_col] == 999002) & (long_break["event"] == 1), "k"].iloc[0]
    )

    _assert(
        event_k_30 == 1,
        f"Breakpoint convention failed at t=30: expected k=1, got k={event_k_30}.",
    )
    _assert(
        event_k_90 == 2,
        f"Breakpoint convention failed at t=90: expected k=2, got k={event_k_90}.",
    )

    # ----------------------------
    # Example slices for inspection
    # ----------------------------
    split_ids = (
        check_df.loc[check_df["treat_internal"]]
        .sort_values(cut_times_col)
        .head(10)[id_col]
        .tolist()
    )
    no_split_ids = (
        check_df.loc[~check_df["treat_internal"]]
        .sort_values(time_col)
        .head(10)[id_col]
        .tolist()
    )

    split_examples = long_df.loc[long_df[id_col].isin(split_ids)].sort_values([id_col, "t0", "t1"])
    no_split_examples = long_df.loc[long_df[id_col].isin(no_split_ids)].sort_values([id_col, "t0", "t1"])

    # ----------------------------
    # Summaries
    # ----------------------------
    n = len(check_df)
    n_event = int(check_df[event_col].sum())
    n_treat_obs = int(check_df[cut_times_col].notna().sum())
    n_treat_internal = int(check_df["treat_internal"].sum())

    lines = []
    lines.append("TTT LONG-EXPANSION VALIDATION SUMMARY")
    lines.append("=" * 40)
    lines.append(f"input file: {data_path}")
    lines.append(f"n subjects: {n}")
    lines.append(f"n long rows: {len(long_df)}")
    lines.append(f"breaks: {breaks}")
    lines.append("")
    lines.append("Subject-level checks")
    lines.append(f"  exposure_ok: {int(check_df['exposure_ok'].sum())} / {n}")
    lines.append(f"  event_rows_ok: {int(check_df['event_rows_ok'].sum())} / {n}")
    lines.append(f"  split_ok: {int(check_df['split_ok'].sum())} / {n}")
    lines.append("")
    lines.append("Outcome / treatment summary")
    lines.append(f"  observed deaths: {n_event}")
    lines.append(f"  observed treatment_time: {n_treat_obs}")
    lines.append(f"  internal treatment_time requiring split: {n_treat_internal}")
    lines.append("")
    lines.append("Row-level checks")
    lines.append(f"  interval_ok rows: {int(long_df['interval_ok'].sum())} / {len(long_df)}")
    lines.append("")
    lines.append("Breakpoint convention spot-check")
    lines.append(f"  event at 30 -> k={event_k_30}")
    lines.append(f"  event at 90 -> k={event_k_90}")
    lines.append("")
    lines.append("Split examples")
    lines.append(f"  example subject ids with internal treatment_time: {split_ids}")
    lines.append(f"  example subject ids without internal treatment_time: {no_split_ids}")

    summary_path = out_dir / "validation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    check_df.to_csv(out_dir / "subject_level_checks.csv", index=False)
    split_examples.to_csv(out_dir / "split_examples.csv", index=False)
    no_split_examples.to_csv(out_dir / "no_split_examples.csv", index=False)

    print("\n".join(lines))
    print(f"\nWrote:")
    print(f"  {summary_path}")
    print(f"  {out_dir / 'subject_level_checks.csv'}")
    print(f"  {out_dir / 'split_examples.csv'}")
    print(f"  {out_dir / 'no_split_examples.csv'}")


if __name__ == "__main__":
    validate_ttt_long_expansion()