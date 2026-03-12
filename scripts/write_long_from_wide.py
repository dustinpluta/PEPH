from __future__ import annotations

import argparse
import pandas as pd

from peph.data.long import expand_long


def main() -> None:

    parser = argparse.ArgumentParser(description="Convert wide survival data to PE long format")

    parser.add_argument("--wide-csv", required=True)
    parser.add_argument("--out-csv", required=True)

    parser.add_argument(
        "--breaks",
        nargs="+",
        type=float,
        required=True,
        help="Piecewise hazard breakpoints",
    )

    parser.add_argument(
        "--x-cols",
        nargs="+",
        default=[
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "ses",
            "sex",
            "stage",
        ],
    )

    parser.add_argument("--id-col", default="id")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--event-col", default="event")

    parser.add_argument(
        "--treatment-col",
        default="treatment_time",
        help="Column containing treatment time",
    )

    args = parser.parse_args()

    wide = pd.read_csv(args.wide_csv)

    long_df = expand_long(
        wide,
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        x_cols=args.x_cols,
        breaks=args.breaks,
        cut_times_col=args.treatment_col,
        td_treatment_col=args.treatment_col,
        treated_td_col="treated_td",
    )

    long_df.to_csv(args.out_csv, index=False)

    print("Wrote long dataset:", args.out_csv)
    print("Rows:", len(long_df))


if __name__ == "__main__":
    main()