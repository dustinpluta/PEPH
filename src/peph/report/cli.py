from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from peph.report.discover import discover_run_artifacts
from peph.report.format import format_metrics_summary, print_df_pretty
from peph.report.tables import (
    coef_with_hr,
    load_baseline_table,
    load_coef_table,
    load_frailty_table,
    load_metrics,
    top_terms,
)


def _ensure_out_dir(out_dir: Optional[str | Path]) -> Path:
    if out_dir is None:
        raise ValueError("--out-dir is required when --to csv")
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def cmd_summary(args: argparse.Namespace) -> None:
    art = discover_run_artifacts(args.run_dir)
    metrics = load_metrics(art)
    df = format_metrics_summary(metrics, horizons=args.horizons)

    if args.to == "csv":
        out_dir = _ensure_out_dir(args.out_dir)
        _save_csv(df, out_dir, "summary_metrics.csv")
    else:
        print_df_pretty(df, max_rows=args.max_rows, max_cols=args.max_cols)


def cmd_coef(args: argparse.Namespace) -> None:
    art = discover_run_artifacts(args.run_dir)
    df = load_coef_table(art)

    if args.hr:
        df = coef_with_hr(df)

    if args.top is not None:
        df = top_terms(df, top=args.top, sort=args.sort)

    if args.to == "csv":
        out_dir = _ensure_out_dir(args.out_dir)
        _save_csv(df, out_dir, "coef_table.csv")
    else:
        print_df_pretty(df, max_rows=args.max_rows, max_cols=args.max_cols)


def cmd_baseline(args: argparse.Namespace) -> None:
    art = discover_run_artifacts(args.run_dir)
    df = load_baseline_table(art)

    if args.to == "csv":
        out_dir = _ensure_out_dir(args.out_dir)
        _save_csv(df, out_dir, "baseline_table.csv")
    else:
        print_df_pretty(df, max_rows=args.max_rows, max_cols=args.max_cols)


def cmd_spatial(args: argparse.Namespace) -> None:
    art = discover_run_artifacts(args.run_dir)

    if art.frailty_table is None:
        print(f"No spatial frailty artifacts found under: {art.run_dir}")
        return

    df = load_frailty_table(art)

    if "u_hat" in df.columns:
        df = df.sort_values("u_hat", ascending=False)
        if args.top is not None:
            top = int(args.top)
            head = df.head(top).copy()
            tail = df.tail(top).copy()
            head.insert(0, "_rank", range(1, len(head) + 1))
            tail.insert(0, "_rank", range(len(df) - len(tail) + 1, len(df) + 1))
            df = pd.concat([head, tail], axis=0)

    if args.to == "csv":
        out_dir = _ensure_out_dir(args.out_dir)
        _save_csv(df, out_dir, "frailty_table.csv")
    else:
        print_df_pretty(df, max_rows=args.max_rows, max_cols=args.max_cols)


def cmd_paths(args: argparse.Namespace) -> None:
    art = discover_run_artifacts(args.run_dir)

    rows = []
    for name, p in [
        ("model.json", art.model_json),
        ("metrics.json", art.metrics_json),
        ("coef_table.parquet", art.coef_table),
        ("baseline_table.parquet", art.baseline_table),
        ("frailty_table.parquet", art.frailty_table),
        ("frailty_summary.json", art.frailty_summary),
        ("spatial_autocorr.json", art.spatial_autocorr),
        ("inference.json", art.inference_json),
        ("plots_dir", art.plots_dir),
        ("predictions_dir", art.predictions_dir),
        ("tables_dir", art.tables_dir),
    ]:
        rows.append((name, str(p) if p is not None else "NA"))

    df = pd.DataFrame(rows, columns=["artifact", "path"])

    if args.to == "csv":
        out_dir = _ensure_out_dir(args.out_dir)
        _save_csv(df, out_dir, "artifact_paths.csv")
    else:
        print_df_pretty(df, max_rows=args.max_rows, max_cols=args.max_cols)


def build_report_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the `peph report ...` subcommand.
    """
    p = subparsers.add_parser("report", help="Inspect run artifacts and print/save result tables.")
    sp = p.add_subparsers(dest="report_cmd", required=True)

    def add_common(a: argparse.ArgumentParser) -> None:
        a.add_argument("--run-dir", required=True, help="Run output directory containing model.json, tables, metrics.")
        a.add_argument("--to", choices=["console", "csv"], default="console", help="Output destination.")
        a.add_argument("--out-dir", default=None, help="Output directory for --to csv.")
        a.add_argument("--max-rows", type=int, default=40, help="Max rows for console printing.")
        a.add_argument("--max-cols", type=int, default=50, help="Max cols for console printing.")

    ps = sp.add_parser("summary", help="Print/save a compact evaluation summary from metrics.json.")
    add_common(ps)
    ps.add_argument("--horizons", type=int, nargs="+", default=[365, 730, 1825], help="Horizons to display.")
    ps.set_defaults(func=cmd_summary)

    pc = sp.add_parser("coef", help="Print/save coefficient table (defaults to coef_table.parquet).")
    add_common(pc)
    pc.add_argument("--top", type=int, default=None, help="Show only top N rows after sorting.")
    pc.add_argument("--sort", choices=["abs_z", "p", "abs_beta"], default="abs_z", help="Sorting rule for --top.")
    pc.add_argument("--hr", action="store_true", help="Add hazard ratio columns (exp(beta), exp(CI)).")
    pc.set_defaults(func=cmd_coef)

    pb = sp.add_parser("baseline", help="Print/save baseline hazard table.")
    add_common(pb)
    pb.set_defaults(func=cmd_baseline)

    px = sp.add_parser("spatial", help="Print/save spatial frailty table (if present).")
    add_common(px)
    px.add_argument("--top", type=int, default=20, help="Show top/bottom by u_hat (if present).")
    px.set_defaults(func=cmd_spatial)

    pp = sp.add_parser("paths", help="Print/save discovered artifact paths under the run-dir.")
    add_common(pp)
    pp.set_defaults(func=cmd_paths)