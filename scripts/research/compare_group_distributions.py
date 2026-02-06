#!/usr/bin/env python
"""CLI for comparing distributions across groups.

Usage:
    python scripts/compare_group_distributions.py \
        --csv data/sample.csv \
        --group-col sector \
        --value-col returns \
        --outdir outputs/reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlstudy.research.analysis import compare_groups
from mlstudy.research.analysis import (
    plot_distance_heatmap,
    plot_group_distributions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare distributions across groups in a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--datetime-col",
        type=str,
        default=None,
        help="Name of datetime column for filtering.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        required=True,
        help="Name of column defining groups.",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        required=True,
        help="Name of column with values to compare.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated list of groups to include (default: all).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start datetime filter (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End datetime filter (inclusive).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=100,
        help="Minimum sample count per group.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/reports/",
        help="Output directory for reports and plots.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Read CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Parse datetime column if specified
    if args.datetime_col and args.datetime_col in df.columns:
        df[args.datetime_col] = pd.to_datetime(df[args.datetime_col])

    # Parse groups
    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",")]

    # Run comparison
    print(f"Comparing {args.value_col} across groups in {args.group_col}...")
    report = compare_groups(
        df=df,
        group_col=args.group_col,
        value_col=args.value_col,
        datetime_col=args.datetime_col,
        groups=groups,
        start=args.start,
        end=args.end,
        min_count=args.min_count,
    )

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write CSV reports
    print(f"Writing reports to {outdir}...")
    paths = report.to_csv(str(outdir))
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Generate plots
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Distance matrix heatmap
    if "wasserstein" in report.distance_matrices:
        heatmap_path = plots_dir / "distance_heatmap_wasserstein.png"
        plot_distance_heatmap(
            report.distance_matrices["wasserstein"],
            title=f"Wasserstein Distance: {args.value_col} by {args.group_col}",
            save_path=heatmap_path,
        )
        print(f"  heatmap: {heatmap_path}")

    # Distribution plots
    used_groups = report.group_summaries.index.tolist()
    if used_groups:
        plot_group_distributions(
            df=df,
            group_col=args.group_col,
            value_col=args.value_col,
            groups=used_groups,
            outdir=plots_dir,
        )
        print(f"  histogram: {plots_dir / 'histogram_overlay.png'}")
        print(f"  ecdf: {plots_dir / 'ecdf_overlay.png'}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
