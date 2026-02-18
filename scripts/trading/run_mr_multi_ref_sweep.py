#!/usr/bin/env python
"""Launch script for multi-ref-instrument mean-reversion sweep.

Runs the same parameter grid across multiple reference instruments,
then produces cross-ref analytics (leaderboard, per-ref best params,
param aggregation).

Usage::

    python scripts/trading/run_mr_multi_ref_sweep.py \
        --config configs/mr_grid_v1.yaml \
        --data-path /mnt/data/20240101 \
        --ref-instruments UST_2Y,UST_5Y,UST_10Y

    python scripts/trading/run_mr_multi_ref_sweep.py \
        --config configs/mr_grid_v1.yaml \
        --data-path D:\\data\\20240101 \
        --ref-instruments UST_2Y,UST_5Y,UST_10Y \
        --ranking-metric total_pnl \
        --top-n 20 \
        --outdir runs/multi_ref_20240101
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_runner import (
    MultiRefSweepRunner,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mean-reversion sweep across multiple reference instruments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sweep YAML config (or config-map name).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Directory with parquet files.",
    )
    parser.add_argument(
        "--ref-instruments",
        type=str,
        required=False,
        help="Comma-separated reference instrument IDs, e.g. UST_2Y,UST_5Y,UST_10Y",
        default="UST_2Y,UST_5Y",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        required=False,
        help="Comma-separated instrument IDs (auto-detected per ref if omitted).",
    )

    ranking = parser.add_argument_group("ranking")
    ranking.add_argument(
        "--ranking-metric",
        type=str,
        default="sharpe_ratio",
        help="Metric used to rank scenarios.",
    )
    ranking.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows in per-ref-best and param leaderboard.",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--outdir", type=str, default=None, help="Output directory.")
    out.add_argument("--no-save", action="store_true", help="Do not persist results to disk.")
    out.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser.parse_args(argv)


def _print(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quiet = args.quiet

    ref_instruments = [s.strip() for s in args.ref_instruments.split(",") if s.strip()]
    if not ref_instruments:
        print("ERROR: --ref-instruments must contain at least one instrument ID.")
        return 1

    instruments = None
    if args.instruments:
        instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    _print(f"Loading sweep config from {args.config} ...", quiet)
    _print(f"Reference instruments ({len(ref_instruments)}): {ref_instruments}", quiet)

    t0 = time.perf_counter()
    result = MultiRefSweepRunner.run(
        config=args.config,
        ref_instrument_ids=ref_instruments,
        data_path=args.data_path,
        instrument_ids=instruments,
        output_dir=args.outdir,
        save=not args.no_save,
        top_n=args.top_n,
        ranking_metric=args.ranking_metric,
    )
    elapsed = time.perf_counter() - t0

    _print(f"\nMulti-ref sweep complete", quiet)
    _print(f"  Refs      : {len(ref_instruments)}", quiet)
    _print(f"  Scenarios : {len(result.cross_ref_summary)}", quiet)
    _print(f"  Elapsed   : {elapsed:.1f}s", quiet)

    if result.output_dir:
        _print(f"  Output    : {result.output_dir}", quiet)

    # Per-ref best
    if not result.per_ref_best.empty:
        metric = args.ranking_metric
        print(f"\n{'=' * 70}")
        print(f"BEST SCENARIO PER REF (by {metric.upper()})")
        print(f"{'=' * 70}")
        display_cols = [c for c in [
            "ref_instrument_id", "name", metric,
            "total_pnl", "sharpe_ratio", "max_drawdown", "n_trades",
        ] if c in result.per_ref_best.columns]
        # Deduplicate while preserving order
        display_cols = list(dict.fromkeys(display_cols))
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(result.per_ref_best[display_cols].to_string(index=False))

    # Param leaderboard
    if not result.param_leaderboard.empty:
        print(f"\n{'=' * 70}")
        print(f"PARAM LEADERBOARD (top {args.top_n} by mean {args.ranking_metric.upper()})")
        print(f"{'=' * 70}")
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(result.param_leaderboard.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
