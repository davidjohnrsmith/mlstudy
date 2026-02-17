#!/usr/bin/env python
"""End-to-end launch script for the mean-reversion L2-book backtester.

Runs a parameter sweep from a YAML config file.  The YAML stores
platform-independent settings (filenames, instruments, grid, thresholds).
The ``--data-path`` is always supplied at runtime so the same config
works on Linux and Windows without modification.

Usage::

    python scripts/trading/run_mr_backtest.py \\
        --config configs/mr_grid_v1.yaml \\
        --data-path /mnt/data/20240101          # Linux

    python scripts/trading/run_mr_backtest.py \\
        --config configs/mr_grid_v1.yaml \\
        --data-path D:\\data\\20240101            # Windows

    python scripts/trading/run_mr_backtest.py \\
        --config configs/mr_grid_v1.yaml \\
        --data-path data/20240101 \\
        --outdir runs/sweep_20240101
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mean-reversion L2 backtest sweep.",
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

    from mlstudy.trading.backtest.mean_reversion import run_sweep_from_config

    _print(f"Loading sweep config from {args.config} ...", quiet)

    t0 = time.perf_counter()
    result = run_sweep_from_config(
        config=args.config,
        data_path=args.data_path,
        output_dir=args.outdir,
        save=not args.no_save,
    )
    elapsed = time.perf_counter() - t0

    cfg = result.config
    table = result.table

    _print(f"\nSweep '{cfg.grid_name}' complete", quiet)
    _print(f"  Scenarios : {len(table)}", quiet)
    _print(f"  Elapsed   : {elapsed:.1f}s", quiet)

    if result.output_dir:
        _print(f"  Output    : {result.output_dir}", quiet)

    # Print top results
    if not table.empty:
        sort_col = "total_pnl" if "total_pnl" in table.columns else table.columns[0]
        top = table.sort_values(sort_col, ascending=False).head(10)
        print(f"\n{'=' * 70}")
        print(f"TOP 10 SCENARIOS BY {sort_col.upper()}")
        print(f"{'=' * 70}")
        display_cols = [c for c in [
            "name", "total_pnl", "sharpe_ratio", "max_drawdown",
            "n_trades", "hit_rate", "sortino_ratio",
        ] if c in top.columns]
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(top[display_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
