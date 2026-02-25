#!/usr/bin/env python
"""End-to-end launch script for the LP portfolio backtester sweep.

Runs a parameter sweep from a YAML config file.  The YAML stores
platform-independent settings (filenames, grid, thresholds).
Runtime parameters (``--data-path``, ``--instruments``, ``--hedges``)
are supplied at launch so the same config works across platforms
and different instrument sets.

Usage::

    # Auto-detect instruments from meta, hedges from hedge_ratios:
    python scripts/trading/run_portfolio_sweep.py \\
        --config portfolio_grid_v1 \\
        --data-path /mnt/data/20240101

    # Explicit instrument and hedge IDs:
    python scripts/trading/run_portfolio_sweep.py \\
        --config configs/portfolio_grid_v1.yaml \\
        --data-path /mnt/data/20240101 \\
        --instruments UST_2Y,UST_5Y,UST_10Y \\
        --hedges UST_30Y

    # Custom output directory, no persistence:
    python scripts/trading/run_portfolio_sweep.py \\
        --config portfolio_grid_debug \\
        --data-path D:\\data\\20240101 \\
        --outdir runs/debug_20240101 \\
        --no-save
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from mlstudy.trading.backtest.portfolio.sweep.sweep_runner import PortfolioSweepRunner

logger = logging.getLogger(__name__)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LP portfolio backtest sweep.",
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
        help="Directory with parquet files (book, mid, dv01, signal, meta, hedge_ratios).",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated trading instrument IDs.  If omitted, auto-detected from meta file.",
    )
    parser.add_argument(
        "--hedges",
        type=str,
        default=None,
        help="Comma-separated hedge instrument IDs.  If omitted, auto-detected from hedge_ratios file.",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--outdir", type=str, default=None, help="Output directory.")
    out.add_argument("--no-save", action="store_true", help="Do not persist results to disk.")
    out.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    out.add_argument("--top-n", type=int, default=10, help="Number of top scenarios to display.")
    out.add_argument("--plot", action="store_true", help="Generate dashboard plots for top scenarios.")

    return parser.parse_args(argv)


def _print(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quiet = args.quiet

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    _print(f"Loading sweep config from {args.config} ...", quiet)

    instrument_ids = None
    if args.instruments:
        instrument_ids = [s.strip() for s in args.instruments.split(",") if s.strip()]

    hedge_ids = None
    if args.hedges:
        hedge_ids = [s.strip() for s in args.hedges.split(",") if s.strip()]

    t0 = time.perf_counter()
    result = PortfolioSweepRunner.run_sweep_from_config(
        config=args.config,
        data_path=args.data_path,
        instrument_ids=instrument_ids,
        hedge_ids=hedge_ids,
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
    top_n = args.top_n
    if not table.empty:
        sort_col = "total_pnl" if "total_pnl" in table.columns else table.columns[0]
        top = table.sort_values(sort_col, ascending=False).head(top_n)
        print(f"\n{'=' * 70}")
        print(f"TOP {top_n} SCENARIOS BY {sort_col.upper()}")
        print(f"{'=' * 70}")
        display_cols = [c for c in [
            "name", "total_pnl", "sharpe_ratio", "max_drawdown",
            "sortino_ratio", "calmar_ratio",
        ] if c in top.columns]
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(top[display_cols].to_string(index=False))

    # Generate plots for top scenarios
    if args.plot and result.output_dir and result.top_full:
        try:
            from mlstudy.trading.backtest.portfolio.sweep.plots import plot_top_scenarios
            from mlstudy.trading.backtest.portfolio.sweep.sweep_results_reader import (
                PortfolioFullScenario,
            )
            from dataclasses import asdict

            scenarios_to_plot = result.top_full[:top_n]
            full_scenarios = []
            for sr in scenarios_to_plot:
                sc = PortfolioFullScenario(
                    spec={
                        "name": sr.scenario.name,
                        "tags": sr.scenario.tags,
                        "config": asdict(sr.scenario.cfg),
                        "metrics": asdict(sr.metrics),
                        "scenario_idx": sr.scenario_idx,
                    },
                    results=sr.results,
                    directory=result.output_dir,
                )
                full_scenarios.append(sc)

            plots_dir = result.output_dir / "plots"
            plot_top_scenarios(full_scenarios, save_dir=plots_dir)
            import matplotlib.pyplot as plt
            plt.close("all")
            _print(f"  Plots     : {plots_dir}", quiet)
        except ImportError:
            _print("  (matplotlib not available — skipping plots)", quiet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
