#!/usr/bin/env python
"""Plot and display top-N results from a persisted portfolio sweep run.

Usage::

    # Show top 10 from a previous run:
    python scripts/trading/plot_portfolio_results.py \\
        --run-dir runs/portfolio_grid_v1/20240101_120000

    # Top 5, save plots:
    python scripts/trading/plot_portfolio_results.py \\
        --run-dir runs/portfolio_grid_v1/20240101_120000 \\
        --top-n 5 --plot

    # Just print the table, no plots:
    python scripts/trading/plot_portfolio_results.py \\
        --run-dir runs/portfolio_grid_v1/20240101_120000 \\
        --no-plot
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from mlstudy.trading.backtest.portfolio.sweep.sweep_results_reader import (
    PortfolioSweepResultsReader,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display and plot portfolio sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to a persisted sweep run directory.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top scenarios to display / plot.",
    )
    parser.add_argument(
        "--sort-by", type=str, default="total_pnl",
        help="Column to sort by (descending).",
    )
    parser.add_argument(
        "--plot", action="store_true", default=True,
        help="Generate dashboard plots (default: on).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Directory to save plots (default: <run-dir>/plots).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Loading sweep run from {args.run_dir} ...")
    run = PortfolioSweepResultsReader.load_sweep_run(args.run_dir)

    print(f"  Grid      : {run.grid_name}")
    print(f"  Scenarios : {run.n_scenarios}")
    if run.elapsed_seconds is not None:
        print(f"  Elapsed   : {run.elapsed_seconds:.1f}s")

    # --- Summary table ---
    table = run.summary if not run.summary.empty else run.all_metrics
    if table is not None and not table.empty:
        sort_col = args.sort_by if args.sort_by in table.columns else table.columns[0]
        top = table.sort_values(sort_col, ascending=False).head(args.top_n)

        print(f"\n{'=' * 70}")
        print(f"TOP {args.top_n} SCENARIOS BY {sort_col.upper()}")
        print(f"{'=' * 70}")
        display_cols = [c for c in [
            "name", "total_pnl", "sharpe_ratio", "max_drawdown",
            "sortino_ratio", "calmar_ratio", "n_trades", "hit_rate",
        ] if c in top.columns]
        with pd.option_context("display.max_columns", 20, "display.width", 140):
            print(top[display_cols].to_string(index=False))
    else:
        print("  (no summary table found)")

    # --- Plots ---
    if not args.no_plot and args.plot and run.full_scenarios:
        try:
            from mlstudy.trading.backtest.portfolio.sweep.plots import (
                plot_top_scenarios,
            )
            from pathlib import Path

            scenarios = run.full_scenarios[:args.top_n]
            plot_dir = Path(args.plot_dir) if args.plot_dir else run.directory / "plots"

            print(f"\nPlotting {len(scenarios)} scenarios to {plot_dir} ...")
            plot_top_scenarios(scenarios, save_dir=plot_dir)

            import matplotlib.pyplot as plt
            plt.close("all")
            print(f"  Done. Plots saved to {plot_dir}")
        except ImportError:
            print("  (matplotlib not available — skipping plots)")
    elif not run.full_scenarios:
        print("\n  (no full scenarios saved — cannot generate plots)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
