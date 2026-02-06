#!/usr/bin/env python
"""Run fly backtest with yield-based signals and price-based PnL.

Usage:
    python scripts/run_fly_backtest.py --csv data/panel.csv --tenors 2,5,10
    python scripts/run_fly_backtest.py --csv data/panel.csv --sizing dv01_target --dv01 10000

Example:
    # With fixed notional sizing
    python scripts/run_fly_backtest.py \
        --csv data/bond_panel.csv \
        --tenors 2,5,10 \
        --window 20 \
        --entry-z 2.0 \
        --exit-z 0.5 \
        --sizing fixed_notional \
        --notional 1000000 \
        --outdir outputs/backtests

    # With DV01 target sizing
    python scripts/run_fly_backtest.py \
        --csv data/bond_panel.csv \
        --tenors 2,5,10 \
        --sizing dv01_target \
        --dv01 10000 \
        --outdir outputs/backtests
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from mlstudy.trading.backtest import (
    BacktestConfig,
    SizingMode,
    backtest_fly_from_panel,
    compute_metrics,
    generate_report,
    print_metrics_summary,
)
from mlstudy.trading.strategy.structures.specs.fly import select_fly_legs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run fly backtest with yield-based signals and price-based PnL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to panel CSV with datetime, bond_id, yield, price, dv01, ttm_years",
    )

    # Fly selection
    parser.add_argument(
        "--tenors",
        type=str,
        default="2,5,10",
        help="Target tenors for fly legs (comma-separated)",
    )
    parser.add_argument(
        "--datetime-col",
        type=str,
        default="datetime",
        help="Datetime column name",
    )
    parser.add_argument(
        "--bond-id-col",
        type=str,
        default="bond_id",
        help="Bond ID column name",
    )
    parser.add_argument(
        "--ttm-col",
        type=str,
        default="ttm_years",
        help="Time-to-maturity column name",
    )

    # Signal parameters
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Z-score lookback window",
    )
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Entry threshold (absolute z-score)",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="Exit threshold (absolute z-score)",
    )
    parser.add_argument(
        "--stop-z",
        type=float,
        default=4.0,
        help="Stop-loss threshold (set to 0 to disable)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use robust z-score (median/MAD)",
    )

    # Sizing
    parser.add_argument(
        "--sizing",
        type=str,
        default="fixed_notional",
        choices=["fixed_notional", "dv01_target"],
        help="Position sizing mode",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=1_000_000.0,
        help="Fixed notional per leg (when sizing=fixed_notional)",
    )
    parser.add_argument(
        "--dv01",
        type=float,
        default=10_000.0,
        help="Target DV01 (when sizing=dv01_target)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in basis points",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage cost in basis points",
    )

    # Output
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/backtests",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Load data
    print(f"Loading data from {args.csv}...")
    panel_df = pd.read_csv(args.csv)

    # Parse datetime if string
    if panel_df[args.datetime_col].dtype == "object":
        panel_df[args.datetime_col] = pd.to_datetime(panel_df[args.datetime_col])

    print(f"  Loaded {len(panel_df):,} rows")
    print(f"  Date range: {panel_df[args.datetime_col].min()} to {panel_df[args.datetime_col].max()}")
    print(f"  Unique bonds: {panel_df[args.bond_id_col].nunique()}")

    # Parse tenors
    tenors = tuple(float(t) for t in args.tenors.split(","))
    print(f"\nSelecting fly legs with tenors: {tenors}")

    # Select fly legs
    legs_table = select_fly_legs(
        panel_df,
        tenors=tenors,
        datetime_col=args.datetime_col,
        bond_id_col=args.bond_id_col,
        ttm_col=args.ttm_col,
    )
    print(f"  Selected legs for {len(legs_table)} dates")

    # Configure backtest
    sizing_mode = SizingMode.DV01_TARGET if args.sizing == "dv01_target" else SizingMode.FIXED_NOTIONAL
    config = BacktestConfig(
        sizing_mode=sizing_mode,
        fixed_notional=args.notional,
        dv01_target=args.dv01,
        transaction_cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )

    print("\nBacktest configuration:")
    print(f"  Sizing mode: {config.sizing_mode.value}")
    if sizing_mode == SizingMode.FIXED_NOTIONAL:
        print(f"  Fixed notional: ${config.fixed_notional:,.0f}")
    else:
        print(f"  DV01 target: ${config.dv01_target:,.0f}")
    print(f"  Transaction cost: {config.transaction_cost_bps} bps")
    print(f"  Slippage: {config.slippage_bps} bps")

    print("\nSignal parameters:")
    print(f"  Window: {args.window}")
    print(f"  Entry Z: {args.entry_z}")
    print(f"  Exit Z: {args.exit_z}")
    print(f"  Stop Z: {args.stop_z if args.stop_z > 0 else 'disabled'}")
    print(f"  Robust: {args.robust}")

    # Run backtest
    print("\nRunning backtest...")
    stop_z = args.stop_z if args.stop_z > 0 else None

    result = backtest_fly_from_panel(
        panel_df=panel_df,
        legs_table=legs_table,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=stop_z,
        config=config,
        use_dv01_weights=True,
        robust_zscore=args.robust,
        datetime_col=args.datetime_col,
        bond_id_col=args.bond_id_col,
    )

    # Compute and display metrics
    metrics = compute_metrics(result.pnl_df)
    print_metrics_summary(metrics)

    # Save outputs
    print(f"\nSaving outputs to {args.outdir}...")
    saved = generate_report(
        result,
        output_dir=args.outdir,
        run_id=args.run_id,
        save_plots=not args.no_plots,
    )

    print("\nSaved files:")
    for key, path in saved.items():
        if key not in ["run_id", "output_dir", "metrics"]:
            print(f"  {key}: {path}")

    print(f"\nBacktest complete. Run ID: {saved['run_id']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
