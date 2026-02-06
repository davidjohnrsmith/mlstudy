#!/usr/bin/env python
"""Run intraday fly backtest with session-aware execution.

Supports:
- Trading session hours with timezone
- Daily leg selection at specified time
- Multiple rebalancing modes
- DV01 target or fixed notional sizing

Usage:
    python scripts/run_fly_backtest_intraday.py --csv data/intraday_panel.csv

Example:
    python scripts/run_fly_backtest_intraday.py \
        --csv data/intraday_panel.csv \
        --tenors 2,5,10 \
        --session-start 07:30 \
        --session-end 17:00 \
        --tz Europe/Berlin \
        --selection-time 07:30 \
        --z-window-bars 20 \
        --entry-z 2.0 \
        --exit-z 0.5 \
        --stop-z 4.0 \
        --sizing dv01_target \
        --dv01-target 10000 \
        --rebalance open_only \
        --outdir outputs/backtests/intraday
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from mlstudy.trading.backtest import (
    IntradayBacktestConfig,
    SizingMode,
    backtest_fly_intraday,
    print_metrics_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run intraday fly backtest with session-aware execution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to intraday panel CSV with datetime, bond_id, yield, price, dv01, ttm_years",
    )

    # Fly configuration
    parser.add_argument(
        "--tenors",
        type=str,
        default="2,5,10",
        help="Target tenors for fly (comma-separated: front,belly,back)",
    )

    # Session configuration
    parser.add_argument(
        "--session-start",
        type=str,
        default="07:30",
        help="Session start time (HH:MM)",
    )
    parser.add_argument(
        "--session-end",
        type=str,
        default="17:00",
        help="Session end time (HH:MM)",
    )
    parser.add_argument(
        "--tz",
        type=str,
        default="Europe/Berlin",
        help="Timezone for session times",
    )
    parser.add_argument(
        "--selection-time",
        type=str,
        default="07:30",
        help="Time to select fly legs each day (HH:MM)",
    )

    # Signal configuration
    parser.add_argument(
        "--z-window-bars",
        type=int,
        default=20,
        help="Z-score lookback window in bars",
    )
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Entry z-score threshold",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="Exit z-score threshold",
    )
    parser.add_argument(
        "--stop-z",
        type=float,
        default=None,
        help="Stop-loss z-score threshold (None to disable)",
    )
    parser.add_argument(
        "--robust-zscore",
        action="store_true",
        help="Use robust z-score (median/MAD) instead of mean/std",
    )

    # Sizing configuration
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
        help="Fixed notional per leg (for fixed_notional mode)",
    )
    parser.add_argument(
        "--dv01-target",
        type=float,
        default=10_000.0,
        help="Target gross DV01 (for dv01_target mode)",
    )

    # Execution configuration
    parser.add_argument(
        "--rebalance",
        type=str,
        default="every_bar",
        choices=["open_only", "every_bar", "every_n_bars", "close_only"],
        help="When to allow position changes",
    )
    parser.add_argument(
        "--rebalance-n-bars",
        type=int,
        default=1,
        help="Number of bars between rebalances (for every_n_bars mode)",
    )
    parser.add_argument(
        "--allow-overnight",
        action="store_true",
        default=True,
        help="Allow positions to persist overnight",
    )

    # Cost configuration
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in bps",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage cost in bps",
    )

    # Column names
    parser.add_argument("--datetime-col", type=str, default="datetime")
    parser.add_argument("--bond-id-col", type=str, default="bond_id")
    parser.add_argument("--ttm-col", type=str, default="ttm_years")
    parser.add_argument("--yield-col", type=str, default="yield")
    parser.add_argument("--price-col", type=str, default="price")
    parser.add_argument("--dv01-col", type=str, default="dv01")

    # Output configuration
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/backtests/intraday",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Generate run ID
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.outdir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    if not args.quiet:
        print(f"Loading data from {args.csv}...")

    panel_df = pd.read_csv(args.csv)
    if panel_df[args.datetime_col].dtype == "object":
        panel_df[args.datetime_col] = pd.to_datetime(panel_df[args.datetime_col])

    if not args.quiet:
        print(f"  Loaded {len(panel_df):,} rows")
        print(f"  Date range: {panel_df[args.datetime_col].min()} to {panel_df[args.datetime_col].max()}")

    # Parse tenors
    tenors = tuple(float(t.strip()) for t in args.tenors.split(","))
    if len(tenors) != 3:
        print("Error: --tenors must have exactly 3 values (front,belly,back)")
        return 1

    if not args.quiet:
        print(f"\nFly configuration: {tenors[0]}y/{tenors[1]}y/{tenors[2]}y")

    # Build config
    sizing_mode = (
        SizingMode.DV01_TARGET if args.sizing == "dv01_target"
        else SizingMode.FIXED_NOTIONAL
    )

    config = IntradayBacktestConfig(
        sizing_mode=sizing_mode,
        fixed_notional=args.notional,
        dv01_target=args.dv01_target,
        transaction_cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        session_start=args.session_start,
        session_end=args.session_end,
        tz=args.tz,
        selection_time=args.selection_time,
        rebalance_mode=args.rebalance,
        rebalance_n_bars=args.rebalance_n_bars,
        allow_overnight=args.allow_overnight,
    )

    if not args.quiet:
        print(f"\nSession: {args.session_start} - {args.session_end} ({args.tz})")
        print(f"Leg selection time: {args.selection_time}")
        print(f"Rebalancing: {args.rebalance}")
        print(f"Sizing: {args.sizing} ({args.dv01_target if args.sizing == 'dv01_target' else args.notional})")

    # Run backtest
    if not args.quiet:
        print("\nRunning intraday backtest...")

    try:
        result = backtest_fly_intraday(
            panel_df=panel_df,
            tenors=tenors,
            window=args.z_window_bars,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            config=config,
            robust_zscore=args.robust_zscore,
            datetime_col=args.datetime_col,
            bond_id_col=args.bond_id_col,
            ttm_col=args.ttm_col,
            yield_col=args.yield_col,
            price_col=args.price_col,
            dv01_col=args.dv01_col,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error running backtest: {e}")
        return 1

    # Save results
    if not args.quiet:
        print(f"\nSaving results to {output_path}...")

    # Save bar-level PnL
    pnl_path = output_path / "pnl_bars.csv"
    result.pnl_df.to_csv(pnl_path, index=False)
    if not args.quiet:
        print(f"  Bar-level P&L: {pnl_path}")

    # Save daily PnL
    daily_path = output_path / "pnl_daily.csv"
    result.daily_df.to_csv(daily_path, index=False)
    if not args.quiet:
        print(f"  Daily P&L: {daily_path}")

    # Save legs table
    legs_path = output_path / "legs_table.csv"
    result.legs_table.to_csv(legs_path, index=False)
    if not args.quiet:
        print(f"  Legs table: {legs_path}")

    # Save metrics
    metrics_dict = result.metrics.to_dict()
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    if not args.quiet:
        print(f"  Metrics: {metrics_path}")

    # Save config
    config_dict = {
        "tenors": tenors,
        "window": args.z_window_bars,
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "stop_z": args.stop_z,
        "robust_zscore": args.robust_zscore,
        "sizing": args.sizing,
        "notional": args.notional,
        "dv01_target": args.dv01_target,
        "session_start": args.session_start,
        "session_end": args.session_end,
        "tz": args.tz,
        "selection_time": args.selection_time,
        "rebalance": args.rebalance,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
    }
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    if not args.quiet:
        print(f"  Config: {config_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print_metrics_summary(result.metrics)

    # Additional intraday stats
    print("\nIntraday Stats:")
    print(f"  Total bars: {len(result.pnl_df):,}")
    print(f"  Trading days: {len(result.daily_df):,}")
    print(f"  Session bars: {result.pnl_df['is_session'].sum():,}")

    if "position" in result.pnl_df.columns:
        in_position = (result.pnl_df["position"] != 0).sum()
        print(f"  Bars in position: {in_position:,} ({100*in_position/len(result.pnl_df):.1f}%)")

    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
