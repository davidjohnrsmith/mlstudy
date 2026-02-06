#!/usr/bin/env python
"""Sweep fly backtests across multiple flies and parameters.

Generates all valid fly combinations and runs backtests with parameter sweep.
Outputs summary tables sorted by various metrics.

Usage:
    python scripts/sweep_fly_backtests.py --csv data/panel.csv
    python scripts/sweep_fly_backtests.py --csv data/panel.csv --tenors 2,5,10,30

Example:
    python scripts/sweep_fly_backtests.py \
        --csv data/bond_panel.csv \
        --tenors 2,3,5,7,10,30 \
        --windows 10,20,30 \
        --entry-zs 1.5,2.0,2.5 \
        --exit-zs 0.3,0.5,0.7 \
        --outdir outputs/sweeps
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from mlstudy.trading.backtest import BacktestConfig, SizingMode
from mlstudy.trading.strategy.structures.specs.fly import (
    ParamGrid,
    build_and_backtest_many_flies,
    filter_valid_flies,
    generate_flies_from_tenors,
    get_best_fly_params,
    summarize_by_fly,
    summarize_by_params,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sweep fly backtests across multiple flies and parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to panel CSV with datetime, bond_id, yield, price, dv01, ttm_years",
    )

    # Fly universe
    parser.add_argument(
        "--tenors",
        type=str,
        default="2,3,5,7,10,30",
        help="Tenors to generate flies from (comma-separated)",
    )
    parser.add_argument(
        "--min-wing-spread",
        type=float,
        default=1.0,
        help="Minimum tenor spread between adjacent legs",
    )
    parser.add_argument(
        "--symmetric-only",
        action="store_true",
        help="Only generate symmetric flies (equal wing spreads)",
    )

    # Parameter grid
    parser.add_argument(
        "--windows",
        type=str,
        default="20",
        help="Z-score windows to test (comma-separated)",
    )
    parser.add_argument(
        "--entry-zs",
        type=str,
        default="2.0",
        help="Entry z thresholds to test (comma-separated)",
    )
    parser.add_argument(
        "--exit-zs",
        type=str,
        default="0.5",
        help="Exit z thresholds to test (comma-separated)",
    )
    parser.add_argument(
        "--stop-zs",
        type=str,
        default="4.0",
        help="Stop z thresholds to test (comma-separated, use 'none' to disable)",
    )
    parser.add_argument(
        "--test-robust",
        action="store_true",
        help="Include robust z-score in sweep",
    )

    # Sizing
    parser.add_argument(
        "--sizing",
        type=str,
        default="fixed_notional",
        choices=["fixed_notional", "dv01_target", "both"],
        help="Sizing mode(s) to test",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=1_000_000.0,
        help="Fixed notional for sizing",
    )
    parser.add_argument(
        "--dv01-target",
        type=float,
        default=10_000.0,
        help="DV01 target for sizing",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in bps",
    )

    # Column names
    parser.add_argument("--datetime-col", type=str, default="datetime")
    parser.add_argument("--bond-id-col", type=str, default="bond_id")
    parser.add_argument("--ttm-col", type=str, default="ttm_years")

    # Output
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/sweeps",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (auto-generated if not provided)",
    )

    # Filters
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=None,
        help="Filter: minimum Sharpe ratio",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help="Filter: maximum drawdown (negative, e.g., -0.1)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help="Filter: minimum number of trades",
    )

    return parser.parse_args(argv)


def parse_list(s: str, dtype: type = float) -> list:
    """Parse comma-separated string to list."""
    items = []
    for item in s.split(","):
        item = item.strip()
        if item.lower() == "none":
            items.append(None)
        else:
            items.append(dtype(item))
    return items


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Generate run ID
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.outdir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.csv}...")
    panel_df = pd.read_csv(args.csv)
    if panel_df[args.datetime_col].dtype == "object":
        panel_df[args.datetime_col] = pd.to_datetime(panel_df[args.datetime_col])

    print(f"  Loaded {len(panel_df):,} rows")
    print(f"  Date range: {panel_df[args.datetime_col].min()} to {panel_df[args.datetime_col].max()}")

    # Generate fly universe
    tenors = parse_list(args.tenors, float)
    print(f"\nGenerating fly universe from tenors: {tenors}")

    flies = generate_flies_from_tenors(
        tenors=tenors,
        min_wing_spread=args.min_wing_spread,
        require_symmetric=args.symmetric_only,
    )
    print(f"  Generated {len(flies)} valid flies")

    # Build parameter grid
    windows = parse_list(args.windows, int)
    entry_zs = parse_list(args.entry_zs, float)
    exit_zs = parse_list(args.exit_zs, float)
    stop_zs = parse_list(args.stop_zs, float)

    sizing_modes = []
    if args.sizing in ["fixed_notional", "both"]:
        sizing_modes.append(SizingMode.FIXED_NOTIONAL)
    if args.sizing in ["dv01_target", "both"]:
        sizing_modes.append(SizingMode.DV01_TARGET)

    robust_zscores = [False]
    if args.test_robust:
        robust_zscores = [False, True]

    param_grid = ParamGrid(
        windows=windows,
        entry_zs=entry_zs,
        exit_zs=exit_zs,
        stop_zs=stop_zs,
        sizing_modes=sizing_modes,
        robust_zscores=robust_zscores,
    )

    print(f"\nParameter grid: {len(param_grid)} combinations")
    print(f"  Windows: {windows}")
    print(f"  Entry Zs: {entry_zs}")
    print(f"  Exit Zs: {exit_zs}")
    print(f"  Stop Zs: {stop_zs}")
    print(f"  Sizing modes: {[m.value for m in sizing_modes]}")
    print(f"  Robust zscore: {robust_zscores}")

    total_runs = len(flies) * len(param_grid)
    print(f"\nTotal backtests to run: {total_runs}")

    # Base config
    config = BacktestConfig(
        fixed_notional=args.notional,
        dv01_target=args.dv01_target,
        transaction_cost_bps=args.cost_bps,
    )

    # Run sweep
    print("\nRunning sweep...")
    sweep_df = build_and_backtest_many_flies(
        panel_df=panel_df,
        flies=flies,
        param_grid=param_grid,
        config=config,
        datetime_col=args.datetime_col,
        bond_id_col=args.bond_id_col,
        ttm_col=args.ttm_col,
        verbose=True,
    )

    if len(sweep_df) == 0:
        print("\nNo successful backtests. Check data and parameters.")
        return 1

    print(f"\nCompleted {len(sweep_df)} successful backtests")

    # Apply filters
    if args.min_sharpe or args.max_drawdown or args.min_trades:
        filtered_df = filter_valid_flies(
            sweep_df,
            min_sharpe=args.min_sharpe,
            max_drawdown=args.max_drawdown,
            min_trades=args.min_trades,
        )
        print(f"  After filters: {len(filtered_df)} results")
    else:
        filtered_df = sweep_df

    # Save full results
    full_path = output_path / "full_results.csv"
    # Drop result column if present (not serializable)
    save_cols = [c for c in sweep_df.columns if c != "result"]
    sweep_df[save_cols].to_csv(full_path, index=False)
    print(f"\nSaved full results to {full_path}")

    # Save sorted summaries
    print("\nGenerating summary tables...")

    # By Sharpe
    sorted_sharpe = filtered_df.sort_values("sharpe_ratio", ascending=False)
    sharpe_path = output_path / "summary_by_sharpe.csv"
    sorted_sharpe[save_cols].head(100).to_csv(sharpe_path, index=False)
    print(f"  Top by Sharpe: {sharpe_path}")

    # By drawdown (least negative = best)
    sorted_dd = filtered_df.sort_values("max_drawdown", ascending=False)
    dd_path = output_path / "summary_by_drawdown.csv"
    sorted_dd[save_cols].head(100).to_csv(dd_path, index=False)
    print(f"  Top by Drawdown: {dd_path}")

    # By turnover (lowest = best)
    if "turnover_annual" in filtered_df.columns:
        sorted_turnover = filtered_df.sort_values("turnover_annual", ascending=True)
        turnover_path = output_path / "summary_by_turnover.csv"
        sorted_turnover[save_cols].head(100).to_csv(turnover_path, index=False)
        print(f"  Top by Turnover: {turnover_path}")

    # Summary by fly
    fly_summary = summarize_by_fly(filtered_df, metric="sharpe_ratio", agg="mean")
    fly_path = output_path / "summary_by_fly.csv"
    fly_summary.to_csv(fly_path, index=False)
    print(f"  By Fly: {fly_path}")

    # Summary by params
    param_summary = summarize_by_params(filtered_df, metric="sharpe_ratio", agg="mean")
    param_path = output_path / "summary_by_params.csv"
    param_summary.to_csv(param_path, index=False)
    print(f"  By Params: {param_path}")

    # Best overall
    best = get_best_fly_params(filtered_df, metric="sharpe_ratio")
    best_dict = best.to_dict()
    # Clean non-serializable
    best_dict = {k: v for k, v in best_dict.items() if k != "result"}
    best_path = output_path / "best_result.json"
    with open(best_path, "w") as f:
        json.dump(best_dict, f, indent=2, default=str)
    print(f"  Best Result: {best_path}")

    # Print top 5
    print("\n" + "=" * 70)
    print("TOP 5 BY SHARPE RATIO")
    print("=" * 70)
    top5_cols = ["fly_name", "window", "entry_z", "exit_z", "sharpe_ratio", "total_pnl", "max_drawdown", "n_trades"]
    top5_cols = [c for c in top5_cols if c in sorted_sharpe.columns]
    print(sorted_sharpe[top5_cols].head(5).to_string(index=False))

    print(f"\nSweep complete. Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
