#!/usr/bin/env python
"""Run portfolio backtest across multiple fly strategies.

Pipeline:
1. Load price/yield panel data
2. Build multiple fly strategies with z-score signals
3. Aggregate signals into portfolio-level targets
4. Run backtest with rebalancing and transaction costs
5. Output results and reports

Example:
    python scripts/run_portfolio_backtest.py \
        --data data/panel.parquet \
        --flies 2,5,10 5,10,30 \
        --window 20 \
        --entry-z 2.0 \
        --gross-budget 50000 \
        --cost-bps 1.0 \
        --output outputs/portfolio_backtest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_fly_spec(spec: str) -> tuple[float, float, float]:
    """Parse fly specification like '2,5,10' into tenors."""
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"Fly spec must be 'front,belly,back', got: {spec}")
    return tuple(float(p) for p in parts)


def build_fly_signals(
    panel_df: pd.DataFrame,
    tenors: tuple[float, float, float],
    window: int,
    entry_z: float,
    exit_z: float,
    datetime_col: str = "datetime",
    tenor_col: str = "tenor",
    yield_col: str = "yield",
    bond_id_col: str = "bond_id",
) -> pd.DataFrame:
    """Build z-score signals for a single fly.

    This is a simplified signal builder. In production, use
    mlstudy.strategy.signals for full functionality.
    """
    from mlstudy.trading.strategy import compute_fly_value, select_fly_legs

    # Select legs
    legs = select_fly_legs(panel_df, tenors, tenor_col=tenor_col, bond_id_col=bond_id_col)

    # Compute fly value
    fly_df = compute_fly_value(
        panel_df,
        legs,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        yield_col=yield_col,
    )

    # Compute z-score
    fly_df["zscore"] = (
        fly_df["fly_value"] - fly_df["fly_value"].rolling(window).mean()
    ) / fly_df["fly_value"].rolling(window).std()

    # Generate signal
    fly_df["signal"] = 0
    fly_df.loc[fly_df["zscore"] <= -entry_z, "signal"] = 1  # Cheap -> long
    fly_df.loc[fly_df["zscore"] >= entry_z, "signal"] = -1  # Rich -> short

    # Add leg info
    fly_df["front_id"] = legs["front"]
    fly_df["belly_id"] = legs["belly"]
    fly_df["back_id"] = legs["back"]
    fly_df["tenors"] = str(tenors)

    return fly_df


def main():
    parser = argparse.ArgumentParser(
        description="Run portfolio backtest across multiple fly strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to panel data (parquet or csv)",
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
        "--tenor-col",
        type=str,
        default="tenor",
        help="Tenor column name",
    )
    parser.add_argument(
        "--yield-col",
        type=str,
        default="yield",
        help="Yield column name",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="price",
        help="Price column name",
    )

    # Fly specifications
    parser.add_argument(
        "--flies", "-f",
        type=str,
        nargs="+",
        default=["2,5,10", "5,10,30"],
        help="Fly specs as 'front,belly,back' (e.g., '2,5,10 5,10,30')",
    )

    # Signal parameters
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=20,
        help="Rolling window for z-score (default: 20)",
    )
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Z-score threshold for entry (default: 2.0)",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="Z-score threshold for exit (default: 0.5)",
    )
    parser.add_argument(
        "--target-dv01",
        type=float,
        default=10000.0,
        help="Target gross DV01 per strategy (default: 10000)",
    )

    # Aggregation
    parser.add_argument(
        "--gross-budget",
        type=float,
        default=None,
        help="Gross DV01 budget constraint",
    )
    parser.add_argument(
        "--net-target",
        type=float,
        default=None,
        help="Net DV01 target",
    )
    parser.add_argument(
        "--per-bond-cap",
        type=float,
        default=None,
        help="Per-bond DV01 cap",
    )
    parser.add_argument(
        "--strategy-weights",
        type=str,
        default="equal",
        help="Strategy weights: 'equal' or JSON dict",
    )

    # Backtest
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in bps (default: 1.0)",
    )
    parser.add_argument(
        "--min-trade-dv01",
        type=float,
        default=100.0,
        help="Minimum DV01 to execute a trade (default: 100)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/portfolio_backtest",
        help="Output directory",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from mlstudy.trading.portfolio import (
        PortfolioBacktestConfig,
        RebalanceRule,
        SizingRules,
        StrategySignal,
        aggregate_signal_batch,
        batch_signals_by_timestamp,
        create_fly_legs,
        fly_name_to_strategy_id,
        run_portfolio_backtest_from_targets,
    )

    print(f"Loading data from {args.data}...")
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        panel_df = pd.read_parquet(data_path)
    else:
        panel_df = pd.read_csv(data_path, parse_dates=[args.datetime_col])

    print(f"  Loaded {len(panel_df):,} rows")
    print(f"  Date range: {panel_df[args.datetime_col].min()} to {panel_df[args.datetime_col].max()}")

    # Parse fly specifications
    fly_specs = [parse_fly_spec(spec) for spec in args.flies]
    print(f"\nBuilding signals for {len(fly_specs)} flies: {fly_specs}")

    # Build signals for each fly
    all_signals = []

    for tenors in fly_specs:
        strategy_id = fly_name_to_strategy_id(tenors[0], tenors[1], tenors[2])
        print(f"  Building {strategy_id}...")

        try:
            fly_df = build_fly_signals(
                panel_df,
                tenors=tenors,
                window=args.window,
                entry_z=args.entry_z,
                exit_z=args.exit_z,
                datetime_col=args.datetime_col,
                tenor_col=args.tenor_col,
                yield_col=args.yield_col,
                bond_id_col=args.bond_id_col,
            )

            # Convert to StrategySignal
            for _, row in fly_df.dropna(subset=["signal"]).iterrows():
                if row["signal"] == 0:
                    continue

                legs = create_fly_legs(
                    front_id=row["front_id"],
                    belly_id=row["belly_id"],
                    back_id=row["back_id"],
                    front_tenor=tenors[0],
                    belly_tenor=tenors[1],
                    back_tenor=tenors[2],
                )

                signal = StrategySignal(
                    timestamp=row[args.datetime_col],
                    strategy_id=strategy_id,
                    legs=legs,
                    direction=int(row["signal"]),
                    signal_value=row["zscore"],
                    target_gross_dv01=args.target_dv01,
                )
                all_signals.append(signal)

            print(f"    Generated {len([s for s in all_signals if s.strategy_id == strategy_id])} signals")

        except Exception as e:
            print(f"    Warning: Failed to build {strategy_id}: {e}")
            continue

    if not all_signals:
        print("Error: No signals generated. Check data and parameters.")
        sys.exit(1)

    print(f"\nTotal signals: {len(all_signals)}")

    # Batch signals by timestamp
    batches = batch_signals_by_timestamp(all_signals)
    print(f"Signal batches: {len(batches)}")

    # Parse strategy weights
    if args.strategy_weights == "equal":
        strategy_weights = "equal"
    else:
        try:
            strategy_weights = json.loads(args.strategy_weights)
        except json.JSONDecodeError:
            print(f"Error: Invalid strategy weights JSON: {args.strategy_weights}")
            sys.exit(1)

    # Aggregation rules
    sizing_rules = SizingRules(
        strategy_weights=strategy_weights,
        default_target_dv01=args.target_dv01,
        gross_dv01_budget=args.gross_budget,
        net_dv01_target=args.net_target,
        per_bond_dv01_cap=args.per_bond_cap,
    )

    print("\nAggregating with rules:")
    print(f"  Strategy weights: {strategy_weights}")
    print(f"  Gross DV01 budget: {args.gross_budget}")
    print(f"  Net DV01 target: {args.net_target}")
    print(f"  Per-bond cap: {args.per_bond_cap}")

    # Aggregate each batch
    portfolio_targets = []
    for batch in batches:
        try:
            target = aggregate_signal_batch(batch.signals, sizing_rules)
            portfolio_targets.append(target)
        except ValueError:
            continue

    print(f"Portfolio targets: {len(portfolio_targets)}")

    if not portfolio_targets:
        print("Error: No portfolio targets generated.")
        sys.exit(1)

    # Backtest configuration
    config = PortfolioBacktestConfig(
        cost_bps=args.cost_bps,
        use_dv01_cost_proxy=True,
    )

    rebalance_rule = RebalanceRule(
        min_trade_dv01=args.min_trade_dv01,
    )

    print(f"\nRunning backtest with cost_bps={args.cost_bps}, min_trade_dv01={args.min_trade_dv01}")

    # Run backtest
    result = run_portfolio_backtest_from_targets(
        price_panel=panel_df,
        portfolio_targets=portfolio_targets,
        config=config,
        rebalance_rule=rebalance_rule,
        datetime_col=args.datetime_col,
        bond_id_col=args.bond_id_col,
        price_col=args.price_col,
    )

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PORTFOLIO BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total P&L:        {result.total_pnl:>12,.2f}")
    print(f"Total Cost:       {result.total_cost:>12,.2f}")
    print(f"Total Turnover:   {result.total_turnover:>12,.2f} DV01")
    print(f"Number of Trades: {len(result.trades):>12,}")

    if result.summary:
        print(f"\nSharpe Ratio:     {result.summary.get('sharpe_ratio', 0):>12.2f}")
        print(f"Max Drawdown:     {result.summary.get('max_drawdown', 0):>12,.2f}")
        cost_pct = result.summary.get('cost_pct_of_gross', 0)
        print(f"Cost % of Gross:  {cost_pct:>12.1f}%")

    # Save results
    result.pnl_df.to_parquet(output_dir / "pnl.parquet")
    result.pnl_df.to_csv(output_dir / "pnl.csv", index=False)
    print(f"\nP&L saved to {output_dir / 'pnl.csv'}")

    if result.holdings_df is not None:
        result.holdings_df.to_parquet(output_dir / "holdings.parquet")
        print(f"Holdings saved to {output_dir / 'holdings.parquet'}")

    trades_df = result.trades_to_dataframe()
    if not trades_df.empty:
        trades_df.to_csv(output_dir / "trades.csv", index=False)
        print(f"Trades saved to {output_dir / 'trades.csv'}")

    # Save summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "data_path": str(args.data),
        "flies": args.flies,
        "window": args.window,
        "entry_z": args.entry_z,
        "cost_bps": args.cost_bps,
        "gross_budget": args.gross_budget,
        **result.summary,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {output_dir / 'summary.json'}")

    # Generate plots
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Cumulative P&L
            axes[0].plot(result.pnl_df["timestamp"], result.pnl_df["cumulative_pnl"])
            axes[0].set_title("Cumulative P&L")
            axes[0].set_ylabel("P&L")
            axes[0].grid(True, alpha=0.3)

            # Gross/Net DV01
            axes[1].plot(result.pnl_df["timestamp"], result.pnl_df["gross_dv01"], label="Gross DV01")
            axes[1].plot(result.pnl_df["timestamp"], result.pnl_df["net_dv01"], label="Net DV01")
            axes[1].set_title("DV01 Exposure")
            axes[1].set_ylabel("DV01")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Daily P&L
            axes[2].bar(result.pnl_df["timestamp"], result.pnl_df["net_pnl"], alpha=0.7)
            axes[2].set_title("Daily Net P&L")
            axes[2].set_ylabel("P&L")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "backtest_plots.png", dpi=150)
            print(f"Plots saved to {output_dir / 'backtest_plots.png'}")
            plt.close()

        except ImportError:
            print("Note: matplotlib not available, skipping plots")

    print(f"\nDone! Results in {output_dir}")


if __name__ == "__main__":
    main()
