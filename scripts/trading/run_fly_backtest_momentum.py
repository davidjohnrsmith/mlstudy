#!/usr/bin/env python
"""Run fly backtest with momentum or mean-reversion signals.

Supports both mean-reversion and momentum (trend-following) strategies
with configurable signal types, trend filters, and execution modes.

Trading constraints:
- Intraday session trading (07:30-17:00 Europe/Berlin default)
- Positions can be held overnight
- Signal at bar t executes at bar t+1 (no lookahead)

Usage:
    python scripts/run_fly_backtest_momentum.py --csv data/panel.csv --signal momentum

Examples:
    # Momentum with time-series momentum signal
    python scripts/run_fly_backtest_momentum.py \
        --csv data/intraday_panel.csv \
        --signal momentum \
        --mom-kind tsmom \
        --lookback-bars 20 \
        --trend-filter on \
        --trend-window 60 \
        --rebalance every_bar

    # Mean-reversion signal
    python scripts/run_fly_backtest_momentum.py \
        --csv data/intraday_panel.csv \
        --signal mr \
        --z-window-bars 20 \
        --entry-z 2.0 \
        --exit-z 0.5

    # Momentum with EMA crossover and trend filter
    python scripts/run_fly_backtest_momentum.py \
        --csv data/intraday_panel.csv \
        --signal momentum \
        --mom-kind ema \
        --fast-span 5 \
        --slow-span 20 \
        --trend-filter on \
        --trend-threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlstudy.trading.backtest import (
    IntradayBacktestConfig,
    SizingMode,
    print_metrics_summary,
)
from mlstudy.trading.backtest.metrics.metrics import compute_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run fly backtest with momentum or mean-reversion signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to panel CSV with datetime, bond_id, yield, price, dv01, ttm_years",
    )

    # Signal type
    parser.add_argument(
        "--signal",
        type=str,
        default="mr",
        choices=["mr", "momentum"],
        help="Signal type: 'mr' (mean-reversion) or 'momentum' (trend-following)",
    )

    # Fly configuration
    parser.add_argument(
        "--tenors",
        type=str,
        default="2,5,10",
        help="Target tenors for fly (comma-separated: front,belly,back)",
    )

    # Mean-reversion signal configuration
    mr_group = parser.add_argument_group("Mean-reversion signal options")
    mr_group.add_argument(
        "--z-window-bars",
        type=int,
        default=20,
        help="Z-score lookback window in bars",
    )
    mr_group.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Entry z-score threshold",
    )
    mr_group.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="Exit z-score threshold",
    )
    mr_group.add_argument(
        "--stop-z",
        type=float,
        default=None,
        help="Stop-loss z-score threshold (None to disable)",
    )
    mr_group.add_argument(
        "--robust-zscore",
        action="store_true",
        help="Use robust z-score (median/MAD) instead of mean/std",
    )

    # Momentum signal configuration
    mom_group = parser.add_argument_group("Momentum signal options")
    mom_group.add_argument(
        "--mom-kind",
        type=str,
        default="tsmom",
        choices=["tsmom", "ema", "breakout", "combined"],
        help="Momentum signal type",
    )
    mom_group.add_argument(
        "--lookback-bars",
        type=int,
        default=20,
        help="Lookback period for time-series momentum",
    )
    mom_group.add_argument(
        "--fast-span",
        type=int,
        default=5,
        help="Fast EMA span for crossover signal",
    )
    mom_group.add_argument(
        "--slow-span",
        type=int,
        default=20,
        help="Slow EMA span for crossover signal",
    )
    mom_group.add_argument(
        "--breakout-window",
        type=int,
        default=20,
        help="Window for breakout detection",
    )
    mom_group.add_argument(
        "--mom-threshold",
        type=float,
        default=0.0,
        help="Threshold for converting momentum score to position",
    )

    # Trend filter configuration
    trend_group = parser.add_argument_group("Trend filter options")
    trend_group.add_argument(
        "--trend-filter",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable trend strength filter",
    )
    trend_group.add_argument(
        "--trend-window",
        type=int,
        default=60,
        help="Window for trend strength calculation",
    )
    trend_group.add_argument(
        "--trend-threshold",
        type=float,
        default=0.2,
        help="Minimum trend strength (R²) to allow position",
    )

    # Session configuration
    session_group = parser.add_argument_group("Session options")
    session_group.add_argument(
        "--session-start",
        type=str,
        default="07:30",
        help="Session start time (HH:MM)",
    )
    session_group.add_argument(
        "--session-end",
        type=str,
        default="17:00",
        help="Session end time (HH:MM)",
    )
    session_group.add_argument(
        "--tz",
        type=str,
        default="Europe/Berlin",
        help="Timezone for session times",
    )
    session_group.add_argument(
        "--selection-time",
        type=str,
        default="07:30",
        help="Time to select fly legs each day (HH:MM)",
    )

    # Sizing configuration
    sizing_group = parser.add_argument_group("Sizing options")
    sizing_group.add_argument(
        "--sizing",
        type=str,
        default="dv01_target",
        choices=["fixed_notional", "dv01_target"],
        help="Position sizing mode",
    )
    sizing_group.add_argument(
        "--notional",
        type=float,
        default=1_000_000.0,
        help="Fixed notional per leg (for fixed_notional mode)",
    )
    sizing_group.add_argument(
        "--dv01-target",
        type=float,
        default=10_000.0,
        help="Target gross DV01 (for dv01_target mode)",
    )

    # Execution configuration
    exec_group = parser.add_argument_group("Execution options")
    exec_group.add_argument(
        "--rebalance",
        type=str,
        default="every_bar",
        choices=["open_only", "every_bar", "every_n_bars", "close_only"],
        help="When to allow position changes",
    )
    exec_group.add_argument(
        "--rebalance-n-bars",
        type=int,
        default=1,
        help="Number of bars between rebalances (for every_n_bars mode)",
    )
    exec_group.add_argument(
        "--signal-lag",
        type=int,
        default=1,
        help="Signal lag: position[t] = f(score[t-lag])",
    )

    # Cost configuration
    cost_group = parser.add_argument_group("Cost options")
    cost_group.add_argument(
        "--cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in bps",
    )
    cost_group.add_argument(
        "--slippage-bps",
        type=float,
        default=0.5,
        help="Slippage cost in bps",
    )

    # Column names
    col_group = parser.add_argument_group("Column name options")
    col_group.add_argument("--datetime-col", type=str, default="datetime")
    col_group.add_argument("--bond-id-col", type=str, default="bond_id")
    col_group.add_argument("--ttm-col", type=str, default="ttm_years")
    col_group.add_argument("--yield-col", type=str, default="yield")
    col_group.add_argument("--price-col", type=str, default="price")
    col_group.add_argument("--dv01-col", type=str, default="dv01")

    # Output configuration
    out_group = parser.add_argument_group("Output options")
    out_group.add_argument(
        "--outdir",
        type=str,
        default="outputs/backtests/momentum",
        help="Output directory for results",
    )
    out_group.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (auto-generated if not provided)",
    )
    out_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args(argv)


def generate_momentum_signal(
    fly_yield: pd.Series,
    kind: str,
    lookback_bars: int,
    fast_span: int,
    slow_span: int,
    breakout_window: int,
    threshold: float,
    trend_filter: bool,
    trend_window: int,
    trend_threshold: float,
) -> pd.Series:
    """Generate momentum signal from fly yield.

    Args:
        fly_yield: Fly yield series.
        kind: Signal type (tsmom, ema, breakout, combined).
        lookback_bars: Lookback for time-series momentum.
        fast_span: Fast EMA span.
        slow_span: Slow EMA span.
        breakout_window: Breakout detection window.
        threshold: Position threshold.
        trend_filter: Whether to apply trend filter.
        trend_window: Trend calculation window.
        trend_threshold: Minimum trend strength.

    Returns:
        Signal series (-1, 0, 1).
    """
    from mlstudy.trading.strategy.alpha.momentum.momentum import (
        breakout_signal,
        combine_momentum_signals,
        ema_crossover_signal,
        signal_to_position,
        trend_strength,
        ts_momentum_signal,
    )

    # Generate raw score based on signal kind
    if kind == "tsmom":
        score = ts_momentum_signal(fly_yield, lookback_bars=lookback_bars)
        # Normalize for position conversion
        score_std = score.rolling(window=slow_span, min_periods=1).std()
        score = score / score_std.replace(0, np.nan)

    elif kind == "ema":
        score = ema_crossover_signal(fly_yield, fast_span=fast_span, slow_span=slow_span)

    elif kind == "breakout":
        score = breakout_signal(fly_yield, window_bars=breakout_window).astype(float)

    elif kind == "combined":
        ts_mom = ts_momentum_signal(fly_yield, lookback_bars=lookback_bars)
        ts_mom_std = ts_mom.rolling(window=slow_span, min_periods=1).std()
        ts_mom_norm = ts_mom / ts_mom_std.replace(0, np.nan)

        ema_xover = ema_crossover_signal(fly_yield, fast_span=fast_span, slow_span=slow_span)
        breakout = breakout_signal(fly_yield, window_bars=breakout_window).astype(float)

        signals = {
            "ts_momentum": ts_mom_norm,
            "ema_crossover": ema_xover,
            "breakout": breakout,
        }
        score = combine_momentum_signals(signals, method="weighted_avg")

    else:
        raise ValueError(f"Unknown momentum kind: {kind}")

    # Convert to position
    position = signal_to_position(score, threshold=threshold)

    # Apply trend filter if enabled
    if trend_filter:
        trend = trend_strength(fly_yield, window_bars=trend_window, method="r2")
        # Only allow positions when trend strength is above threshold
        weak_trend = abs(trend) < trend_threshold
        position = position.copy()
        position[weak_trend] = 0

    return position


def generate_mr_signal(
    fly_yield: pd.Series,
    window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float | None,
    robust: bool,
) -> pd.Series:
    """Generate mean-reversion signal from fly yield.

    Args:
        fly_yield: Fly yield series.
        window: Z-score lookback window.
        entry_z: Entry threshold.
        exit_z: Exit threshold.
        stop_z: Stop-loss threshold.
        robust: Use robust z-score.

    Returns:
        Signal series (-1, 0, 1).
    """
    from mlstudy.trading.strategy.alpha.mean_reversion.signals import generate_mean_reversion_signal, rolling_zscore

    zscore = rolling_zscore(fly_yield, window=window, robust=robust, clip=4.0)
    signal = generate_mean_reversion_signal(zscore, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)

    return signal


def run_backtest_with_signal(
    panel_df: pd.DataFrame,
    signal: pd.Series,
    tenors: tuple[float, float, float],
    config: IntradayBacktestConfig,
    signal_lag: int,
    datetime_col: str,
    bond_id_col: str,
    ttm_col: str,
    yield_col: str,
    price_col: str,
    dv01_col: str,
    verbose: bool,
) -> dict:
    """Run backtest with pre-computed signal.

    Args:
        panel_df: Panel data.
        signal: Signal series.
        tenors: Fly tenors.
        config: Backtest config.
        signal_lag: Signal execution lag.
        datetime_col, bond_id_col, etc.: Column names.
        verbose: Print progress.

    Returns:
        Dict with pnl_df, metrics, etc.
    """
    from mlstudy.trading.strategy.structures.selection.curve_selection import select_fly_legs
    from mlstudy.trading.strategy.structures.specs.fly.old.fly import build_fly

    # Select fly legs
    legs_table = select_fly_legs(
        panel_df,
        target_tenors=tenors,
        datetime_col=datetime_col,
        ttm_col=ttm_col,
    )

    if verbose:
        print(f"  Selected fly legs for {len(legs_table[datetime_col].unique())} dates")

    # Build fly
    fly_result = build_fly(
        df=panel_df,
        legs_table=legs_table,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        value_cols=[yield_col, price_col, dv01_col],
        use_dv01_weights=True,
        dv01_col=dv01_col,
        yield_col=yield_col,
    )

    fly_df = fly_result.fly_df
    legs_df = fly_result.legs_df

    # Align signal with fly_df
    fly_df = fly_df.set_index(datetime_col)
    signal_aligned = signal.reindex(fly_df.index)

    # Apply signal lag: position[t] = signal[t - lag]
    position = signal_aligned.shift(signal_lag).fillna(0).astype(int)

    # Compute P&L
    fly_df["position"] = position.values
    fly_df["fly_yield_change"] = fly_df["fly_yield"].diff()

    # Gross return from fly yield changes
    fly_df["gross_return"] = fly_df["position"].shift(1).fillna(0) * fly_df["fly_yield_change"]

    # Track position changes for costs
    fly_df["position_change"] = fly_df["position"].diff().abs().fillna(0)
    total_cost_bps = config.transaction_cost_bps + config.slippage_bps

    # Apply DV01 sizing
    if config.sizing_mode == SizingMode.DV01_TARGET:
        # Scale returns by DV01 target
        if "net_dv01" in fly_df.columns:
            scale = config.dv01_target / fly_df["net_dv01"].abs().replace(0, np.nan).fillna(1)
        else:
            scale = config.dv01_target
        fly_df["gross_return"] = fly_df["gross_return"] * scale

    # Transaction costs
    fly_df["transaction_cost"] = fly_df["position_change"] * total_cost_bps / 10000

    # Net return
    fly_df["net_return"] = fly_df["gross_return"] - fly_df["transaction_cost"]
    fly_df["cumulative_pnl"] = fly_df["net_return"].cumsum()

    # Reset index
    pnl_df = fly_df.reset_index()

    # Compute metrics
    metrics = compute_metrics(pnl_df)

    return {
        "pnl_df": pnl_df,
        "legs_df": legs_df,
        "legs_table": legs_table,
        "metrics": metrics,
        "fly_result": fly_result,
    }


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
        print(f"Signal type: {args.signal}")

    # Build fly and get fly yield
    from mlstudy.trading.strategy.structures.selection.curve_selection import select_fly_legs
    from mlstudy.trading.strategy.structures.specs.fly.old.fly import build_fly

    legs_table = select_fly_legs(
        panel_df,
        target_tenors=tenors,
        datetime_col=args.datetime_col,
        ttm_col=args.ttm_col,
    )

    fly_result = build_fly(
        df=panel_df,
        legs_table=legs_table,
        datetime_col=args.datetime_col,
        bond_id_col=args.bond_id_col,
        value_cols=[args.yield_col, args.price_col, args.dv01_col],
        use_dv01_weights=True,
        dv01_col=args.dv01_col,
        yield_col=args.yield_col,
    )

    fly_yield = fly_result.get_fly_yield()

    if not args.quiet:
        print(f"  Fly yield computed: {len(fly_yield)} observations")

    # Generate signal
    if not args.quiet:
        print("\nGenerating signal...")

    if args.signal == "mr":
        signal = generate_mr_signal(
            fly_yield=fly_yield,
            window=args.z_window_bars,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            robust=args.robust_zscore,
        )
        if not args.quiet:
            print(f"  Mean-reversion signal (window={args.z_window_bars}, entry={args.entry_z}, exit={args.exit_z})")

    else:  # momentum
        signal = generate_momentum_signal(
            fly_yield=fly_yield,
            kind=args.mom_kind,
            lookback_bars=args.lookback_bars,
            fast_span=args.fast_span,
            slow_span=args.slow_span,
            breakout_window=args.breakout_window,
            threshold=args.mom_threshold,
            trend_filter=args.trend_filter == "on",
            trend_window=args.trend_window,
            trend_threshold=args.trend_threshold,
        )
        if not args.quiet:
            print(f"  Momentum signal ({args.mom_kind})")
            if args.trend_filter == "on":
                print(f"  Trend filter: window={args.trend_window}, threshold={args.trend_threshold}")

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
    )

    if not args.quiet:
        print("\nExecution settings:")
        print(f"  Signal lag: {args.signal_lag} bars")
        print(f"  Rebalancing: {args.rebalance}")
        print(f"  Sizing: {args.sizing} ({args.dv01_target if args.sizing == 'dv01_target' else args.notional})")

    # Run backtest
    if not args.quiet:
        print("\nRunning backtest...")

    try:
        result = run_backtest_with_signal(
            panel_df=panel_df,
            signal=signal,
            tenors=tenors,
            config=config,
            signal_lag=args.signal_lag,
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
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    if not args.quiet:
        print(f"\nSaving results to {output_path}...")

    # Save P&L
    pnl_path = output_path / "pnl.csv"
    result["pnl_df"].to_csv(pnl_path, index=False)
    if not args.quiet:
        print(f"  P&L: {pnl_path}")

    # Save legs table
    legs_path = output_path / "legs_table.csv"
    result["legs_table"].to_csv(legs_path, index=False)
    if not args.quiet:
        print(f"  Legs table: {legs_path}")

    # Save metrics
    metrics_dict = result["metrics"].to_dict()
    # Handle numpy types
    clean_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (np.integer, np.int64)):
            clean_metrics[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            if np.isnan(v) or np.isinf(v):
                clean_metrics[k] = None
            else:
                clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    if not args.quiet:
        print(f"  Metrics: {metrics_path}")

    # Save config
    config_dict = {
        "signal": args.signal,
        "tenors": tenors,
        "signal_lag": args.signal_lag,
    }
    if args.signal == "mr":
        config_dict.update({
            "z_window_bars": args.z_window_bars,
            "entry_z": args.entry_z,
            "exit_z": args.exit_z,
            "stop_z": args.stop_z,
            "robust_zscore": args.robust_zscore,
        })
    else:
        config_dict.update({
            "mom_kind": args.mom_kind,
            "lookback_bars": args.lookback_bars,
            "fast_span": args.fast_span,
            "slow_span": args.slow_span,
            "breakout_window": args.breakout_window,
            "mom_threshold": args.mom_threshold,
            "trend_filter": args.trend_filter,
            "trend_window": args.trend_window,
            "trend_threshold": args.trend_threshold,
        })
    config_dict.update({
        "sizing": args.sizing,
        "notional": args.notional,
        "dv01_target": args.dv01_target,
        "rebalance": args.rebalance,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
    })

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    if not args.quiet:
        print(f"  Config: {config_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print_metrics_summary(result["metrics"])

    # Signal statistics
    pnl_df = result["pnl_df"]
    print("\nSignal Stats:")
    print(f"  Total bars: {len(pnl_df):,}")
    n_long = (pnl_df["position"] == 1).sum()
    n_short = (pnl_df["position"] == -1).sum()
    n_flat = (pnl_df["position"] == 0).sum()
    print(f"  Long: {n_long:,} ({100*n_long/len(pnl_df):.1f}%)")
    print(f"  Short: {n_short:,} ({100*n_short/len(pnl_df):.1f}%)")
    print(f"  Flat: {n_flat:,} ({100*n_flat/len(pnl_df):.1f}%)")

    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
