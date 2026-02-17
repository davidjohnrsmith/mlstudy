#!/usr/bin/env python
"""End-to-end launch script for the mean-reversion L2-book backtester.

Supports three modes:

1. **YAML config sweep** — point at a config YAML (with ``data`` section for
   filenames/instruments) and pass ``--data-path`` at launch time::

    python scripts/trading/run_mr_backtest.py \\
        --config configs/mr_grid_v1.yaml \\
        --data-path /mnt/data/20240101          # Linux
        --data-path D:\\data\\20240101            # Windows

2. **Single run** — run a single backtest against parquet data::

    python scripts/trading/run_mr_backtest.py \\
        --data-path data/20240101 \\
        --instruments UST_2Y,UST_5Y,UST_10Y \\
        --ref-instrument UST_5Y \\
        --entry-z 2.0

3. **Single run from YAML** — use a config YAML but skip the grid (base_config only)::

    python scripts/trading/run_mr_backtest.py \\
        --config configs/mr_grid_v1.yaml --single \\
        --data-path data/20240101

The YAML config stores platform-independent settings (filenames, instruments,
grid, thresholds).  The ``--data-path`` is always supplied at runtime so the
same config works on Linux and Windows without modification.

Example (full sweep)::

    python scripts/trading/run_mr_backtest.py \\
        --config configs/mr_grid_v1.yaml \\
        --data-path data/20240101 \\
        --outdir runs/sweep_20240101

Example (single run with overrides)::

    python scripts/trading/run_mr_backtest.py \\
        --data-path data/20240101 \\
        --instruments UST_2Y,UST_5Y,UST_10Y \\
        --ref-instrument UST_5Y \\
        --entry-z 2.5 \\
        --max-holding-bars 30 \\
        --outdir runs/single_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mean-reversion L2 backtest (single or sweep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Mode selection -------------------------------------------------------
    mode = parser.add_argument_group("mode")
    mode.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sweep YAML config (or config-map name).",
    )
    mode.add_argument(
        "--single",
        action="store_true",
        help="Run single backtest using base_config from --config (skip grid).",
    )

    # --- Data (used when no config.data section) ------------------------------
    data = parser.add_argument_group("data")
    data.add_argument("--data-path", type=str, default=None, help="Directory with parquet files.")
    data.add_argument("--book-file", type=str, default="book.parquet")
    data.add_argument("--mid-file", type=str, default="mid.parquet")
    data.add_argument("--dv01-file", type=str, default="dv01.parquet")
    data.add_argument("--signal-file", type=str, default="signal.parquet")
    data.add_argument("--hedge-file", type=str, default="hedge_ratios.parquet")
    data.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated ordered instrument IDs (e.g. UST_2Y,UST_5Y,UST_10Y).",
    )
    data.add_argument("--ref-instrument", type=str, default=None, help="Reference instrument ID for signal.")
    data.add_argument(
        "--fill-method",
        type=str,
        default="ffill",
        choices=["ffill", "drop"],
        help="Datetime alignment method.",
    )

    # --- Backtest config overrides (single-run) -------------------------------
    cfg = parser.add_argument_group("backtest config overrides (single-run)")
    cfg.add_argument("--ref-leg-idx", type=int, default=None)
    cfg.add_argument("--target-notional", type=float, default=None)
    cfg.add_argument("--entry-z", type=float, default=None)
    cfg.add_argument("--tp-zscore-soft", type=float, default=None)
    cfg.add_argument("--tp-yield-soft", type=float, default=None)
    cfg.add_argument("--tp-yield-hard", type=float, default=None)
    cfg.add_argument("--sl-yield-hard", type=float, default=None)
    cfg.add_argument("--max-holding-bars", type=int, default=None)
    cfg.add_argument("--max-levels-to-cross", type=int, default=None)
    cfg.add_argument("--size-haircut", type=float, default=None)
    cfg.add_argument("--tp-quarantine", type=int, default=None)
    cfg.add_argument("--sl-quarantine", type=int, default=None)
    cfg.add_argument("--time-quarantine", type=int, default=None)
    cfg.add_argument("--validate-scope", type=str, default=None, choices=["REF_ONLY", "ALL_LEGS"])
    cfg.add_argument("--use-jit", action="store_true", default=False)

    # --- Output ---------------------------------------------------------------
    out = parser.add_argument_group("output")
    out.add_argument("--outdir", type=str, default=None, help="Output directory.")
    out.add_argument("--no-save", action="store_true", help="Do not persist results to disk.")
    out.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser.parse_args(argv)


def _print(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


# ---------------------------------------------------------------------------
# Single-run mode
# ---------------------------------------------------------------------------

def _run_single(args: argparse.Namespace) -> int:
    from mlstudy.trading.backtest.mean_reversion import (
        MRBacktestConfig,
        run_backtest,
    )
    from mlstudy.trading.backtest.mean_reversion.analysis import (
        compute_performance_metrics,
    )
    from mlstudy.trading.backtest.mean_reversion.data_loader import (
        BacktestDataLoader,
    )

    quiet = args.quiet

    # --- Build config ---------------------------------------------------------
    if args.config is not None:
        from mlstudy.trading.backtest.mean_reversion.sweep_config import load_sweep_config

        sweep_cfg = load_sweep_config(args.config)
        base_dict = asdict(sweep_cfg.base_config)
    else:
        base_dict = {}

    # Apply CLI overrides
    override_map = {
        "ref_leg_idx": args.ref_leg_idx,
        "target_notional_ref": args.target_notional,
        "entry_z_threshold": args.entry_z,
        "take_profit_zscore_soft_threshold": args.tp_zscore_soft,
        "take_profit_yield_change_soft_threshold": args.tp_yield_soft,
        "take_profit_yield_change_hard_threshold": args.tp_yield_hard,
        "stop_loss_yield_change_hard_threshold": args.sl_yield_hard,
        "max_holding_bars": args.max_holding_bars,
        "max_levels_to_cross": args.max_levels_to_cross,
        "size_haircut": args.size_haircut,
        "tp_quarantine_bars": args.tp_quarantine,
        "sl_quarantine_bars": args.sl_quarantine,
        "time_quarantine_bars": args.time_quarantine,
        "validate_scope": args.validate_scope,
        "use_jit": args.use_jit or None,
    }
    for key, val in override_map.items():
        if val is not None:
            base_dict[key] = val

    cfg = MRBacktestConfig(**base_dict)
    _print(f"Backtest config: {cfg}", quiet)

    # --- Load market data -----------------------------------------------------
    loader = _build_data_loader(args)
    data_path = args.data_path
    _print(f"\nLoading market data from {data_path or loader.data_path} ...", quiet)
    t0 = time.perf_counter()
    md = loader.load(data_path=data_path)
    load_time = time.perf_counter() - t0
    _print(f"  Loaded {md.bid_px.shape[0]} bars x {md.bid_px.shape[1]} instruments "
           f"x {md.bid_px.shape[2]} levels  ({load_time:.1f}s)", quiet)

    # --- Run backtest ---------------------------------------------------------
    _print("\nRunning backtest ...", quiet)
    t0 = time.perf_counter()
    results = run_backtest(cfg=cfg, **md.to_dict())
    bt_time = time.perf_counter() - t0
    _print(f"  Done ({bt_time:.1f}s)", quiet)

    # --- Metrics --------------------------------------------------------------
    metrics = compute_performance_metrics(results)
    _print_metrics(metrics, quiet)

    # --- Persist --------------------------------------------------------------
    if not args.no_save:
        outdir = _resolve_outdir(args.outdir, "mr_single")
        _save_single_results(outdir, cfg, md, results, metrics, load_time, bt_time)
        _print(f"\nResults saved to {outdir}", quiet)

    return 0


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def _run_sweep(args: argparse.Namespace) -> int:
    from mlstudy.trading.backtest.mean_reversion import run_sweep_from_config

    quiet = args.quiet

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_data_loader(args: argparse.Namespace):
    """Build a BacktestDataLoader from CLI args or from config YAML.

    ``data_path`` is intentionally *not* baked into the loader here —
    it is passed at ``load(data_path=...)`` time so the same YAML
    config works across platforms.
    """
    from mlstudy.trading.backtest.mean_reversion.data_loader import BacktestDataLoader

    # If a config file has a data section, prefer that
    if args.config is not None:
        from mlstudy.trading.backtest.mean_reversion.sweep_config import load_sweep_config

        sweep_cfg = load_sweep_config(args.config)
        if sweep_cfg.data_loader is not None:
            return sweep_cfg.data_loader

    # Build from CLI args
    if args.instruments is None:
        print("Error: --instruments is required (no data section in config).", file=sys.stderr)
        sys.exit(1)
    if args.instruments is None:
        print("Error: --instruments is required.", file=sys.stderr)
        sys.exit(1)
    if args.ref_instrument is None:
        print("Error: --ref-instrument is required.", file=sys.stderr)
        sys.exit(1)

    instruments = [s.strip() for s in args.instruments.split(",")]

    return BacktestDataLoader(
        book_filename=args.book_file,
        mid_filename=args.mid_file,
        dv01_filename=args.dv01_file,
        signal_filename=args.signal_file,
        hedge_ratio_filename=args.hedge_file,
        instrument_ids=instruments,
        ref_instrument_id=args.ref_instrument,
        fill_method=args.fill_method,
    )


def _resolve_outdir(outdir: str | None, prefix: str) -> Path:
    if outdir is not None:
        d = Path(outdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("runs") / prefix / timestamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _print_metrics(metrics, quiet: bool) -> None:
    if quiet:
        return
    m = metrics
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total PnL           : {m.total_pnl:>12.2f}")
    print(f"  Sharpe Ratio        : {m.sharpe_ratio:>12.4f}")
    print(f"  Sortino Ratio       : {m.sortino_ratio:>12.4f}")
    print(f"  Max Drawdown        : {m.max_drawdown:>12.2f}")
    print(f"  Max DD Duration     : {m.max_drawdown_duration:>12d} bars")
    print(f"  Calmar Ratio        : {m.calmar_ratio:>12.4f}")
    print(f"  # Trades            : {m.n_trades:>12d}")
    print(f"  Hit Rate            : {m.hit_rate:>12.2%}")
    print(f"  Profit Factor       : {m.profit_factor:>12.4f}")
    print(f"  Avg Win / Avg Loss  : {m.avg_win:>8.2f} / {m.avg_loss:>8.2f}")
    print(f"  Win/Loss Ratio      : {m.win_loss_ratio:>12.4f}")
    print(f"  Avg Holding Period  : {m.avg_holding_period:>12.1f} bars")
    print(f"  % Time in Market    : {m.pct_time_in_market:>12.2%}")
    print(f"  Skewness            : {m.skewness:>12.4f}")
    print(f"  Kurtosis            : {m.kurtosis:>12.4f}")
    print(f"  VaR (95%)           : {m.var_95:>12.4f}")
    print(f"  CVaR (95%)          : {m.cvar_95:>12.4f}")
    print(f"{'=' * 60}")


def _save_single_results(
    outdir: Path,
    cfg,
    md,
    results,
    metrics,
    load_time: float,
    bt_time: float,
) -> None:
    """Persist single-run artifacts."""
    import yaml

    # Config snapshot
    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)

    # Metrics
    metrics_dict = asdict(metrics)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)

    # Run metadata
    meta = {
        "T": int(md.bid_px.shape[0]),
        "N": int(md.bid_px.shape[1]),
        "L": int(md.bid_px.shape[2]),
        "instruments": md.instrument_ids,
        "load_time_s": round(load_time, 2),
        "backtest_time_s": round(bt_time, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(outdir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Per-bar arrays
    arrays_dir = outdir / "arrays"
    arrays_dir.mkdir(exist_ok=True)
    np.save(arrays_dir / "pnl.npy", results.pnl)
    np.save(arrays_dir / "equity.npy", results.equity)
    np.save(arrays_dir / "positions.npy", results.positions)
    np.save(arrays_dir / "codes.npy", results.codes)
    np.save(arrays_dir / "state.npy", results.state)
    np.save(arrays_dir / "datetimes.npy", md.datetimes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.config is not None and not args.single:
        return _run_sweep(args)
    else:
        return _run_single(args)


if __name__ == "__main__":
    sys.exit(main())
