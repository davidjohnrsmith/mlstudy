"""Backtest reporting with plots and output saving.

Generates performance reports, plots, and saves outputs to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.old.engine import BacktestResult
from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics, compute_metrics

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class RegimeData:
    """Optional regime data for extended backtest reporting.

    Attributes:
        fly_yield: Fly yield time series.
        zscore: Z-score time series.
        regime: Regime classification series.
        exit_reasons: Optional ExitReason series.
        entry_z: Entry threshold used.
        exit_z: Exit threshold used.
        stop_z: Stop threshold used.
    """

    fly_yield: pd.Series
    zscore: pd.Series
    regime: pd.Series
    exit_reasons: pd.Series | None = None
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = 4.0


def generate_report(
    result: BacktestResult,
    output_dir: str | Path,
    run_id: str | None = None,
    save_plots: bool = True,
    save_csv: bool = True,
    save_json: bool = True,
    regime_data: RegimeData | None = None,
) -> dict:
    """Generate complete backtest report and save to disk.

    Creates output directory structure:
        outputs/backtests/<run_id>/
            pnl.csv                    # Daily P&L data
            metrics.json               # Performance metrics
            config.json                # Backtest configuration
            cumulative_pnl.png         # Cumulative P&L chart
            drawdown.png               # Drawdown chart
            returns_hist.png           # Returns histogram
            exposures.png              # Exposure time series
            fly_regime_overlay.png     # Fly yield with regime (if regime_data)
            zscore_positions_stops.png # Z-score + positions (if regime_data)
            regime_diagnostics.json    # Regime statistics (if regime_data)

    Args:
        result: BacktestResult from backtest_fly.
        output_dir: Base output directory (e.g., "outputs/backtests").
        run_id: Unique run identifier. Auto-generated if None.
        save_plots: Whether to save plot images.
        save_csv: Whether to save CSV files.
        save_json: Whether to save JSON files.
        regime_data: Optional RegimeData for extended regime diagnostics.

    Returns:
        Dict with paths to saved files and computed metrics.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    metrics = compute_metrics(result.pnl_df)
    metrics_dict = metrics.to_dict()

    saved_files = {"run_id": run_id, "output_dir": str(output_path)}

    # Save CSV
    if save_csv:
        csv_path = output_path / "pnl.csv"
        result.pnl_df.to_csv(csv_path, index=False)
        saved_files["pnl_csv"] = str(csv_path)

    # Save JSON files
    if save_json:
        # Metrics
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            # Handle inf and numpy types
            clean_metrics = {}
            for k, v in metrics_dict.items():
                if v == np.inf or v == -np.inf or (isinstance(v, float) and np.isnan(v)):
                    clean_metrics[k] = None
                elif isinstance(v, (np.integer, np.int64)):
                    clean_metrics[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    clean_metrics[k] = float(v)
                else:
                    clean_metrics[k] = v
            json.dump(clean_metrics, f, indent=2)
        saved_files["metrics_json"] = str(metrics_path)

        # Config
        config_path = output_path / "config.json"
        config_dict = {
            "sizing_mode": result.config.sizing_mode.value,
            "fixed_notional": result.config.fixed_notional,
            "dv01_target": result.config.dv01_target,
            "transaction_cost_bps": result.config.transaction_cost_bps,
            "slippage_bps": result.config.slippage_bps,
            "signal_lag": result.config.signal_lag,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        saved_files["config_json"] = str(config_path)

    # Save plots
    if save_plots and HAS_MATPLOTLIB:
        plot_paths = save_backtest_plots(result, output_path)
        saved_files.update(plot_paths)

    # Regime diagnostics (if provided)
    if regime_data is not None:
        from mlstudy.trading.backtest.old.regime_diagnostics import (
            compute_regime_diagnostics,
            save_regime_plots,
        )

        # Compute regime diagnostics
        regime_diag = compute_regime_diagnostics(
            pnl_df=result.pnl_df,
            regime=regime_data.regime,
            exit_reasons=regime_data.exit_reasons,
        )

        # Save regime diagnostics JSON
        if save_json:
            regime_path = output_path / "regime_diagnostics.json"
            with open(regime_path, "w") as f:
                json.dump(regime_diag.to_dict(), f, indent=2)
            saved_files["regime_diagnostics_json"] = str(regime_path)

        # Save regime plots
        if save_plots and HAS_MATPLOTLIB:
            regime_plots = save_regime_plots(
                output_path=output_path,
                fly_yield=regime_data.fly_yield,
                zscore=regime_data.zscore,
                signal=result.pnl_df.set_index("datetime")["signal"],
                regime=regime_data.regime,
                exit_reasons=regime_data.exit_reasons,
                entry_z=regime_data.entry_z,
                exit_z=regime_data.exit_z,
                stop_z=regime_data.stop_z,
            )
            saved_files.update(regime_plots)

        # Add regime diagnostics to return
        saved_files["regime_diagnostics"] = regime_diag.to_dict()

    # Add metrics to return
    saved_files["metrics"] = metrics_dict

    return saved_files


def save_backtest_plots(
    result: BacktestResult,
    output_path: Path,
) -> dict:
    """Save backtest visualization plots.

    Args:
        result: BacktestResult from backtest_fly.
        output_path: Directory to save plots.

    Returns:
        Dict with paths to saved plot files.
    """
    if not HAS_MATPLOTLIB:
        return {}

    saved = {}
    pnl_df = result.pnl_df

    # 1. Cumulative P&L
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pnl_df["datetime"], pnl_df["cumulative_pnl"], linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title("Cumulative P&L")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_path / "cumulative_pnl.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved["plot_cumulative_pnl"] = str(path)

    # 2. Drawdown
    cumulative = pnl_df["cumulative_pnl"]
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(pnl_df["datetime"], 0, drawdown, alpha=0.7, color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_path / "drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved["plot_drawdown"] = str(path)

    # 3. Returns histogram
    returns = pnl_df["net_return"].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=returns.mean(), color="green", linestyle="-", alpha=0.7, label="Mean")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily Returns Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_path / "returns_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved["plot_returns_hist"] = str(path)

    # 4. Exposures
    if "gross_dv01" in pnl_df.columns and "net_dv01" in pnl_df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # DV01
        ax1 = axes[0]
        ax1.plot(pnl_df["datetime"], pnl_df["gross_dv01"], label="Gross DV01", alpha=0.8)
        ax1.plot(pnl_df["datetime"], pnl_df["net_dv01"], label="Net DV01", alpha=0.8)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("DV01")
        ax1.set_title("DV01 Exposures")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Position
        ax2 = axes[1]
        ax2.fill_between(
            pnl_df["datetime"],
            0,
            pnl_df["position"],
            alpha=0.5,
            step="post",
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Position")
        ax2.set_title("Position")
        ax2.set_ylim(-1.5, 1.5)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = output_path / "exposures.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved["plot_exposures"] = str(path)

    return saved


def print_metrics_summary(metrics: BacktestMetrics) -> None:
    """Print formatted metrics summary to console.

    Args:
        metrics: BacktestMetrics from compute_metrics.
    """
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("=" * 60)

    print("\n--- Returns ---")
    print(f"  Total P&L:        {metrics.total_pnl:>15,.2f}")
    print(f"  Mean Daily:       {metrics.mean_daily_return:>15.6f}")
    print(f"  Std Daily:        {metrics.std_daily_return:>15.6f}")

    print("\n--- Risk-Adjusted ---")
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:>15.2f}")
    print(f"  Sortino Ratio:    {metrics.sortino_ratio:>15.2f}")
    print(f"  Calmar Ratio:     {metrics.calmar_ratio:>15.2f}")

    print("\n--- Drawdown ---")
    print(f"  Max Drawdown:     {metrics.max_drawdown:>15,.2f}")
    print(f"  Max DD Duration:  {metrics.max_drawdown_duration:>15d} days")

    print("\n--- Trading ---")
    print(f"  Num Trades:       {metrics.n_trades:>15d}")
    print(f"  Avg Hold Period:  {metrics.avg_holding_period:>15.1f} days")
    print(f"  Annual Turnover:  {metrics.turnover_annual:>15.2f}x")
    print(f"  Time in Market:   {metrics.pct_time_in_market:>15.1%}")

    print("\n--- Win/Loss ---")
    print(f"  Hit Rate:         {metrics.hit_rate:>15.1%}")
    print(f"  Profit Factor:    {metrics.profit_factor:>15.2f}")
    print(f"  Avg Win:          {metrics.avg_win:>15.6f}")
    print(f"  Avg Loss:         {metrics.avg_loss:>15.6f}")
    print(f"  Win/Loss Ratio:   {metrics.win_loss_ratio:>15.2f}")

    print("\n--- Tail Risk ---")
    print(f"  Skewness:         {metrics.skewness:>15.2f}")
    print(f"  Kurtosis:         {metrics.kurtosis:>15.2f}")
    print(f"  VaR (5%):         {metrics.var_95:>15.6f}")
    print(f"  CVaR (5%):        {metrics.cvar_95:>15.6f}")

    print("\n" + "=" * 60)
