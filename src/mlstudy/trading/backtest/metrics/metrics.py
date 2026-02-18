"""Backtest performance metrics.

Provides functions to compute standard trading performance metrics
from backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MetricCategory(str, Enum):
    EQUITY = "equity"
    TRADE = "trade"


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics.

    Attributes:
        total_pnl: Total P&L over backtest period.
        mean_daily_return: Average daily return.
        std_daily_return: Standard deviation of daily returns.
        sharpe_ratio: Annualized Sharpe ratio (252 trading days).
        sortino_ratio: Annualized Sortino ratio (downside deviation).
        max_drawdown: Maximum drawdown (negative).
        max_drawdown_duration: Maximum drawdown duration in days.
        calmar_ratio: Annualized return / max drawdown.
        turnover_annual: Annualized turnover (traded notional / gross notional).
        avg_holding_period: Average position holding period in days.
        hit_rate: Fraction of profitable days when in position.
        profit_factor: Gross profit / gross loss.
        avg_win: Average winning day P&L.
        avg_loss: Average losing day P&L.
        win_loss_ratio: avg_win / abs(avg_loss).
        skewness: Skewness of daily returns.
        kurtosis: Excess kurtosis of daily returns.
        var_95: 5% Value at Risk (daily).
        cvar_95: 5% Conditional VaR (Expected Shortfall).
        n_trades: Number of position changes.
        pct_time_in_market: Fraction of days with non-zero position.
    """

    total_pnl: float
    mean_daily_return: float
    std_daily_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    turnover_annual: float
    avg_holding_period: float
    hit_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    n_trades: int
    pct_time_in_market: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "mean_daily_return": self.mean_daily_return,
            "std_daily_return": self.std_daily_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": self.calmar_ratio,
            "turnover_annual": self.turnover_annual,
            "avg_holding_period": self.avg_holding_period,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_loss_ratio": self.win_loss_ratio,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "n_trades": self.n_trades,
            "pct_time_in_market": self.pct_time_in_market,
        }


def compute_metrics(
    bar_df: pd.DataFrame,
    trade_df: pd.DataFrame | None = None,
    *,
    annualization_factor: int = 252,
) -> BacktestMetrics:
    """Compute all backtest metrics from bar_df and trade_df.

    Parameters
    ----------
    bar_df : pd.DataFrame
        Per-bar DataFrame with columns ``pnl``, ``cumulative_pnl``, ``state``.
    trade_df : pd.DataFrame, optional
        Round-trip trade DataFrame with columns ``pnl``, ``holding_bars``.
    annualization_factor : int
        Bars per year for annualizing ratios (default 252).

    Returns
    -------
    BacktestMetrics
    """
    from mlstudy.trading.backtest.metrics.metrics_calculator import MetricsCalculator

    return MetricsCalculator(
        bar_df,
        trade_df,
        annualization_factor=annualization_factor,
    ).compute_all()
