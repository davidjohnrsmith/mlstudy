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

    Trade metrics are computed for three PnL views of each round-trip:

    * **net-hedged** (default, no suffix) — uses actual hedge PnL from
      the backtest, allocated pro-rata by DV01 demand.
    * **unhedged** (``_unhedged`` suffix) — raw instrument PnL only.
    * **theoretical-hedged** (``_theo_hedged`` suffix) — hedge PnL
      estimated from entry/exit hedge prices and hedge ratios.

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
        hit_rate: Fraction of profitable round-trips (net-hedged).
        profit_factor: Gross profit / gross loss (net-hedged).
        avg_win: Average winning round-trip P&L (net-hedged).
        avg_loss: Average losing round-trip P&L (net-hedged).
        win_loss_ratio: avg_win / abs(avg_loss) (net-hedged).
        hit_rate_unhedged: Hit rate using unhedged PnL.
        profit_factor_unhedged: Profit factor using unhedged PnL.
        avg_win_unhedged: Average win using unhedged PnL.
        avg_loss_unhedged: Average loss using unhedged PnL.
        win_loss_ratio_unhedged: Win/loss ratio using unhedged PnL.
        hit_rate_theo_hedged: Hit rate using theoretical-hedged PnL.
        profit_factor_theo_hedged: Profit factor using theoretical-hedged PnL.
        avg_win_theo_hedged: Average win using theoretical-hedged PnL.
        avg_loss_theo_hedged: Average loss using theoretical-hedged PnL.
        win_loss_ratio_theo_hedged: Win/loss ratio using theoretical-hedged PnL.
        skewness: Skewness of daily returns.
        kurtosis: Excess kurtosis of daily returns.
        var_95: 5% Value at Risk (daily).
        cvar_95: 5% Conditional VaR (Expected Shortfall).
        n_trades: Number of fills.
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
    # Net-hedged trade metrics (default, most realistic)
    hit_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    # Unhedged trade metrics
    hit_rate_unhedged: float = 0.0
    profit_factor_unhedged: float = 0.0
    avg_win_unhedged: float = 0.0
    avg_loss_unhedged: float = 0.0
    win_loss_ratio_unhedged: float = 0.0
    # Theoretical-hedged trade metrics
    hit_rate_theo_hedged: float = 0.0
    profit_factor_theo_hedged: float = 0.0
    avg_win_theo_hedged: float = 0.0
    avg_loss_theo_hedged: float = 0.0
    win_loss_ratio_theo_hedged: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    n_trades: int = 0
    pct_time_in_market: float = 0.0

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
            "hit_rate_unhedged": self.hit_rate_unhedged,
            "profit_factor_unhedged": self.profit_factor_unhedged,
            "avg_win_unhedged": self.avg_win_unhedged,
            "avg_loss_unhedged": self.avg_loss_unhedged,
            "win_loss_ratio_unhedged": self.win_loss_ratio_unhedged,
            "hit_rate_theo_hedged": self.hit_rate_theo_hedged,
            "profit_factor_theo_hedged": self.profit_factor_theo_hedged,
            "avg_win_theo_hedged": self.avg_win_theo_hedged,
            "avg_loss_theo_hedged": self.avg_loss_theo_hedged,
            "win_loss_ratio_theo_hedged": self.win_loss_ratio_theo_hedged,
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
    annualization_factor: int | None = None,
) -> BacktestMetrics:
    """Compute all backtest metrics from bar_df and trade_df.

    Parameters
    ----------
    bar_df : pd.DataFrame
        Per-bar DataFrame with columns ``equity``, ``state``.
    trade_df : pd.DataFrame, optional
        Round-trip trade DataFrame with columns ``pnl``, ``holding_bars``.
    annualization_factor : int or None
        Bars per year for annualizing ratios.  If *None*, inferred from
        the ``datetime`` column in *bar_df*.

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
