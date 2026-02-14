"""Backtest performance metrics.

Provides functions to compute standard trading performance metrics
from backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
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


def compute_sharpe_ratio(
    returns: pd.Series,
    annualization_factor: float = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Daily returns series.
        annualization_factor: Trading days per year.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret < 1e-10:
        return 0.0
    return mean_ret / std_ret * np.sqrt(annualization_factor)


def compute_sortino_ratio(
    returns: pd.Series,
    annualization_factor: float = 252,
) -> float:
    """Compute annualized Sortino ratio (downside deviation).

    Args:
        returns: Daily returns series.
        annualization_factor: Trading days per year.

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) == 0:
        return 0.0
    mean_ret = returns.mean()
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf if mean_ret > 0 else 0.0
    downside_std = downside_returns.std()
    if downside_std < 1e-10:
        return np.inf if mean_ret > 0 else 0.0
    return mean_ret / downside_std * np.sqrt(annualization_factor)


def compute_max_drawdown(cumulative_pnl: pd.Series) -> tuple[float, int]:
    """Compute maximum drawdown and duration.

    Args:
        cumulative_pnl: Cumulative P&L series.

    Returns:
        Tuple of (max_drawdown, max_duration_days).
    """
    if len(cumulative_pnl) == 0:
        return 0.0, 0

    rolling_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - rolling_max
    max_dd = drawdown.min()

    # Compute max duration
    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return 0.0, 0

    # Find drawdown periods
    dd_start = None
    max_duration = 0
    current_duration = 0

    for i, is_dd in enumerate(in_drawdown):
        if is_dd:
            if dd_start is None:
                dd_start = i
            current_duration = i - dd_start + 1
            max_duration = max(max_duration, current_duration)
        else:
            dd_start = None

    return max_dd, max_duration


def compute_turnover(
    traded_notional: pd.Series,
    gross_notional: pd.Series,
    annualization_factor: float = 252,
) -> float:
    """Compute annualized turnover.

    Turnover = traded_notional / average_gross_notional * annualization

    Args:
        traded_notional: Daily traded notional series.
        gross_notional: Daily gross notional series.
        annualization_factor: Trading days per year.

    Returns:
        Annualized turnover ratio.
    """
    total_traded = traded_notional.sum()
    avg_gross = gross_notional[gross_notional > 0].mean()
    if avg_gross < 1e-10:
        return 0.0
    n_days = len(traded_notional)
    if n_days == 0:
        return 0.0
    daily_turnover = total_traded / (n_days * avg_gross)
    return daily_turnover * annualization_factor


def compute_avg_holding_period(position: pd.Series) -> float:
    """Compute average holding period in days.

    Args:
        position: Position series (-1, 0, 1).

    Returns:
        Average holding period in days.
    """
    if len(position) == 0:
        return 0.0

    # Find holding periods
    holding_periods = []
    current_hold = 0

    for pos in position:
        if pos != 0:
            current_hold += 1
        else:
            if current_hold > 0:
                holding_periods.append(current_hold)
            current_hold = 0

    # Don't forget last period
    if current_hold > 0:
        holding_periods.append(current_hold)

    if len(holding_periods) == 0:
        return 0.0
    return np.mean(holding_periods)


def compute_hit_rate(
    returns: pd.Series,
    position: pd.Series,
) -> float:
    """Compute hit rate (fraction of profitable days when in position).

    Args:
        returns: Daily returns series.
        position: Position series.

    Returns:
        Hit rate (0 to 1).
    """
    active_mask = position != 0
    active_returns = returns[active_mask]
    if len(active_returns) == 0:
        return 0.0
    return (active_returns > 0).mean()


def compute_profit_factor(returns: pd.Series) -> float:
    """Compute profit factor (gross profit / gross loss).

    Args:
        returns: Daily returns series.

    Returns:
        Profit factor (> 1 is good).
    """
    profits = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses < 1e-10:
        return np.inf if profits > 0 else 0.0
    return profits / losses


def compute_win_loss_stats(returns: pd.Series) -> tuple[float, float, float]:
    """Compute average win, average loss, and win/loss ratio.

    Args:
        returns: Daily returns series.

    Returns:
        Tuple of (avg_win, avg_loss, win_loss_ratio).
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0  # Negative
    win_loss_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-10 else np.inf

    return avg_win, avg_loss, win_loss_ratio


def compute_tail_stats(returns: pd.Series) -> tuple[float, float, float, float]:
    """Compute tail statistics: skewness, kurtosis, VaR, CVaR.

    Args:
        returns: Daily returns series.

    Returns:
        Tuple of (skewness, excess_kurtosis, var_95, cvar_95).
    """
    if len(returns) < 3:
        return 0.0, 0.0, 0.0, 0.0

    skewness = returns.skew()
    kurtosis = returns.kurtosis()  # Excess kurtosis

    # VaR and CVaR at 5%
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

    return skewness, kurtosis, var_95, cvar_95


def compute_n_trades(position: pd.Series) -> int:
    """Compute number of trades (position changes).

    Args:
        position: Position series.

    Returns:
        Number of position changes.
    """
    changes = position.diff().fillna(0)
    return (changes != 0).sum()


class MetricsCalculator:
    """Compute backtest metrics from a pnl DataFrame.

    Provides three interfaces: equity-only, trades-only, or both.
    """

    def __init__(
        self,
        pnl_df: pd.DataFrame,
        return_col: str = "net_return",
        position_col: str = "position",
        traded_notional_col: str = "traded_notional",
        gross_notional_col: str = "gross_notional",
        cumulative_col: str = "cumulative_pnl",
    ):
        self._returns = pnl_df[return_col].dropna()
        self._position = pnl_df[position_col]
        self._cumulative = pnl_df[cumulative_col].dropna()
        self._pnl_df = pnl_df
        self._traded_notional_col = traded_notional_col
        self._gross_notional_col = gross_notional_col

    def _compute_equity_fields(self) -> dict:
        returns = self._returns
        cumulative = self._cumulative

        total_pnl = returns.sum()
        mean_ret = returns.mean()
        std_ret = returns.std()

        sharpe = compute_sharpe_ratio(returns)
        sortino = compute_sortino_ratio(returns)

        max_dd, max_dd_duration = compute_max_drawdown(cumulative)
        annual_return = mean_ret * 252
        calmar = abs(annual_return / max_dd) if abs(max_dd) > 1e-10 else 0.0

        skew, kurt, var_95, cvar_95 = compute_tail_stats(returns)

        return {
            "total_pnl": total_pnl,
            "mean_daily_return": mean_ret,
            "std_daily_return": std_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "calmar_ratio": calmar,
            "skewness": skew,
            "kurtosis": kurt,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def _compute_trade_fields(self) -> dict:
        returns = self._returns
        position = self._position
        pnl_df = self._pnl_df

        if (
            self._traded_notional_col in pnl_df.columns
            and self._gross_notional_col in pnl_df.columns
        ):
            turnover = compute_turnover(
                pnl_df[self._traded_notional_col],
                pnl_df[self._gross_notional_col],
            )
        else:
            turnover = 0.0

        avg_hold = compute_avg_holding_period(position)
        hit_rate = compute_hit_rate(returns, position)
        profit_factor = compute_profit_factor(returns)
        avg_win, avg_loss, win_loss = compute_win_loss_stats(returns)
        n_trades = compute_n_trades(position)
        pct_in_market = (position != 0).mean()

        return {
            "turnover_annual": turnover,
            "avg_holding_period": avg_hold,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss,
            "n_trades": n_trades,
            "pct_time_in_market": pct_in_market,
        }

    def compute_equity(self) -> BacktestMetrics:
        """Compute equity-curve metrics only; trade fields set to zero."""
        equity = self._compute_equity_fields()
        return BacktestMetrics(
            **equity,
            turnover_annual=0.0,
            avg_holding_period=0.0,
            hit_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            n_trades=0,
            pct_time_in_market=0.0,
        )

    def compute_trades(self) -> BacktestMetrics:
        """Compute trade metrics only; equity fields set to zero."""
        trade = self._compute_trade_fields()
        return BacktestMetrics(
            total_pnl=0.0,
            mean_daily_return=0.0,
            std_daily_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
            **trade,
        )

    def compute_all(self) -> BacktestMetrics:
        """Compute all metrics (equivalent to ``compute_metrics()``)."""
        fields = {**self._compute_equity_fields(), **self._compute_trade_fields()}
        return BacktestMetrics(**fields)


def compute_metrics(
    pnl_df: pd.DataFrame,
    return_col: str = "net_return",
    position_col: str = "position",
    traded_notional_col: str = "traded_notional",
    gross_notional_col: str = "gross_notional",
    cumulative_col: str = "cumulative_pnl",
) -> BacktestMetrics:
    """Compute all backtest metrics from pnl_df.

    Args:
        pnl_df: DataFrame from backtest_fly with required columns.
        return_col: Column with net returns.
        position_col: Column with position.
        traded_notional_col: Column with traded notional.
        gross_notional_col: Column with gross notional.
        cumulative_col: Column with cumulative P&L.

    Returns:
        BacktestMetrics with all computed metrics.
    """
    return MetricsCalculator(
        pnl_df,
        return_col=return_col,
        position_col=position_col,
        traded_notional_col=traded_notional_col,
        gross_notional_col=gross_notional_col,
        cumulative_col=cumulative_col,
    ).compute_all()
