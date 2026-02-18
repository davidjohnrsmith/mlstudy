from __future__ import annotations

import numpy as np
import pandas as pd


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
