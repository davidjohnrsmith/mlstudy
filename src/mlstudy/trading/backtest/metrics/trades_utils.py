from __future__ import annotations

import numpy as np
import pandas as pd


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


def compute_n_trades(position: pd.Series) -> int:
    """Compute number of trades (position changes).

    Args:
        position: Position series.

    Returns:
        Number of position changes.
    """
    changes = position.diff().fillna(0)
    return (changes != 0).sum()
