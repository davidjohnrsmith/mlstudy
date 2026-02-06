"""Performance and risk metrics for strategy evaluation.

This module consolidates metric functions from strategy_pre_selector and
new_issuance/portfolio into a single unified API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(r: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualized mean return."""
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * ann_factor)


def annualized_vol(r: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualized volatility (std * sqrt(ann_factor))."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def sharpe(r: pd.Series, ann_factor: float = 252.0) -> float:
    """Annualized Sharpe ratio (mean / std * sqrt(ann_factor))."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float((r.mean() / sd) * np.sqrt(ann_factor))


def max_drawdown(r: pd.Series) -> float:
    """
    Max drawdown computed on a cumulative curve from returns.

    Returns are assumed to be arithmetic (not log). Output is negative (e.g. -0.15).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def worst_rolling_return(r: pd.Series, window: int = 21) -> float:
    """
    Worst rolling cumulative return over 'window' periods, arithmetic compounding.

    For daily data, window=21 approximates one month.
    """
    r = r.dropna()
    if len(r) < window:
        return np.nan
    roll = (1.0 + r).rolling(window).apply(np.prod, raw=True) - 1.0
    return float(roll.min())


def downside_deviation(r: pd.Series, ann_factor: float = 252.0, mar: float = 0.0) -> float:
    """
    Downside deviation (Sortino denominator), MAR default 0.

    Parameters
    ----------
    r : pd.Series
        Return series.
    ann_factor : float
        Annualization factor.
    mar : float
        Minimum acceptable return (typically 0).
    """
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    downside = np.minimum(r - mar, 0.0)
    dd = downside.std(ddof=1)
    return float(dd * np.sqrt(ann_factor))


def sortino(r: pd.Series, ann_factor: float = 252.0, mar: float = 0.0) -> float:
    """Sortino ratio (mean / downside_deviation * sqrt(ann_factor))."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    dd = downside_deviation(r, ann_factor=ann_factor, mar=mar)
    if dd == 0 or not np.isfinite(dd):
        return np.nan
    return float((r.mean() - mar) / dd * np.sqrt(ann_factor))


def newey_west_tstat(r: pd.Series, lags: int = 5) -> float:
    """
    Newey-West t-stat for mean of returns (robust to autocorr/heterosk).

    Lightweight implementation; for production consider statsmodels.

    Parameters
    ----------
    r : pd.Series
        Return series.
    lags : int
        Number of lags for Bartlett kernel. 5 is a common default for daily data.
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return np.nan
    mu = x.mean()
    u = x - mu

    # gamma_0
    gamma0 = (u @ u) / n
    var = gamma0

    # Bartlett weights
    L = min(lags, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gamma = (u[k:] @ u[:-k]) / n
        var += 2.0 * w * gamma

    se = np.sqrt(var / n)
    if se == 0:
        return np.nan
    return float(mu / se)


def turnover_proxy(r: pd.Series) -> float:
    """
    Generic turnover proxy for return streams when you don't have trades.

    Uses roughness of returns (mean of absolute differences).
    If you *do* have turnover, override this with your own metric.
    """
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.diff().abs().mean())


def positive_year_fraction(r: pd.Series) -> float:
    """
    Fraction of calendar years with positive cumulative return.

    Parameters
    ----------
    r : pd.Series
        Return series with DatetimeIndex.

    Returns
    -------
    float
        Fraction in [0, 1], or nan if no complete years.
    """
    r = r.dropna()
    if r.empty or not hasattr(r.index, "year"):
        return np.nan
    by_year = (1.0 + r).groupby(r.index.year).prod() - 1.0
    if by_year.shape[0] < 1:
        return np.nan
    return float((by_year > 0).mean())
