"""Time-series operations (vectorized, leak-safe).

All operations include a `shift` parameter to prevent lookahead bias.
By default, shift=1 means the feature at time t uses data up to t-1.
"""

from __future__ import annotations

import pandas as pd


def lag(
    series: pd.Series,
    periods: int = 1,
    shift: int = 0,
) -> pd.Series:
    """Lag a series by specified periods.

    Args:
        series: Input series.
        periods: Number of periods to lag.
        shift: Additional shift for leak safety (default 0, add 1 for features).

    Returns:
        Lagged series.
    """
    total_shift = periods + shift
    return series.shift(total_shift)


def delta(
    series: pd.Series,
    periods: int = 1,
    shift: int = 1,
) -> pd.Series:
    """Compute difference from `periods` ago.

    Args:
        series: Input series.
        periods: Number of periods for difference.
        shift: Shift for leak safety (default 1).

    Returns:
        Differenced series: value[t-shift] - value[t-shift-periods]
    """
    shifted = series.shift(shift)
    return shifted - shifted.shift(periods)


def pct_change(
    series: pd.Series,
    periods: int = 1,
    shift: int = 1,
) -> pd.Series:
    """Compute percentage change from `periods` ago.

    Args:
        series: Input series.
        periods: Number of periods for change.
        shift: Shift for leak safety (default 1).

    Returns:
        Percent change series.
    """
    shifted = series.shift(shift)
    return shifted.pct_change(periods=periods)


def rolling_mean(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling mean.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum observations required. Defaults to window.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling mean series.
    """
    if min_periods is None:
        min_periods = window
    shifted = series.shift(shift)
    return shifted.rolling(window=window, min_periods=min_periods).mean()


def rolling_std(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling standard deviation.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum observations required. Defaults to window.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling std series.
    """
    if min_periods is None:
        min_periods = window
    shifted = series.shift(shift)
    return shifted.rolling(window=window, min_periods=min_periods).std()


def rolling_sum(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling sum.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum observations required. Defaults to window.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling sum series.
    """
    if min_periods is None:
        min_periods = window
    shifted = series.shift(shift)
    return shifted.rolling(window=window, min_periods=min_periods).sum()


def rolling_min(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling minimum.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum observations required. Defaults to window.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling min series.
    """
    if min_periods is None:
        min_periods = window
    shifted = series.shift(shift)
    return shifted.rolling(window=window, min_periods=min_periods).min()


def rolling_max(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling maximum.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum observations required. Defaults to window.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling max series.
    """
    if min_periods is None:
        min_periods = window
    shifted = series.shift(shift)
    return shifted.rolling(window=window, min_periods=min_periods).max()


def ewma(
    series: pd.Series,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 1,
    shift: int = 1,
) -> pd.Series:
    """Compute exponentially weighted moving average.

    Args:
        series: Input series.
        span: Span for decay (specify one of span/halflife/alpha).
        halflife: Halflife for decay.
        alpha: Smoothing factor (0 < alpha <= 1).
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        EWMA series.
    """
    shifted = series.shift(shift)
    return shifted.ewm(
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
    ).mean()
