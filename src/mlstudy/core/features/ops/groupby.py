"""Group-aware feature operations with leak-safe shifting.

These operations apply transformations within each group (e.g., per asset)
while maintaining proper time ordering and preventing lookahead bias.
"""

from __future__ import annotations

import pandas as pd


def grouped_lag(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    periods: int = 1,
    shift: int = 0,
) -> pd.Series:
    """Lag a column within each group.

    Args:
        df: Input DataFrame.
        col: Column to lag.
        group_col: Column defining groups.
        periods: Number of periods to lag.
        shift: Additional shift for leak safety.

    Returns:
        Lagged series.
    """
    total_shift = periods + shift
    return df.groupby(group_col)[col].shift(total_shift)


def grouped_delta(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    periods: int = 1,
    shift: int = 1,
) -> pd.Series:
    """Compute difference within each group.

    Args:
        df: Input DataFrame.
        col: Column to difference.
        group_col: Column defining groups.
        periods: Number of periods for difference.
        shift: Shift for leak safety (default 1).

    Returns:
        Differenced series.
    """

    def _delta(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted - shifted.shift(periods)

    return df.groupby(group_col)[col].transform(_delta)


def grouped_pct_change(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    periods: int = 1,
    shift: int = 1,
) -> pd.Series:
    """Compute percentage change within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute pct change.
        group_col: Column defining groups.
        periods: Number of periods for change.
        shift: Shift for leak safety (default 1).

    Returns:
        Percent change series.
    """

    def _pct_change(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.pct_change(periods=periods)

    return df.groupby(group_col)[col].transform(_pct_change)


def grouped_rolling_mean(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling mean within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute mean.
        group_col: Column defining groups.
        window: Rolling window size.
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling mean series.
    """
    if min_periods is None:
        min_periods = window

    def _rolling_mean(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.rolling(window=window, min_periods=min_periods).mean()

    return df.groupby(group_col)[col].transform(_rolling_mean)


def grouped_rolling_std(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling std within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute std.
        group_col: Column defining groups.
        window: Rolling window size.
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling std series.
    """
    if min_periods is None:
        min_periods = window

    def _rolling_std(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.rolling(window=window, min_periods=min_periods).std()

    return df.groupby(group_col)[col].transform(_rolling_std)


def grouped_rolling_sum(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling sum within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute sum.
        group_col: Column defining groups.
        window: Rolling window size.
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling sum series.
    """
    if min_periods is None:
        min_periods = window

    def _rolling_sum(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.rolling(window=window, min_periods=min_periods).sum()

    return df.groupby(group_col)[col].transform(_rolling_sum)


def grouped_rolling_min(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling min within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute min.
        group_col: Column defining groups.
        window: Rolling window size.
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling min series.
    """
    if min_periods is None:
        min_periods = window

    def _rolling_min(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.rolling(window=window, min_periods=min_periods).min()

    return df.groupby(group_col)[col].transform(_rolling_min)


def grouped_rolling_max(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    window: int,
    min_periods: int | None = None,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling max within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute max.
        group_col: Column defining groups.
        window: Rolling window size.
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        Rolling max series.
    """
    if min_periods is None:
        min_periods = window

    def _rolling_max(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.rolling(window=window, min_periods=min_periods).max()

    return df.groupby(group_col)[col].transform(_rolling_max)


def grouped_ewma(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    span: int | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    min_periods: int = 1,
    shift: int = 1,
) -> pd.Series:
    """Compute EWMA within each group.

    Args:
        df: Input DataFrame.
        col: Column to compute EWMA.
        group_col: Column defining groups.
        span: Span for decay.
        halflife: Halflife for decay.
        alpha: Smoothing factor.
        min_periods: Minimum observations required.
        shift: Shift for leak safety (default 1).

    Returns:
        EWMA series.
    """

    def _ewma(s: pd.Series) -> pd.Series:
        shifted = s.shift(shift)
        return shifted.ewm(
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
        ).mean()

    return df.groupby(group_col)[col].transform(_ewma)
