"""Pairwise operations between two aligned series."""

from __future__ import annotations

import pandas as pd


def series_diff(
    series_a: pd.Series,
    series_b: pd.Series,
    shift: int = 1,
) -> pd.Series:
    """Compute difference between two series.

    Args:
        series_a: First series.
        series_b: Second series.
        shift: Shift for leak safety (default 1).

    Returns:
        series_a - series_b (shifted).
    """
    return series_a.shift(shift) - series_b.shift(shift)


def series_ratio(
    series_a: pd.Series,
    series_b: pd.Series,
    shift: int = 1,
    epsilon: float = 1e-10,
) -> pd.Series:
    """Compute ratio between two series.

    Args:
        series_a: Numerator series.
        series_b: Denominator series.
        shift: Shift for leak safety (default 1).
        epsilon: Small value added to denominator to avoid division by zero.

    Returns:
        series_a / series_b (shifted).
    """
    a_shifted = series_a.shift(shift)
    b_shifted = series_b.shift(shift)
    return a_shifted / (b_shifted + epsilon)


def series_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    shift: int = 1,
) -> pd.Series:
    """Compute spread (difference) between two series.

    Alias for series_diff for clarity in spread-based strategies.

    Args:
        series_a: First series (e.g., ask).
        series_b: Second series (e.g., bid).
        shift: Shift for leak safety (default 1).

    Returns:
        Spread series.
    """
    return series_diff(series_a, series_b, shift=shift)


def series_log_ratio(
    series_a: pd.Series,
    series_b: pd.Series,
    shift: int = 1,
    epsilon: float = 1e-10,
) -> pd.Series:
    """Compute log ratio between two series.

    Args:
        series_a: Numerator series.
        series_b: Denominator series.
        shift: Shift for leak safety (default 1).
        epsilon: Small value to avoid log(0).

    Returns:
        log(series_a / series_b) (shifted).
    """
    import numpy as np

    a_shifted = series_a.shift(shift)
    b_shifted = series_b.shift(shift)
    return np.log((a_shifted + epsilon) / (b_shifted + epsilon))


def relative_strength(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
    shift: int = 1,
) -> pd.Series:
    """Compute rolling relative strength between two series.

    Relative strength = rolling_mean(series_a) / rolling_mean(series_b)

    Args:
        series_a: First series.
        series_b: Second series.
        window: Rolling window size.
        shift: Shift for leak safety (default 1).

    Returns:
        Relative strength series.
    """
    a_shifted = series_a.shift(shift)
    b_shifted = series_b.shift(shift)

    a_mean = a_shifted.rolling(window=window, min_periods=window).mean()
    b_mean = b_shifted.rolling(window=window, min_periods=window).mean()

    return a_mean / b_mean.replace(0, float("nan"))
