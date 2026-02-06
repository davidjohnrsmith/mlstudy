"""Volume-based time-series features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlstudy.core.features.ops.groupby import grouped_rolling_mean, grouped_rolling_std
from mlstudy.core.features.ops.time import rolling_mean, rolling_std
from mlstudy.core.features.registry import register_feature


@register_feature(
    name="volume_shock",
    required_cols=["{volume_col}"],
    output_cols_fn=lambda p: [f"{p.get('volume_col', 'volume')}_shock_{p.get('window', 20)}"],
    description="Compute volume shock (z-score relative to rolling mean).",
)
def compute_volume_shock(
    df: pd.DataFrame,
    volume_col: str = "volume",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute volume shock as z-score relative to rolling statistics.

    Volume shock = (volume - rolling_mean) / rolling_std

    Args:
        df: Input DataFrame.
        volume_col: Volume column name.
        window: Rolling window for mean/std calculation.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with volume shock column.
    """
    out_col = f"{volume_col}_shock_{window}"

    if group_col is not None:
        vol_mean = grouped_rolling_mean(df, volume_col, group_col, window=window, shift=shift)
        vol_std = grouped_rolling_std(df, volume_col, group_col, window=window, shift=shift)
        current = df.groupby(group_col)[volume_col].shift(shift)
    else:
        vol_mean = rolling_mean(df[volume_col], window=window, shift=shift)
        vol_std = rolling_std(df[volume_col], window=window, shift=shift)
        current = df[volume_col].shift(shift)

    shock = (current - vol_mean) / vol_std.replace(0, np.nan)

    return pd.DataFrame({out_col: shock})


@register_feature(
    name="rolling_volume",
    required_cols=["{volume_col}"],
    output_cols_fn=lambda p: [f"{p.get('volume_col', 'volume')}_ma_{p.get('window', 20)}"],
    description="Compute rolling average volume.",
)
def compute_rolling_volume(
    df: pd.DataFrame,
    volume_col: str = "volume",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute rolling average volume.

    Args:
        df: Input DataFrame.
        volume_col: Volume column name.
        window: Rolling window size.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with rolling volume column.
    """
    out_col = f"{volume_col}_ma_{window}"

    if group_col is not None:
        result = grouped_rolling_mean(df, volume_col, group_col, window=window, shift=shift)
    else:
        result = rolling_mean(df[volume_col], window=window, shift=shift)

    return pd.DataFrame({out_col: result})


@register_feature(
    name="dollar_volume",
    required_cols=["{price_col}", "{volume_col}"],
    output_cols_fn=lambda p: [f"dollar_volume_{p.get('window', 20)}"],
    description="Compute rolling average dollar volume (price * volume).",
)
def compute_dollar_volume(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute rolling average dollar volume.

    Args:
        df: Input DataFrame.
        price_col: Price column name.
        volume_col: Volume column name.
        window: Rolling window size.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with dollar volume column.
    """
    out_col = f"dollar_volume_{window}"

    # Compute dollar volume
    dv = df[price_col] * df[volume_col]

    if group_col is not None:
        df_temp = df.assign(_dv=dv)
        result = grouped_rolling_mean(df_temp, "_dv", group_col, window=window, shift=shift)
    else:
        result = rolling_mean(dv, window=window, shift=shift)

    return pd.DataFrame({out_col: result})


@register_feature(
    name="volume_ratio",
    required_cols=["{volume_col}"],
    output_cols_fn=lambda p: [
        f"{p.get('volume_col', 'volume')}_ratio_{p.get('short_window', 5)}_{p.get('long_window', 20)}"
    ],
    description="Compute ratio of short-term to long-term volume.",
)
def compute_volume_ratio(
    df: pd.DataFrame,
    volume_col: str = "volume",
    short_window: int = 5,
    long_window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute volume ratio (short-term MA / long-term MA).

    Args:
        df: Input DataFrame.
        volume_col: Volume column name.
        short_window: Short-term window.
        long_window: Long-term window.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with volume ratio column.
    """
    out_col = f"{volume_col}_ratio_{short_window}_{long_window}"

    if group_col is not None:
        short_ma = grouped_rolling_mean(df, volume_col, group_col, window=short_window, shift=shift)
        long_ma = grouped_rolling_mean(df, volume_col, group_col, window=long_window, shift=shift)
    else:
        short_ma = rolling_mean(df[volume_col], window=short_window, shift=shift)
        long_ma = rolling_mean(df[volume_col], window=long_window, shift=shift)

    ratio = short_ma / long_ma.replace(0, np.nan)

    return pd.DataFrame({out_col: ratio})
