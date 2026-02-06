"""Flow-based time-series features (order flow, imbalance, etc.)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlstudy.core.features.ops.groupby import grouped_rolling_sum
from mlstudy.core.features.ops.time import rolling_sum
from mlstudy.core.features.registry import register_feature


@register_feature(
    name="flow_imbalance",
    required_cols=["{buy_col}", "{sell_col}"],
    output_cols_fn=lambda p: [f"flow_imbalance_{p.get('window', 20)}"],
    description="Compute order flow imbalance (buy - sell) / (buy + sell).",
)
def compute_flow_imbalance(
    df: pd.DataFrame,
    buy_col: str = "buy_volume",
    sell_col: str = "sell_volume",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute rolling order flow imbalance.

    Imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    Args:
        df: Input DataFrame.
        buy_col: Buy volume column name.
        sell_col: Sell volume column name.
        window: Rolling window for smoothing.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with flow imbalance column.

    Note:
        If buy_col or sell_col don't exist, returns NaN column with warning.
    """
    out_col = f"flow_imbalance_{window}"

    if buy_col not in df.columns or sell_col not in df.columns:
        # Return NaN column if required columns missing
        return pd.DataFrame({out_col: np.nan}, index=df.index)

    if group_col is not None:
        buy_sum = grouped_rolling_sum(df, buy_col, group_col, window=window, shift=shift)
        sell_sum = grouped_rolling_sum(df, sell_col, group_col, window=window, shift=shift)
    else:
        buy_sum = rolling_sum(df[buy_col], window=window, shift=shift)
        sell_sum = rolling_sum(df[sell_col], window=window, shift=shift)

    total = buy_sum + sell_sum
    imbalance = (buy_sum - sell_sum) / total.replace(0, np.nan)

    return pd.DataFrame({out_col: imbalance})


@register_feature(
    name="signed_volume",
    required_cols=["{volume_col}", "{price_col}"],
    output_cols_fn=lambda p: [f"signed_volume_{p.get('window', 20)}"],
    description="Compute signed volume based on price direction.",
)
def compute_signed_volume(
    df: pd.DataFrame,
    volume_col: str = "volume",
    price_col: str = "close",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute rolling signed volume.

    Signs volume based on price change direction:
    - Positive volume if price went up
    - Negative volume if price went down

    Args:
        df: Input DataFrame.
        volume_col: Volume column name.
        price_col: Price column name for direction.
        window: Rolling window for smoothing.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with signed volume column.
    """
    out_col = f"signed_volume_{window}"

    # Compute price direction (sign of return)
    if group_col is not None:
        price_change = df.groupby(group_col)[price_col].diff()
    else:
        price_change = df[price_col].diff()

    sign = np.sign(price_change)
    signed_vol = df[volume_col] * sign

    # Rolling sum of signed volume
    if group_col is not None:
        df_temp = df.assign(_sv=signed_vol)
        result = grouped_rolling_sum(df_temp, "_sv", group_col, window=window, shift=shift)
    else:
        result = rolling_sum(signed_vol, window=window, shift=shift)

    return pd.DataFrame({out_col: result})


@register_feature(
    name="volume_imbalance",
    required_cols=["{volume_col}", "{price_col}"],
    output_cols_fn=lambda p: [f"volume_imbalance_{p.get('window', 20)}"],
    description="Compute volume imbalance (up_volume - down_volume) / total.",
)
def compute_volume_imbalance(
    df: pd.DataFrame,
    volume_col: str = "volume",
    price_col: str = "close",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute rolling volume imbalance ratio.

    Imbalance = (up_volume - down_volume) / total_volume

    Args:
        df: Input DataFrame.
        volume_col: Volume column name.
        price_col: Price column name for direction.
        window: Rolling window.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with volume imbalance column.
    """
    out_col = f"volume_imbalance_{window}"

    # Compute price direction
    if group_col is not None:
        price_change = df.groupby(group_col)[price_col].diff()
    else:
        price_change = df[price_col].diff()

    # Split volume by direction
    up_vol = df[volume_col].where(price_change > 0, 0)
    down_vol = df[volume_col].where(price_change < 0, 0)

    if group_col is not None:
        df_temp = df.assign(_up=up_vol, _down=down_vol)
        up_sum = grouped_rolling_sum(df_temp, "_up", group_col, window=window, shift=shift)
        down_sum = grouped_rolling_sum(df_temp, "_down", group_col, window=window, shift=shift)
        total = grouped_rolling_sum(df, volume_col, group_col, window=window, shift=shift)
    else:
        up_sum = rolling_sum(up_vol, window=window, shift=shift)
        down_sum = rolling_sum(down_vol, window=window, shift=shift)
        total = rolling_sum(df[volume_col], window=window, shift=shift)

    imbalance = (up_sum - down_sum) / total.replace(0, np.nan)

    return pd.DataFrame({out_col: imbalance})
