"""Price-based time-series features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlstudy.core.features.ops.groupby import grouped_pct_change, grouped_rolling_std
from mlstudy.core.features.ops.time import pct_change, rolling_std
from mlstudy.core.features.registry import register_feature


@register_feature(
    name="returns",
    required_cols=["{price_col}"],
    output_cols_fn=lambda p: [f"{p.get('price_col', 'close')}_return_{p.get('periods', 1)}"],
    description="Compute simple returns over specified periods.",
)
def compute_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: int = 1,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute simple returns.

    Args:
        df: Input DataFrame.
        price_col: Price column name.
        periods: Number of periods for return calculation.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with return column.
    """
    out_col = f"{price_col}_return_{periods}"

    if group_col is not None:
        result = grouped_pct_change(df, price_col, group_col, periods=periods, shift=shift)
    else:
        result = pct_change(df[price_col], periods=periods, shift=shift)

    return pd.DataFrame({out_col: result})


@register_feature(
    name="log_returns",
    required_cols=["{price_col}"],
    output_cols_fn=lambda p: [f"{p.get('price_col', 'close')}_log_return_{p.get('periods', 1)}"],
    description="Compute log returns over specified periods.",
)
def compute_log_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: int = 1,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute log returns.

    Args:
        df: Input DataFrame.
        price_col: Price column name.
        periods: Number of periods for return calculation.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with log return column.
    """
    out_col = f"{price_col}_log_return_{periods}"

    if group_col is not None:
        shifted = df.groupby(group_col)[price_col].shift(shift)
        lagged = df.groupby(group_col)[price_col].shift(shift + periods)
    else:
        shifted = df[price_col].shift(shift)
        lagged = df[price_col].shift(shift + periods)

    log_return = np.log(shifted / lagged)

    return pd.DataFrame({out_col: log_return})


@register_feature(
    name="momentum",
    required_cols=["{price_col}"],
    output_cols_fn=lambda p: [f"{p.get('price_col', 'close')}_momentum_{p.get('window', 20)}"],
    description="Compute momentum as cumulative return over window.",
)
def compute_momentum(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Compute momentum (cumulative return over window).

    Args:
        df: Input DataFrame.
        price_col: Price column name.
        window: Lookback window in periods.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.

    Returns:
        DataFrame with momentum column.
    """
    out_col = f"{price_col}_momentum_{window}"

    if group_col is not None:
        current = df.groupby(group_col)[price_col].shift(shift)
        past = df.groupby(group_col)[price_col].shift(shift + window)
    else:
        current = df[price_col].shift(shift)
        past = df[price_col].shift(shift + window)

    momentum = (current - past) / past

    return pd.DataFrame({out_col: momentum})


@register_feature(
    name="rolling_volatility",
    required_cols=["{price_col}"],
    output_cols_fn=lambda p: [f"{p.get('price_col', 'close')}_vol_{p.get('window', 20)}"],
    description="Compute rolling volatility of returns.",
)
def compute_rolling_volatility(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    shift: int = 1,
    group_col: str | None = None,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Compute rolling volatility of returns.

    Args:
        df: Input DataFrame.
        price_col: Price column name.
        window: Rolling window size.
        shift: Shift for leak safety (default 1).
        group_col: If provided, compute within groups.
        annualize: Whether to annualize volatility.
        periods_per_year: Trading periods per year for annualization.

    Returns:
        DataFrame with volatility column.
    """
    out_col = f"{price_col}_vol_{window}"

    # First compute returns
    if group_col is not None:
        returns = df.groupby(group_col)[price_col].pct_change()
        vol = grouped_rolling_std(
            df.assign(_ret=returns),
            "_ret",
            group_col,
            window=window,
            shift=shift,
        )
    else:
        returns = df[price_col].pct_change()
        vol = rolling_std(returns, window=window, shift=shift)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return pd.DataFrame({out_col: vol})
