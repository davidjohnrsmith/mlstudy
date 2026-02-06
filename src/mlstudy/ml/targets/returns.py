"""Target variable generation based on future returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_forward_return_target(
    df: pd.DataFrame,
    price_col: str = "close",
    datetime_col: str = "datetime",
    group_col: str | None = None,
    horizon_steps: int = 1,
    log_return: bool = True,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Create forward-looking return target variable.

    Computes the return from time t to time t+horizon_steps.
    This is the target we want to predict at time t.

    IMPORTANT: The target at time t uses price at t and t+horizon_steps.
    This is correct for supervised learning where we want to predict
    future returns based on features available at time t.

    Args:
        df: Input DataFrame with price data.
        price_col: Name of price column.
        datetime_col: Name of datetime column.
        group_col: Name of group column (e.g., asset). If provided,
            computes returns within each group.
        horizon_steps: Number of steps forward for return calculation.
        log_return: If True, compute log return. If False, simple return.
        target_col: Name for output column. If None, auto-generates.

    Returns:
        DataFrame with original index and target column.
        Rows where forward return cannot be computed (end of series)
        will have NaN.

    Example:
        >>> df = pd.DataFrame({
        ...     'datetime': pd.date_range('2024-01-01', periods=5),
        ...     'close': [100, 102, 101, 105, 103]
        ... })
        >>> target_df = make_forward_return_target(df, horizon_steps=1)
        >>> # target at t=0 is return from price[0]=100 to price[1]=102
    """
    if target_col is None:
        ret_type = "log_return" if log_return else "return"
        target_col = f"forward_{ret_type}_{horizon_steps}"

    df = df.copy()

    if group_col is not None:
        # Sort within groups
        df = df.sort_values([group_col, datetime_col]).reset_index(drop=True)

        # Compute forward price within groups
        df["_future_price"] = df.groupby(group_col)[price_col].shift(-horizon_steps)
    else:
        df = df.sort_values(datetime_col).reset_index(drop=True)
        df["_future_price"] = df[price_col].shift(-horizon_steps)

    # Compute return
    if log_return:
        df[target_col] = np.log(df["_future_price"] / df[price_col])
    else:
        df[target_col] = (df["_future_price"] - df[price_col]) / df[price_col]

    # Return only the target column with original index
    return df[[target_col]]


def make_forward_direction_target(
    df: pd.DataFrame,
    price_col: str = "close",
    datetime_col: str = "datetime",
    group_col: str | None = None,
    horizon_steps: int = 1,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Create forward-looking direction (up/down) target for classification.

    Args:
        df: Input DataFrame with price data.
        price_col: Name of price column.
        datetime_col: Name of datetime column.
        group_col: Name of group column.
        horizon_steps: Number of steps forward.
        target_col: Name for output column.

    Returns:
        DataFrame with binary target (1 = price went up, 0 = price went down).
    """
    if target_col is None:
        target_col = f"forward_direction_{horizon_steps}"

    # First compute forward return
    return_df = make_forward_return_target(
        df,
        price_col=price_col,
        datetime_col=datetime_col,
        group_col=group_col,
        horizon_steps=horizon_steps,
        log_return=False,
        target_col="_return",
    )

    # Convert to direction
    direction = (return_df["_return"] > 0).astype(int)
    return pd.DataFrame({target_col: direction})
