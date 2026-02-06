"""Calendar and time-based features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlstudy.core.features.registry import register_feature


@register_feature(
    name="day_of_week",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: ["day_of_week"],
    description="Extract day of week (0=Monday, 6=Sunday).",
)
def compute_day_of_week(
    df: pd.DataFrame,
    datetime_col: str,
) -> pd.DataFrame:
    """Extract day of week from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with day_of_week column (0=Monday, 6=Sunday).
    """
    dt = pd.to_datetime(df[datetime_col])
    return pd.DataFrame({"day_of_week": dt.dt.dayofweek})


@register_feature(
    name="day_of_month",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: ["day_of_month"],
    description="Extract day of month (1-31).",
)
def compute_day_of_month(
    df: pd.DataFrame,
    datetime_col: str,
) -> pd.DataFrame:
    """Extract day of month from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with day_of_month column (1-31).
    """
    dt = pd.to_datetime(df[datetime_col])
    return pd.DataFrame({"day_of_month": dt.dt.day})


@register_feature(
    name="month",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: ["month"],
    description="Extract month (1-12).",
)
def compute_month(
    df: pd.DataFrame,
    datetime_col: str,
) -> pd.DataFrame:
    """Extract month from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with month column (1-12).
    """
    dt = pd.to_datetime(df[datetime_col])
    return pd.DataFrame({"month": dt.dt.month})


@register_feature(
    name="quarter",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: ["quarter"],
    description="Extract quarter (1-4).",
)
def compute_quarter(
    df: pd.DataFrame,
    datetime_col: str,
) -> pd.DataFrame:
    """Extract quarter from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with quarter column (1-4).
    """
    dt = pd.to_datetime(df[datetime_col])
    return pd.DataFrame({"quarter": dt.dt.quarter})


@register_feature(
    name="week_of_year",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: ["week_of_year"],
    description="Extract ISO week of year (1-53).",
)
def compute_week_of_year(
    df: pd.DataFrame,
    datetime_col: str,
) -> pd.DataFrame:
    """Extract ISO week of year from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with week_of_year column (1-53).
    """
    dt = pd.to_datetime(df[datetime_col])
    return pd.DataFrame({"week_of_year": dt.dt.isocalendar().week.astype(int)})


@register_feature(
    name="hour_of_day",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: ["hour_of_day"],
    description="Extract hour of day (0-23).",
)
def compute_hour_of_day(
    df: pd.DataFrame,
    datetime_col: str,
) -> pd.DataFrame:
    """Extract hour of day from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with hour_of_day column (0-23).
    """
    dt = pd.to_datetime(df[datetime_col])
    return pd.DataFrame({"hour_of_day": dt.dt.hour})


@register_feature(
    name="time_features",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: [
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "week_of_year",
    ],
    description="Extract multiple time features at once.",
)
def compute_time_features(
    df: pd.DataFrame,
    datetime_col: str,
    include_hour: bool = False,
) -> pd.DataFrame:
    """Extract multiple time features from datetime.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.
        include_hour: Whether to include hour_of_day.

    Returns:
        DataFrame with time feature columns.
    """
    dt = pd.to_datetime(df[datetime_col])

    result = pd.DataFrame(
        {
            "day_of_week": dt.dt.dayofweek,
            "day_of_month": dt.dt.day,
            "month": dt.dt.month,
            "quarter": dt.dt.quarter,
            "week_of_year": dt.dt.isocalendar().week.astype(int),
        }
    )

    if include_hour:
        result["hour_of_day"] = dt.dt.hour

    return result


@register_feature(
    name="cyclical_time",
    required_cols=["{datetime_col}"],
    output_cols_fn=lambda p: [
        f"{p.get('component', 'day_of_week')}_sin",
        f"{p.get('component', 'day_of_week')}_cos",
    ],
    description="Encode time component as cyclical (sin/cos).",
)
def compute_cyclical_time(
    df: pd.DataFrame,
    datetime_col: str,
    component: str = "day_of_week",
) -> pd.DataFrame:
    """Encode time component as cyclical sin/cos features.

    Useful for capturing cyclical patterns (e.g., day of week, month)
    without artificial discontinuity between last and first values.

    Args:
        df: Input DataFrame.
        datetime_col: Datetime column name.
        component: Time component to encode. Options:
            - "day_of_week" (period=7)
            - "month" (period=12)
            - "hour_of_day" (period=24)
            - "day_of_month" (period=31)

    Returns:
        DataFrame with sin and cos columns.
    """
    dt = pd.to_datetime(df[datetime_col])

    component_map = {
        "day_of_week": (dt.dt.dayofweek, 7),
        "month": (dt.dt.month - 1, 12),  # 0-indexed for proper cycling
        "hour_of_day": (dt.dt.hour, 24),
        "day_of_month": (dt.dt.day - 1, 31),  # 0-indexed
    }

    if component not in component_map:
        raise ValueError(f"Unknown component: {component}. Valid: {list(component_map.keys())}")

    values, period = component_map[component]

    sin_col = f"{component}_sin"
    cos_col = f"{component}_cos"

    return pd.DataFrame(
        {
            sin_col: np.sin(2 * np.pi * values / period),
            cos_col: np.cos(2 * np.pi * values / period),
        }
    )
