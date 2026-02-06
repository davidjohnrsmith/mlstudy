"""Intraday trading session utilities.

Filter data to trading hours and add session-related flags.
"""

from __future__ import annotations

from datetime import time

import pandas as pd


def parse_time(t: str | time) -> time:
    """Parse time string or pass through time object.

    Args:
        t: Time as string "HH:MM" or "HH:MM:SS", or datetime.time object.

    Returns:
        datetime.time object.
    """
    if isinstance(t, time):
        return t
    parts = t.split(":")
    if len(parts) == 2:
        return time(int(parts[0]), int(parts[1]))
    elif len(parts) == 3:
        return time(int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError(f"Invalid time format: {t}. Expected HH:MM or HH:MM:SS")


def filter_session(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    start: str | time = "07:30",
    end: str | time = "17:00",
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Filter DataFrame to trading session hours.

    Args:
        df: DataFrame with datetime column.
        datetime_col: Name of datetime column.
        start: Session start time (inclusive).
        end: Session end time (inclusive).
        tz: Timezone for session times.

    Returns:
        DataFrame filtered to session hours only.

    Example:
        >>> df_session = filter_session(df, start="07:30", end="17:00")
    """
    df = df.copy()
    start_time = parse_time(start)
    end_time = parse_time(end)

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Localize if naive, convert to target timezone
    dt_series = df[datetime_col]
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize(tz)
    else:
        dt_series = dt_series.dt.tz_convert(tz)

    # Extract time and filter
    times = dt_series.dt.time
    mask = (times >= start_time) & (times <= end_time)

    return df[mask].reset_index(drop=True)


def add_session_flags(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    start: str | time = "07:30",
    end: str | time = "17:00",
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Add session-related flags to DataFrame.

    Adds columns:
    - is_session: True if bar is within trading session
    - trading_date: The trading date (date of session start)
    - is_open_bar: True if this is the first bar of the session
    - is_close_bar: True if this is the last bar of the session

    Args:
        df: DataFrame with datetime column.
        datetime_col: Name of datetime column.
        start: Session start time.
        end: Session end time.
        tz: Timezone for session times.

    Returns:
        DataFrame with session flag columns added.

    Example:
        >>> df = add_session_flags(df, start="07:30", end="17:00")
        >>> # df now has is_session, trading_date, is_open_bar, is_close_bar columns
    """
    df = df.copy()
    start_time = parse_time(start)
    end_time = parse_time(end)

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Localize if naive, convert to target timezone
    dt_series = df[datetime_col]
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize(tz)
    else:
        dt_series = dt_series.dt.tz_convert(tz)

    # Store localized datetime for computations
    df["_dt_local"] = dt_series

    # Extract time
    times = dt_series.dt.time

    # is_session flag
    df["is_session"] = (times >= start_time) & (times <= end_time)

    # trading_date: the calendar date of the session
    # For bars before session start, assign to previous trading date
    # For simplicity, use the date part of the localized datetime
    df["trading_date"] = dt_series.dt.date

    # is_open_bar and is_close_bar: first/last session bar per trading_date
    # Only consider session bars
    session_mask = df["is_session"]

    # Initialize flags
    df["is_open_bar"] = False
    df["is_close_bar"] = False

    if session_mask.any():
        # Get first and last session bar indices per trading date
        session_df = df[session_mask].copy()

        # Group by trading_date and find first/last indices
        first_indices = session_df.groupby("trading_date")[datetime_col].idxmin()
        last_indices = session_df.groupby("trading_date")[datetime_col].idxmax()

        df.loc[first_indices.values, "is_open_bar"] = True
        df.loc[last_indices.values, "is_close_bar"] = True

    # Clean up temp column
    df = df.drop(columns=["_dt_local"])

    return df


def get_session_boundaries(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    start: str | time = "07:30",
    end: str | time = "17:00",
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Get session open and close times for each trading date.

    Args:
        df: DataFrame with datetime column.
        datetime_col: Name of datetime column.
        start: Session start time.
        end: Session end time.
        tz: Timezone for session times.

    Returns:
        DataFrame with trading_date, session_open, session_close columns.
    """
    df_flagged = add_session_flags(df, datetime_col, start, end, tz)
    session_df = df_flagged[df_flagged["is_session"]]

    if len(session_df) == 0:
        return pd.DataFrame(columns=["trading_date", "session_open", "session_close"])

    boundaries = (
        session_df.groupby("trading_date")[datetime_col]
        .agg(["min", "max"])
        .reset_index()
    )
    boundaries.columns = ["trading_date", "session_open", "session_close"]

    return boundaries


def compute_trading_date(
    dt: pd.Timestamp | pd.Series,
    session_start: str | time = "07:30",
    tz: str = "Europe/Berlin",
) -> pd.Timestamp | pd.Series:
    """Compute trading date for a datetime.

    Bars before session start are assigned to the previous calendar date's session.
    This handles overnight positions correctly.

    Args:
        dt: Datetime or Series of datetimes.
        session_start: Session start time.
        tz: Timezone.

    Returns:
        Trading date(s).
    """
    start_time = parse_time(session_start)

    if isinstance(dt, pd.Series):
        # Localize if needed
        dt_local = dt.dt.tz_localize(tz) if dt.dt.tz is None else dt.dt.tz_convert(tz)

        # If time is before session start, trading date is previous day
        is_before_start = dt_local.dt.time < start_time
        trading_dates = dt_local.dt.date
        # Shift back by 1 day for pre-session bars
        trading_dates = pd.to_datetime(trading_dates)
        trading_dates = trading_dates.where(
            ~is_before_start,
            trading_dates - pd.Timedelta(days=1),
        )
        return trading_dates.dt.date
    else:
        # Single timestamp
        dt_local = dt.tz_localize(tz) if dt.tz is None else dt.tz_convert(tz)

        if dt_local.time() < start_time:
            return (dt_local - pd.Timedelta(days=1)).date()
        return dt_local.date()


def is_rebalance_bar(
    df: pd.DataFrame,
    rebalance_mode: str = "open_only",
    n_bars: int = 1,
    datetime_col: str = "datetime",
) -> pd.Series:
    """Determine which bars allow position changes.

    Args:
        df: DataFrame with session flags (must have is_session, is_open_bar).
        rebalance_mode: One of:
            - "open_only": Only rebalance at session open
            - "every_bar": Rebalance every session bar
            - "every_n_bars": Rebalance every N session bars
            - "close_only": Only rebalance at session close
        n_bars: Number of bars between rebalances (for every_n_bars mode).
        datetime_col: Datetime column name.

    Returns:
        Boolean Series indicating rebalance bars.
    """
    if "is_session" not in df.columns:
        raise ValueError("DataFrame must have is_session column. Call add_session_flags first.")

    if rebalance_mode == "open_only":
        return df["is_open_bar"]
    elif rebalance_mode == "close_only":
        return df["is_close_bar"]
    elif rebalance_mode == "every_bar":
        return df["is_session"]
    elif rebalance_mode == "every_n_bars":
        # Rebalance every N session bars
        result = pd.Series(False, index=df.index)
        session_mask = df["is_session"]

        if not session_mask.any():
            return result

        # Number session bars
        session_indices = df[session_mask].index
        for i, idx in enumerate(session_indices):
            if i % n_bars == 0:
                result.loc[idx] = True

        return result
    else:
        raise ValueError(
            f"Unknown rebalance_mode: {rebalance_mode}. "
            "Expected one of: open_only, close_only, every_bar, every_n_bars"
        )
