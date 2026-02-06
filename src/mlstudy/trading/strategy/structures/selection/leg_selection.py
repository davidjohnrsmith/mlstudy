"""Daily leg selection for stable intraday trading.

Selects fly legs once per trading day at a specified time and maintains
that selection throughout the day for consistent execution.
"""

from __future__ import annotations

import warnings
from datetime import date, time

import pandas as pd

from mlstudy.core.data.session import parse_time


def build_daily_legs_table(
    panel_df: pd.DataFrame,
    tenors: tuple[float, float, float],
    selection_time: str | time = "07:30",
    tz: str = "Europe/Berlin",
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    max_deviation: float = 1.0,
    fallback_to_previous: bool = True,
) -> pd.DataFrame:
    """Build daily leg selection table for a fly.

    Selects the nearest bonds to target tenors at the selection time each day.
    Selection is fixed for the entire trading day to ensure stability.

    Args:
        panel_df: Panel DataFrame with bond data.
        tenors: Target tenors (front, belly, back).
        selection_time: Time to make leg selection (e.g., "07:30").
        tz: Timezone for selection time.
        datetime_col: Datetime column name.
        bond_id_col: Bond ID column name.
        ttm_col: TTM column name.
        max_deviation: Maximum TTM deviation from target (years).
        fallback_to_previous: If True, use previous day's legs when
            selection bar is missing.

    Returns:
        DataFrame with columns:
        - trading_date: The trading date
        - front_id, belly_id, back_id: Selected bond IDs
        - front_ttm, belly_ttm, back_ttm: Actual TTMs at selection
        - selection_datetime: The actual datetime of selection
        - is_fallback: True if using previous day's selection

    Example:
        >>> legs_table = build_daily_legs_table(
        ...     panel_df, tenors=(2, 5, 10), selection_time="07:30"
        ... )
        >>> # Join to intraday data
        >>> df_with_legs = attach_daily_legs(intraday_df, legs_table)
    """
    df = panel_df.copy()
    sel_time = parse_time(selection_time)

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Localize/convert timezone
    dt_series = df[datetime_col]
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize(tz)
    else:
        dt_series = dt_series.dt.tz_convert(tz)

    df["_dt_local"] = dt_series
    df["_date"] = dt_series.dt.date
    df["_time"] = dt_series.dt.time

    # Get unique trading dates
    trading_dates = sorted(df["_date"].unique())

    results = []
    prev_selection = None

    for trading_date in trading_dates:
        day_df = df[df["_date"] == trading_date]

        # Find bars at or near selection time
        selection_df = _find_selection_bar(day_df, sel_time, datetime_col)

        if len(selection_df) == 0:
            # No data at selection time
            if fallback_to_previous and prev_selection is not None:
                # Use previous day's selection
                result = prev_selection.copy()
                result["trading_date"] = trading_date
                result["is_fallback"] = True
                warnings.warn(
                    f"No data at selection time for {trading_date}, "
                    f"using previous day's legs",
                    stacklevel=2,
                )
            else:
                # Skip this day
                warnings.warn(
                    f"No data at selection time for {trading_date}, skipping",
                    stacklevel=2,
                )
                continue
        else:
            # Select legs at this time
            result = _select_legs_at_time(
                selection_df,
                tenors,
                trading_date,
                bond_id_col,
                ttm_col,
                max_deviation,
                datetime_col,
            )
            result["is_fallback"] = False
            prev_selection = result.copy()

        results.append(result)

    if not results:
        return pd.DataFrame(columns=[
            "trading_date", "front_id", "belly_id", "back_id",
            "front_ttm", "belly_ttm", "back_ttm",
            "selection_datetime", "is_fallback"
        ])

    return pd.DataFrame(results)


def _find_selection_bar(
    day_df: pd.DataFrame,
    sel_time: time,
    datetime_col: str,
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """Find the bar closest to selection time within tolerance.

    Args:
        day_df: Single day's data.
        sel_time: Target selection time.
        datetime_col: Datetime column name.
        tolerance_minutes: Maximum minutes from target time.

    Returns:
        DataFrame with bars at/near selection time.
    """
    times = day_df["_time"]

    # Find exact match first
    exact = day_df[times == sel_time]
    if len(exact) > 0:
        return exact

    # Convert times to minutes for comparison
    target_mins = sel_time.hour * 60 + sel_time.minute

    def time_to_mins(t):
        return t.hour * 60 + t.minute

    time_mins = times.apply(time_to_mins)
    diff = (time_mins - target_mins).abs()

    within_tolerance = diff <= tolerance_minutes
    if not within_tolerance.any():
        return pd.DataFrame()

    # Get the closest time and return all rows at that time
    closest_idx = diff[within_tolerance].idxmin()
    closest_time = times.loc[closest_idx]
    return day_df[times == closest_time]


def _select_legs_at_time(
    selection_df: pd.DataFrame,
    tenors: tuple[float, float, float],
    trading_date: date,
    bond_id_col: str,
    ttm_col: str,
    max_deviation: float,
    datetime_col: str,
) -> dict:
    """Select fly legs from available bonds at selection time.

    Args:
        selection_df: DataFrame with bonds available at selection time.
        tenors: Target tenors (front, belly, back).
        trading_date: Trading date.
        bond_id_col: Bond ID column name.
        ttm_col: TTM column name.
        max_deviation: Maximum TTM deviation.
        datetime_col: Datetime column name.

    Returns:
        Dict with leg selections.
    """
    front_tenor, belly_tenor, back_tenor = tenors

    result = {
        "trading_date": trading_date,
        "selection_datetime": selection_df[datetime_col].iloc[0],
    }

    for leg_name, target_tenor in [
        ("front", front_tenor),
        ("belly", belly_tenor),
        ("back", back_tenor),
    ]:
        # Find bond closest to target tenor
        deviations = (selection_df[ttm_col] - target_tenor).abs()
        min_dev = deviations.min()

        if min_dev > max_deviation:
            warnings.warn(
                f"No bond within {max_deviation}y of {target_tenor}y tenor "
                f"on {trading_date}",
                stacklevel=2,
            )
            result[f"{leg_name}_id"] = None
            result[f"{leg_name}_ttm"] = None
        else:
            closest_idx = deviations.idxmin()
            closest_row = selection_df.loc[closest_idx]
            result[f"{leg_name}_id"] = closest_row[bond_id_col]
            result[f"{leg_name}_ttm"] = closest_row[ttm_col]

    return result


def attach_daily_legs(
    df: pd.DataFrame,
    legs_table: pd.DataFrame,
    datetime_col: str = "datetime",
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Attach daily leg selections to intraday data.

    Joins the daily leg mapping to each bar based on trading date.

    Args:
        df: Intraday DataFrame.
        legs_table: Daily legs table from build_daily_legs_table.
        datetime_col: Datetime column name.
        tz: Timezone.

    Returns:
        DataFrame with leg ID columns added for each bar.
    """
    df = df.copy()

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Get trading date for each bar
    dt_series = df[datetime_col]
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize(tz)
    else:
        dt_series = dt_series.dt.tz_convert(tz)

    df["trading_date"] = dt_series.dt.date

    # Ensure legs_table trading_date is same type
    legs_table = legs_table.copy()
    if not isinstance(legs_table["trading_date"].iloc[0], date):
        legs_table["trading_date"] = pd.to_datetime(legs_table["trading_date"]).dt.date

    # Join legs
    leg_cols = [
        "trading_date", "front_id", "belly_id", "back_id",
        "front_ttm", "belly_ttm", "back_ttm"
    ]
    leg_cols = [c for c in leg_cols if c in legs_table.columns]

    df = df.merge(
        legs_table[leg_cols],
        on="trading_date",
        how="left",
    )

    return df


def get_leg_values(
    panel_df: pd.DataFrame,
    df_with_legs: pd.DataFrame,
    value_cols: list[str],
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
) -> pd.DataFrame:
    """Get values for each leg at each bar.

    Joins panel data to get price, yield, dv01 etc. for each leg.

    Args:
        panel_df: Panel DataFrame with bond data.
        df_with_legs: DataFrame with leg IDs attached (from attach_daily_legs).
        value_cols: Columns to retrieve (e.g., ["price", "yield", "dv01"]).
        datetime_col: Datetime column name.
        bond_id_col: Bond ID column name.

    Returns:
        DataFrame with {leg}_{value} columns for each leg and value.
    """
    df = df_with_legs.copy()

    for leg in ["front", "belly", "back"]:
        leg_id_col = f"{leg}_id"
        if leg_id_col not in df.columns:
            continue

        # Get values for this leg
        leg_values = df[[datetime_col, leg_id_col]].merge(
            panel_df[[datetime_col, bond_id_col] + value_cols],
            left_on=[datetime_col, leg_id_col],
            right_on=[datetime_col, bond_id_col],
            how="left",
        )

        # Rename columns
        for col in value_cols:
            df[f"{leg}_{col}"] = leg_values[col].values

    return df


def validate_leg_stability(
    df_with_legs: pd.DataFrame,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Validate that leg selections are stable within each day.

    Args:
        df_with_legs: DataFrame with leg IDs attached.
        datetime_col: Datetime column name.

    Returns:
        DataFrame with stability metrics per trading date.
    """
    results = []

    for trading_date, group in df_with_legs.groupby("trading_date"):
        n_bars = len(group)
        is_stable = True

        for leg in ["front", "belly", "back"]:
            id_col = f"{leg}_id"
            if id_col in group.columns:
                unique_ids = group[id_col].nunique()
                if unique_ids > 1:
                    is_stable = False

        results.append({
            "trading_date": trading_date,
            "n_bars": n_bars,
            "is_stable": is_stable,
            "front_id": group["front_id"].iloc[0] if "front_id" in group.columns else None,
            "belly_id": group["belly_id"].iloc[0] if "belly_id" in group.columns else None,
            "back_id": group["back_id"].iloc[0] if "back_id" in group.columns else None,
        })

    return pd.DataFrame(results)
