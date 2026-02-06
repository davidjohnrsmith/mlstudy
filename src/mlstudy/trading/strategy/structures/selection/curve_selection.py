"""Curve trading bond selection utilities.

Select bonds for curve trades (e.g., 2s5s10s butterfly) based on
time-to-maturity matching and optional liquidity constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FlyLegs:
    """Selected legs for a butterfly trade.

    Attributes:
        datetime: Date of selection.
        short_bond: Bond ID for short leg (e.g., 2Y).
        belly_bond: Bond ID for belly leg (e.g., 5Y).
        long_bond: Bond ID for long leg (e.g., 10Y).
        short_ttm: Actual TTM of short leg.
        belly_ttm: Actual TTM of belly leg.
        long_ttm: Actual TTM of long leg.
    """

    datetime: Any
    short_bond: str
    belly_bond: str
    long_bond: str
    short_ttm: float
    belly_ttm: float
    long_ttm: float


def select_nearest_to_tenor(
    df: pd.DataFrame,
    tenor_years: float,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    max_deviation: float | None = None,
) -> pd.DataFrame:
    """Select the bond nearest to target tenor for each date.

    For each date, finds the bond whose time-to-maturity is closest
    to the target tenor.

    Args:
        df: Panel DataFrame with datetime, bond_id, and ttm columns.
        tenor_years: Target tenor in years (e.g., 5.0 for 5-year).
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        ttm_col: Time-to-maturity column name (in years).
        max_deviation: Maximum allowed deviation from target tenor.
            If set, returns NaN for dates without suitable bonds.

    Returns:
        DataFrame with columns: [datetime_col, bond_id_col, ttm_col, "deviation"].
        One row per date with the selected bond.

    Example:
        >>> # Select bond closest to 5-year tenor each day
        >>> selected = select_nearest_to_tenor(df, tenor_years=5.0)
        >>> # Get the 5Y bond IDs for merging
        >>> df_with_5y = df.merge(selected[[datetime_col, bond_id_col]], on=datetime_col)
    """
    df = df.copy()
    df["_deviation"] = np.abs(df[ttm_col] - tenor_years)

    # For each date, find bond with minimum deviation
    idx = df.groupby(datetime_col)["_deviation"].idxmin()
    result = df.loc[idx, [datetime_col, bond_id_col, ttm_col, "_deviation"]].copy()
    result = result.rename(columns={"_deviation": "deviation"})

    # Filter by max deviation if specified
    if max_deviation is not None:
        mask = result["deviation"] <= max_deviation
        # Keep all dates but mark invalid selections
        result.loc[~mask, bond_id_col] = None
        result.loc[~mask, ttm_col] = np.nan

    return result.reset_index(drop=True)


def select_fly_legs(
    df: pd.DataFrame,
    tenors: tuple[float, float, float] = (2.0, 5.0, 10.0),
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    method: str = "nearest_ttm",
    max_deviation: float | None = None,
    liquidity_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    min_liquidity_col: str | None = None,
    min_liquidity_threshold: float | None = None,
) -> pd.DataFrame:
    """Select three bonds for a butterfly trade.

    For each date, selects bonds closest to the three target tenors
    (e.g., 2Y-5Y-10Y for a standard butterfly).

    Args:
        df: Panel DataFrame with datetime, bond_id, ttm, and optionally liquidity columns.
        tenors: Tuple of (short_tenor, belly_tenor, long_tenor) in years.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        ttm_col: Time-to-maturity column name.
        method: Selection method. Currently only "nearest_ttm" supported.
        max_deviation: Maximum allowed TTM deviation from target.
        liquidity_filter: Optional function to filter bonds before selection.
        min_liquidity_col: Column to use for minimum liquidity filtering.
        min_liquidity_threshold: Minimum value for liquidity column.

    Returns:
        DataFrame with columns:
        - datetime_col
        - short_bond, belly_bond, long_bond (bond IDs)
        - short_ttm, belly_ttm, long_ttm (actual TTMs)
        - short_deviation, belly_deviation, long_deviation

    Example:
        >>> # Select 2s5s10s butterfly legs
        >>> legs = select_fly_legs(df, tenors=(2, 5, 10))
        >>>
        >>> # With liquidity filter
        >>> legs = select_fly_legs(
        ...     df, tenors=(2, 5, 10),
        ...     min_liquidity_col="dv01",
        ...     min_liquidity_threshold=1000,
        ... )
    """
    short_tenor, belly_tenor, long_tenor = tenors

    # Apply liquidity filter if specified
    df_filtered = df.copy()
    if liquidity_filter is not None:
        df_filtered = liquidity_filter(df_filtered)
    elif min_liquidity_col is not None and min_liquidity_threshold is not None:
        df_filtered = df_filtered[df_filtered[min_liquidity_col] >= min_liquidity_threshold]

    # Select each leg
    short_df = select_nearest_to_tenor(
        df_filtered, short_tenor, datetime_col, bond_id_col, ttm_col, max_deviation
    )
    belly_df = select_nearest_to_tenor(
        df_filtered, belly_tenor, datetime_col, bond_id_col, ttm_col, max_deviation
    )
    long_df = select_nearest_to_tenor(
        df_filtered, long_tenor, datetime_col, bond_id_col, ttm_col, max_deviation
    )

    # Merge results
    result = short_df[[datetime_col]].copy()
    result["short_bond"] = short_df[bond_id_col].values
    result["short_ttm"] = short_df[ttm_col].values
    result["short_deviation"] = short_df["deviation"].values

    result["belly_bond"] = belly_df[bond_id_col].values
    result["belly_ttm"] = belly_df[ttm_col].values
    result["belly_deviation"] = belly_df["deviation"].values

    result["long_bond"] = long_df[bond_id_col].values
    result["long_ttm"] = long_df[ttm_col].values
    result["long_deviation"] = long_df["deviation"].values

    return result


def select_nearest_three_sorted_by_ttm(
    df: pd.DataFrame,
    center_tenor: float = 5.0,
    window_years: float = 3.0,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    require_distinct: bool = True,
) -> pd.DataFrame:
    """Select three bonds around a center tenor, sorted by TTM.

    For each date, selects the three bonds closest to the center tenor
    (within a window) and returns them sorted by TTM. This is useful
    for constructing butterfly trades around a specific point on the curve.

    Args:
        df: Panel DataFrame.
        center_tenor: Center tenor to search around (in years).
        window_years: Maximum distance from center_tenor to consider.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        ttm_col: Time-to-maturity column name.
        require_distinct: If True, require three distinct bonds.

    Returns:
        DataFrame with columns:
        - datetime_col
        - bond_1, bond_2, bond_3 (bond IDs sorted by TTM)
        - ttm_1, ttm_2, ttm_3 (corresponding TTMs)

    Example:
        >>> # Get 3 bonds closest to 5Y
        >>> nearby = select_nearest_three_sorted_by_ttm(df, center_tenor=5.0, window_years=2.0)
        >>> # bond_1 is shortest, bond_3 is longest
    """
    results = []

    for dt, group in df.groupby(datetime_col):
        # Filter to window around center
        group = group.copy()
        group["_distance"] = np.abs(group[ttm_col] - center_tenor)
        in_window = group[group["_distance"] <= window_years]

        if len(in_window) < 3:
            if require_distinct:
                # Not enough bonds in window
                results.append({
                    datetime_col: dt,
                    "bond_1": None, "bond_2": None, "bond_3": None,
                    "ttm_1": np.nan, "ttm_2": np.nan, "ttm_3": np.nan,
                })
                continue
            else:
                # Use all available bonds
                in_window = group.copy()

        # Get 3 closest to center, then sort by TTM
        closest = in_window.nsmallest(3, "_distance")
        sorted_by_ttm = closest.sort_values(ttm_col)

        bonds = sorted_by_ttm[bond_id_col].values
        ttms = sorted_by_ttm[ttm_col].values

        # Pad if fewer than 3
        while len(bonds) < 3:
            bonds = np.append(bonds, None)
            ttms = np.append(ttms, np.nan)

        results.append({
            datetime_col: dt,
            "bond_1": bonds[0],
            "bond_2": bonds[1],
            "bond_3": bonds[2],
            "ttm_1": ttms[0],
            "ttm_2": ttms[1],
            "ttm_3": ttms[2],
        })

    return pd.DataFrame(results)


def compute_fly_weights(
    df_legs: pd.DataFrame,
    weight_method: str = "duration_neutral",
    dv01_short: pd.Series | None = None,
    dv01_belly: pd.Series | None = None,
    dv01_long: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute butterfly trade weights.

    Args:
        df_legs: DataFrame from select_fly_legs with leg selections.
        weight_method: Weighting method:
            - "equal": Equal weights (1, -2, 1)
            - "duration_neutral": DV01-weighted to be duration neutral
            - "regression": Weights from regression (not implemented)
        dv01_short: DV01 values for short leg bonds.
        dv01_belly: DV01 values for belly leg bonds.
        dv01_long: DV01 values for long leg bonds.

    Returns:
        DataFrame with columns: w_short, w_belly, w_long.
    """
    n = len(df_legs)

    if weight_method == "equal":
        return pd.DataFrame({
            "w_short": np.ones(n),
            "w_belly": -2 * np.ones(n),
            "w_long": np.ones(n),
        })

    elif weight_method == "duration_neutral":
        if dv01_short is None or dv01_belly is None or dv01_long is None:
            raise ValueError("DV01 values required for duration_neutral weighting")

        # Duration neutral: w_s * DV01_s + w_b * DV01_b + w_l * DV01_l = 0
        # Standard approach: w_s = 1, w_l = 1, solve for w_b
        w_short = np.ones(n)
        w_long = np.ones(n)

        # w_b = -(DV01_s + DV01_l) / DV01_b
        w_belly = -(dv01_short.values + dv01_long.values) / dv01_belly.values

        return pd.DataFrame({
            "w_short": w_short,
            "w_belly": w_belly,
            "w_long": w_long,
        })

    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")


def get_fly_values(
    df: pd.DataFrame,
    df_legs: pd.DataFrame,
    value_col: str,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    weights: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute butterfly spread values from panel data.

    Args:
        df: Panel DataFrame with values.
        df_legs: DataFrame from select_fly_legs with leg selections.
        value_col: Value column (e.g., "yield", "price").
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        weights: Optional weights DataFrame with w_short, w_belly, w_long.
            If None, uses equal weights (1, -2, 1).

    Returns:
        Series with butterfly spread values indexed by datetime.

    Example:
        >>> legs = select_fly_legs(df, tenors=(2, 5, 10))
        >>> fly_yield = get_fly_values(df, legs, value_col="yield")
        >>> # fly_yield = yield_2y - 2*yield_5y + yield_10y
    """
    if weights is None:
        weights = compute_fly_weights(df_legs, "equal")

    # Create lookup for values
    value_lookup = df.set_index([datetime_col, bond_id_col])[value_col]

    results = []
    for i, row in df_legs.iterrows():
        dt = row[datetime_col]

        try:
            v_short = value_lookup.loc[(dt, row["short_bond"])]
            v_belly = value_lookup.loc[(dt, row["belly_bond"])]
            v_long = value_lookup.loc[(dt, row["long_bond"])]

            fly_value = (
                weights.iloc[i]["w_short"] * v_short +
                weights.iloc[i]["w_belly"] * v_belly +
                weights.iloc[i]["w_long"] * v_long
            )
        except KeyError:
            fly_value = np.nan

        results.append(fly_value)

    return pd.Series(results, index=df_legs[datetime_col], name=f"fly_{value_col}")


def select_steepener_legs(
    df: pd.DataFrame,
    short_tenor: float = 2.0,
    long_tenor: float = 10.0,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    max_deviation: float | None = None,
) -> pd.DataFrame:
    """Select two bonds for a steepener/flattener trade.

    Args:
        df: Panel DataFrame.
        short_tenor: Short leg tenor in years.
        long_tenor: Long leg tenor in years.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        ttm_col: Time-to-maturity column name.
        max_deviation: Maximum TTM deviation from target.

    Returns:
        DataFrame with short_bond, long_bond, and corresponding TTMs.
    """
    short_df = select_nearest_to_tenor(
        df, short_tenor, datetime_col, bond_id_col, ttm_col, max_deviation
    )
    long_df = select_nearest_to_tenor(
        df, long_tenor, datetime_col, bond_id_col, ttm_col, max_deviation
    )

    result = short_df[[datetime_col]].copy()
    result["short_bond"] = short_df[bond_id_col].values
    result["short_ttm"] = short_df[ttm_col].values
    result["long_bond"] = long_df[bond_id_col].values
    result["long_ttm"] = long_df[ttm_col].values

    return result
