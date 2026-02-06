"""Butterfly (fly) trade construction for yields and PnL.

Constructs butterfly spreads from panel data and leg selections,
supporting both equal and DV01-neutral weighting.

Typical butterfly: short 2Y, long 5Y, short 10Y
  - Yield fly = yield_2Y - 2*yield_5Y + yield_10Y
  - DV01-neutral: weights adjusted so net DV01 = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class FlyWeights:
    """Butterfly trade weights.

    Convention: positive = long, negative = short.
    Standard fly: front=+1, belly=-2, back=+1 (long wings, short belly).
    """

    front: float
    belly: float
    back: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return as (front, belly, back) tuple."""
        return (self.front, self.belly, self.back)

    def as_array(self) -> NDArray:
        """Return as numpy array."""
        return np.array([self.front, self.belly, self.back])

    @classmethod
    def equal(cls) -> FlyWeights:
        """Standard equal-weighted fly (1, -2, 1)."""
        return cls(front=1.0, belly=-2.0, back=1.0)

    @classmethod
    def from_dv01(
        cls,
        dv01_front: float,
        dv01_belly: float,
        dv01_back: float,
        belly_weight: float = -1.0,
    ) -> FlyWeights:
        """Compute DV01-neutral weights.

        Solves for weights such that:
        - w_front * DV01_front + w_belly * DV01_belly + w_back * DV01_back = 0
        - w_belly = belly_weight (typically -1 or -2)
        - w_front = w_back (symmetric wings)

        Args:
            dv01_front: DV01 of front leg.
            dv01_belly: DV01 of belly leg.
            dv01_back: DV01 of back leg.
            belly_weight: Fixed belly weight (negative for short).

        Returns:
            FlyWeights with DV01-neutral configuration.
        """
        # With w_front = w_back = w, we solve:
        # w * (DV01_front + DV01_back) + belly_weight * DV01_belly = 0
        # w = -belly_weight * DV01_belly / (DV01_front + DV01_back)
        wing_dv01_sum = dv01_front + dv01_back
        if abs(wing_dv01_sum) < 1e-10:
            # Degenerate case
            return cls.equal()

        wing_weight = -belly_weight * dv01_belly / wing_dv01_sum

        return cls(front=wing_weight, belly=belly_weight, back=wing_weight)


def compute_fly_value(
    front: float | NDArray,
    belly: float | NDArray,
    back: float | NDArray,
    weights: tuple[float, float, float] | FlyWeights = (1.0, -2.0, 1.0),
) -> float | NDArray:
    """Compute butterfly spread value.

    Args:
        front: Front leg value(s) (e.g., 2Y yield).
        belly: Belly leg value(s) (e.g., 5Y yield).
        back: Back leg value(s) (e.g., 10Y yield).
        weights: (w_front, w_belly, w_back) weights or FlyWeights object.

    Returns:
        Fly value = w_front*front + w_belly*belly + w_back*back.

    Example:
        >>> fly_yield = compute_fly_value(yield_2y, yield_5y, yield_10y)
        >>> # fly_yield = yield_2y - 2*yield_5y + yield_10y
    """
    w = weights.as_tuple() if isinstance(weights, FlyWeights) else weights
    return w[0] * front + w[1] * belly + w[2] * back


def build_fly_timeseries(
    df: pd.DataFrame,
    legs_table: pd.DataFrame,
    value_col: str = "yield",
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    weights: tuple[float, float, float] | None = None,
    use_dv01_weights: bool = False,
    dv01_col: str = "dv01",
) -> pd.DataFrame:
    """Build butterfly spread time series from panel data and legs selection.

    Args:
        df: Panel DataFrame with datetime, bond_id, and value columns.
        legs_table: DataFrame from select_fly_legs with columns:
            datetime, short_bond (front), belly_bond, long_bond (back).
        value_col: Value column to compute fly for (e.g., "yield", "price").
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        weights: Fixed weights (w_front, w_belly, w_back). If None, uses (1, -2, 1).
        use_dv01_weights: If True, compute DV01-neutral weights per date.
        dv01_col: DV01 column name (required if use_dv01_weights=True).

    Returns:
        DataFrame with columns:
        - datetime: Date
        - fly_value: Butterfly spread value
        - front_bond, belly_bond, back_bond: Selected bond IDs
        - front_value, belly_value, back_value: Leg values
        - w_front, w_belly, w_back: Weights used

    Example:
        >>> legs = select_fly_legs(df, tenors=(2, 5, 10))
        >>> fly_df = build_fly_timeseries(df, legs, value_col="yield")
        >>> # fly_df["fly_value"] is the butterfly yield spread
    """
    # Create lookup for values
    value_lookup = df.set_index([datetime_col, bond_id_col])[value_col]

    if use_dv01_weights:
        dv01_lookup = df.set_index([datetime_col, bond_id_col])[dv01_col]

    results = []

    for _, row in legs_table.iterrows():
        dt = row[datetime_col]
        front_bond = row.get("short_bond", row.get("front_bond"))
        belly_bond = row["belly_bond"]
        back_bond = row.get("long_bond", row.get("back_bond"))

        # Get values
        try:
            front_val = value_lookup.loc[(dt, front_bond)]
            belly_val = value_lookup.loc[(dt, belly_bond)]
            back_val = value_lookup.loc[(dt, back_bond)]
        except KeyError:
            # Missing data
            results.append({
                datetime_col: dt,
                "fly_value": np.nan,
                "front_bond": front_bond,
                "belly_bond": belly_bond,
                "back_bond": back_bond,
                "front_value": np.nan,
                "belly_value": np.nan,
                "back_value": np.nan,
                "w_front": np.nan,
                "w_belly": np.nan,
                "w_back": np.nan,
            })
            continue

        # Determine weights
        if use_dv01_weights:
            try:
                dv01_front = dv01_lookup.loc[(dt, front_bond)]
                dv01_belly = dv01_lookup.loc[(dt, belly_bond)]
                dv01_back = dv01_lookup.loc[(dt, back_bond)]
                fly_weights = FlyWeights.from_dv01(dv01_front, dv01_belly, dv01_back)
            except KeyError:
                fly_weights = FlyWeights.equal()
        elif weights is not None:
            fly_weights = FlyWeights(*weights)
        else:
            fly_weights = FlyWeights.equal()

        # Compute fly value
        fly_val = compute_fly_value(front_val, belly_val, back_val, fly_weights)

        results.append({
            datetime_col: dt,
            "fly_value": fly_val,
            "front_bond": front_bond,
            "belly_bond": belly_bond,
            "back_bond": back_bond,
            "front_value": front_val,
            "belly_value": belly_val,
            "back_value": back_val,
            "w_front": fly_weights.front,
            "w_belly": fly_weights.belly,
            "w_back": fly_weights.back,
        })

    return pd.DataFrame(results)


def build_fly_legs_panel(
    df: pd.DataFrame,
    legs_table: pd.DataFrame,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    value_cols: list[str] | None = None,
    use_dv01_weights: bool = False,
    dv01_col: str = "dv01",
    fixed_weights: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """Build per-date per-leg panel with weights and values.

    Creates a long-format DataFrame with one row per leg per date,
    including weights and all requested value columns.

    Args:
        df: Panel DataFrame with datetime, bond_id, and value columns.
        legs_table: DataFrame from select_fly_legs.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        value_cols: Value columns to include. Default: ["yield", "price", "dv01"].
        use_dv01_weights: If True, compute DV01-neutral weights.
        dv01_col: DV01 column for weight computation.
        fixed_weights: Fixed weights if not using DV01 weights.

    Returns:
        DataFrame with columns:
        - datetime: Date
        - bond_id: Bond identifier
        - leg: "front", "belly", or "back"
        - weight: Leg weight
        - yield, price, dv01, ...: Value columns

    Example:
        >>> legs = select_fly_legs(df, tenors=(2, 5, 10))
        >>> legs_panel = build_fly_legs_panel(df, legs)
        >>> # legs_panel has 3 rows per date (front, belly, back)
    """
    if value_cols is None:
        # Default columns
        potential_cols = ["yield", "price", "dv01"]
        value_cols = [c for c in potential_cols if c in df.columns]

    # Create lookup
    lookups = {col: df.set_index([datetime_col, bond_id_col])[col] for col in value_cols}

    if use_dv01_weights and dv01_col in df.columns:
        dv01_lookup = df.set_index([datetime_col, bond_id_col])[dv01_col]
    else:
        dv01_lookup = None

    records = []

    for _, row in legs_table.iterrows():
        dt = row[datetime_col]
        front_bond = row.get("short_bond", row.get("front_bond"))
        belly_bond = row["belly_bond"]
        back_bond = row.get("long_bond", row.get("back_bond"))

        bonds = [front_bond, belly_bond, back_bond]
        leg_names = ["front", "belly", "back"]

        # Compute weights
        if use_dv01_weights and dv01_lookup is not None:
            try:
                dv01_front = dv01_lookup.loc[(dt, front_bond)]
                dv01_belly = dv01_lookup.loc[(dt, belly_bond)]
                dv01_back = dv01_lookup.loc[(dt, back_bond)]
                fly_weights = FlyWeights.from_dv01(dv01_front, dv01_belly, dv01_back)
            except KeyError:
                fly_weights = FlyWeights.equal()
        elif fixed_weights is not None:
            fly_weights = FlyWeights(*fixed_weights)
        else:
            fly_weights = FlyWeights.equal()

        weights = [fly_weights.front, fly_weights.belly, fly_weights.back]

        for bond_id, leg_name, weight in zip(bonds, leg_names, weights):  # noqa: B905
            record = {
                datetime_col: dt,
                bond_id_col: bond_id,
                "leg": leg_name,
                "weight": weight,
            }

            # Add values
            for col in value_cols:
                try:
                    record[col] = lookups[col].loc[(dt, bond_id)]
                except KeyError:
                    record[col] = np.nan

            records.append(record)

    return pd.DataFrame(records)


@dataclass
class FlyResult:
    """Complete butterfly trade result.

    Attributes:
        fly_df: Time series with fly values and metadata.
        legs_df: Per-date per-leg panel with weights and values.
    """

    fly_df: pd.DataFrame
    legs_df: pd.DataFrame

    def get_fly_yield(self) -> pd.Series:
        """Get fly yield time series."""
        return self.fly_df.set_index("datetime")["fly_value"]

    def get_leg_values(self, leg: str, value_col: str) -> pd.Series:
        """Get values for a specific leg."""
        mask = self.legs_df["leg"] == leg
        return self.legs_df[mask].set_index("datetime")[value_col]

    def compute_weighted_pnl(
        self,
        price_col: str = "price",
        notional: float = 1.0,
    ) -> pd.DataFrame:
        """Compute PnL from price changes.

        Returns DataFrame with daily PnL per leg and total.
        """
        legs_df = self.legs_df.copy()

        # Compute weighted price
        legs_df["weighted_price"] = legs_df["weight"] * legs_df[price_col] * notional

        # Pivot to get per-leg prices
        pnl_df = legs_df.pivot(
            index="datetime",
            columns="leg",
            values="weighted_price",
        )

        # Compute changes (PnL)
        pnl_df = pnl_df.diff()

        # Total PnL
        pnl_df["total"] = pnl_df.sum(axis=1)

        return pnl_df

    def get_net_dv01(self) -> pd.Series:
        """Compute net DV01 of the fly per date."""
        legs_df = self.legs_df.copy()
        legs_df["weighted_dv01"] = legs_df["weight"] * legs_df["dv01"]
        return legs_df.groupby("datetime")["weighted_dv01"].sum()


def build_fly(
    df: pd.DataFrame,
    legs_table: pd.DataFrame,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    value_cols: list[str] | None = None,
    use_dv01_weights: bool = False,
    dv01_col: str = "dv01",
    fixed_weights: tuple[float, float, float] | None = None,
    yield_col: str = "yield",
) -> FlyResult:
    """Build complete butterfly trade from panel data.

    Convenience function that returns both fly time series and legs panel.

    Args:
        df: Panel DataFrame.
        legs_table: DataFrame from select_fly_legs.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        value_cols: Value columns for legs panel.
        use_dv01_weights: If True, compute DV01-neutral weights.
        dv01_col: DV01 column for weight computation.
        fixed_weights: Fixed weights if not using DV01 weights.
        yield_col: Yield column for fly value computation.

    Returns:
        FlyResult with fly_df and legs_df.

    Example:
        >>> legs = select_fly_legs(df, tenors=(2, 5, 10))
        >>> result = build_fly(df, legs, use_dv01_weights=True)
        >>> fly_yield = result.get_fly_yield()
        >>> net_dv01 = result.get_net_dv01()  # Should be ~0
    """
    # Build fly time series
    weights = fixed_weights if fixed_weights else None
    fly_df = build_fly_timeseries(
        df=df,
        legs_table=legs_table,
        value_col=yield_col,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        weights=weights,
        use_dv01_weights=use_dv01_weights,
        dv01_col=dv01_col,
    )

    # Build legs panel
    legs_df = build_fly_legs_panel(
        df=df,
        legs_table=legs_table,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        value_cols=value_cols,
        use_dv01_weights=use_dv01_weights,
        dv01_col=dv01_col,
        fixed_weights=fixed_weights,
    )

    return FlyResult(fly_df=fly_df, legs_df=legs_df)


def compute_fly_carry(
    fly_result: FlyResult,
    yield_col: str = "yield",
    roll_days: int = 30,
) -> pd.Series:
    """Compute fly carry (roll-down) from yield changes.

    Approximates carry as the change in fly yield over roll_days.

    Args:
        fly_result: FlyResult from build_fly.
        yield_col: Yield column name.
        roll_days: Days for roll-down calculation.

    Returns:
        Series with fly carry values.
    """
    fly_yield = fly_result.fly_df.set_index("datetime")["fly_value"]
    carry = fly_yield.diff(roll_days)
    carry.name = "fly_carry"
    return carry


def compute_fly_richness(
    fly_result: FlyResult,
    lookback: int = 20,
) -> pd.Series:
    """Compute fly richness (z-score) vs recent history.

    Args:
        fly_result: FlyResult from build_fly.
        lookback: Lookback window for z-score.

    Returns:
        Series with z-score values.
    """
    fly_yield = fly_result.fly_df.set_index("datetime")["fly_value"]
    mean = fly_yield.rolling(lookback).mean()
    std = fly_yield.rolling(lookback).std()
    zscore = (fly_yield - mean) / std
    zscore.name = "fly_zscore"
    return zscore
