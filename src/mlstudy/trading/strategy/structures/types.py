"""
Shared types for multi-leg trading structures (fly, switch, curve trades, CTD-aware packages).

This module intentionally stays dependency-light and does not encode instrument-specific
pricing logic. It focuses on representing:
- leg identifiers
- leg time series
- structure specs (what to build)
- time-varying weights
- the resulting constructed structure series
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import pandas as pd


class StructureKind(str, Enum):
    """High-level structure family.

    Note: This is descriptive metadata; construction logic should not branch heavily on it.
    """

    FLY = "fly"  # 3-leg
    SWITCH = "switch"  # 2-leg
    SPREAD = "spread"  # general 2-leg
    CURVE = "curve"  # curve trade (often 2-leg)
    CUSTOM = "custom"  # any multi-leg linear combination


class LegRole(str, Enum):
    """Role labels used by common structures (optional)."""

    FRONT = "front"
    BELLY = "belly"
    BACK = "back"
    NEAR = "near"
    FAR = "far"
    LEG1 = "leg1"
    LEG2 = "leg2"
    LEG3 = "leg3"
    LEG4 = "leg4"


@dataclass(frozen=True, slots=True)
class LegId:
    """Identifier for a tradable leg.

    Keep this generic: a leg can be a cash bond, swap tenor, future contract, CTD-deliverable,
    or any other instrument. Use `meta` to store instrument-specific fields.

    Recommended fields in meta (examples):
      - "asset_class": "bond" | "swap" | "future"
      - "ticker": "RXH6"
      - "isin": "DE000..."
      - "tenor": "5Y"
      - "currency": "EUR"
      - "curve": "eureur"
      - "ctd_key": "RX | 2026-03 | CTD=DE..."
    """

    name: str
    role: LegRole | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LegSeries:
    """Time series for one leg.

    `value` is the raw series you construct on (yield, spread, price, etc.).
    The choice of what `value` represents is up to the caller/spec.

    Optional risk series can be attached (dv01, pvbp) if you want execution overlays.
    """

    leg: LegId
    value: pd.Series

    dv01: pd.Series | None = None  # per-unit notional risk; indexed like value
    meta: Mapping[str, Any] = field(default_factory=dict)

    def aligned_df(self) -> pd.DataFrame:
        """Return a DataFrame with aligned columns for value and dv01 (if present)."""
        df = pd.DataFrame({"value": self.value})
        if self.dv01 is not None:
            df["dv01"] = self.dv01
        return df


@dataclass(frozen=True, slots=True)
class StructureSpec:
    """Specification describing what structure to build.

    `legs` is an ordered tuple defining leg order for weights.
    For a fly, typical order is (front, belly, back).
    For a switch, typical order is (near, far) or (leg1, leg2).

    `kind` is descriptive and helps downstream naming.
    `name` should be a human-friendly identifier, e.g. "2-5-10" or "RXU6-RXH7 switch".

    `meta` can store:
      - "currency", "curve", "universe_id"
      - "construction_method": "levels_residual" | "dv01_balanced" | ...
      - "constraints": {...}
    """

    kind: StructureKind
    name: str
    legs: tuple[LegId, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def n_legs(self) -> int:
        return len(self.legs)


@dataclass(slots=True)
class WeightsFrame:
    """Time-varying weights for a structure.

    `weights` must be a DataFrame indexed by datetime, with one column per leg,
    using a stable column naming convention.

    Recommended convention:
      - columns are leg names OR roles:
          e.g. ["front", "belly", "back"] or ["leg1","leg2","leg3"]
      - include optional diagnostic columns:
          e.g. ["objective_value","net_dv01","beta_front","beta_back"]

    `leg_columns` explicitly tells which columns correspond to legs (in order).
    """

    weights: pd.DataFrame
    leg_columns: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Basic validation of weights frame structure."""
        missing = [c for c in self.leg_columns if c not in self.weights.columns]
        if missing:
            raise ValueError(f"WeightsFrame missing leg columns: {missing}")
        if not isinstance(self.weights.index, pd.DatetimeIndex):
            # Allow RangeIndex in tests, but prefer DateTimeIndex in production
            return

    def leg_weights(self) -> pd.DataFrame:
        """Return only the leg weight columns."""
        return self.weights.loc[:, list(self.leg_columns)]


@dataclass(slots=True)
class StructureSeries:
    """Result of constructing a structure series.

    `value` is the constructed time series (fly level, spread level, residual, etc.).
    `weights_frame` provides the time-varying weights used (or constant weights repeated).
    `spec` is the structure spec.

    `meta` can include:
      - "units": "yield" | "price" | "spread"
      - "construction_method": ...
      - "notes": ...
    """

    spec: StructureSpec
    value: pd.Series
    weights_frame: WeightsFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def aligned_frame(
        self,
        leg_values: Mapping[str, pd.Series] | None = None,
        include_weights: bool = True,
    ) -> pd.DataFrame:
        """Build a convenient DataFrame for analysis/plotting.

        Args:
            leg_values: Optional mapping from column name -> series of leg values
                (e.g., yields for each leg). If provided, they will be aligned to `value`.
            include_weights: If True, join weights columns.

        Returns:
            DataFrame with at least ["value"], plus optional leg values and weights.
        """
        df = pd.DataFrame({"value": self.value})
        if leg_values:
            for k, s in leg_values.items():
                df[k] = s.reindex(df.index)
        if include_weights:
            w = self.weights_frame.weights.reindex(df.index)
            df = df.join(w, how="left")
        return df


def align_leg_series(
    legs: Sequence[LegSeries],
    *,
    join: str = "inner",
    dropna: bool = True,
) -> pd.DataFrame:
    """Align multiple LegSeries into a single DataFrame.

    Returns:
        DataFrame with one column per leg.name containing the `value` series.
    """
    if not legs:
        raise ValueError("No legs provided")

    data = {}
    for ls in legs:
        col = ls.leg.role.value if ls.leg.role is not None else ls.leg.name
        data[col] = ls.value

    df = pd.concat(data, axis=1, join=join)

    if dropna:
        df = df.dropna()

    return df
