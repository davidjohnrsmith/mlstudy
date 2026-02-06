"""Portfolio interface types for multi-strategy trading.

Defines standard output types for strategies to enable portfolio-level
aggregation, risk management, and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class LegWeight:
    """A single leg in a strategy position.

    Attributes:
        bond_id: Identifier for the instrument.
        weight: Position weight (positive=long, negative=short).
            For a standard fly: front=1, belly=-2, back=1.
        tenor: Optional tenor in years for reference.
    """

    bond_id: str
    weight: float
    tenor: float | None = None

    def __repr__(self) -> str:
        sign = "+" if self.weight >= 0 else ""
        tenor_str = f" ({self.tenor}y)" if self.tenor else ""
        return f"{self.bond_id}: {sign}{self.weight:.2f}{tenor_str}"


@dataclass
class StrategySignal:
    """Standard output from any strategy at a point in time.

    Represents the desired position from a single strategy, which can be
    combined with other strategies at the portfolio level.

    Attributes:
        timestamp: Time of the signal.
        strategy_id: Unique identifier for this strategy (e.g., "fly_2y5y10y_zscore").
        legs: List of (bond_id, weight) pairs representing the position structure.
            Weights define the relative sizing of each leg.
        direction: Desired direction: 1 (long), -1 (short), or 0 (flat).
            Used for discrete signal strategies.
        signal_value: Continuous signal value (e.g., z-score, alpha forecast).
            Can be used for signal-weighted position sizing.
        target_gross_dv01: Optional target gross DV01 for this strategy.
            If None, portfolio-level sizing applies.
        confidence: Optional confidence score [0, 1] for the signal.
        metadata: Optional additional data (e.g., z-score components, entry reason).

    Example:
        >>> signal = StrategySignal(
        ...     timestamp=pd.Timestamp("2024-01-15 08:00"),
        ...     strategy_id="fly_2y5y10y",
        ...     legs=[
        ...         LegWeight("UST_2Y", 1.0, tenor=2.0),
        ...         LegWeight("UST_5Y", -2.0, tenor=5.0),
        ...         LegWeight("UST_10Y", 1.0, tenor=10.0),
        ...     ],
        ...     direction=1,
        ...     signal_value=-2.5,  # z-score indicating fly is cheap
        ...     target_gross_dv01=10000,
        ... )
    """

    timestamp: datetime | pd.Timestamp
    strategy_id: str
    legs: list[LegWeight]
    direction: int = 0  # -1, 0, or 1
    signal_value: float | None = None
    target_gross_dv01: float | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal fields."""
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0, or 1, got {self.direction}")
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def is_flat(self) -> bool:
        """True if strategy wants no position."""
        return self.direction == 0

    @property
    def is_long(self) -> bool:
        """True if strategy wants long position."""
        return self.direction == 1

    @property
    def is_short(self) -> bool:
        """True if strategy wants short position."""
        return self.direction == -1

    @property
    def leg_ids(self) -> list[str]:
        """List of bond IDs in the position."""
        return [leg.bond_id for leg in self.legs]

    @property
    def weights_dict(self) -> dict:
        """Leg weights as {bond_id: weight} dict."""
        return {leg.bond_id: leg.weight for leg in self.legs}

    def scaled_weights(self, scale: float = 1.0) -> dict:
        """Get weights scaled by direction and optional multiplier.

        Args:
            scale: Additional scaling factor.

        Returns:
            Dict of {bond_id: scaled_weight}.
        """
        return {leg.bond_id: leg.weight * self.direction * scale for leg in self.legs}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "strategy_id": self.strategy_id,
            "legs": [(leg.bond_id, leg.weight, leg.tenor) for leg in self.legs],
            "direction": self.direction,
            "signal_value": self.signal_value,
            "target_gross_dv01": self.target_gross_dv01,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategySignal:
        """Create from dictionary."""
        legs = [
            LegWeight(bond_id=leg[0], weight=leg[1], tenor=leg[2] if len(leg) > 2 else None)
            for leg in d["legs"]
        ]
        return cls(
            timestamp=d["timestamp"],
            strategy_id=d["strategy_id"],
            legs=legs,
            direction=d.get("direction", 0),
            signal_value=d.get("signal_value"),
            target_gross_dv01=d.get("target_gross_dv01"),
            confidence=d.get("confidence"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class PortfolioTarget:
    """Aggregated target position across multiple strategies.

    Represents the combined desired position after aggregating signals
    from multiple strategies, applying risk limits, and resolving conflicts.

    Attributes:
        timestamp: Time of the target.
        positions: Dict of {bond_id: target_weight} across all instruments.
        strategy_contributions: Dict of {strategy_id: {bond_id: weight}} showing
            each strategy's contribution to the aggregate.
        total_gross_dv01: Target gross DV01 for the portfolio.
        total_net_dv01: Target net DV01 (sum of signed DV01s).
        total_gross_notional: Target gross notional exposure.
        metadata: Additional portfolio-level data.

    Example:
        >>> target = PortfolioTarget(
        ...     timestamp=pd.Timestamp("2024-01-15 08:00"),
        ...     positions={
        ...         "UST_2Y": 1.5,
        ...         "UST_5Y": -3.0,
        ...         "UST_10Y": 1.5,
        ...     },
        ...     strategy_contributions={
        ...         "fly_2y5y10y": {"UST_2Y": 1.0, "UST_5Y": -2.0, "UST_10Y": 1.0},
        ...         "fly_5y10y30y": {"UST_5Y": -1.0, "UST_10Y": 0.5, "UST_30Y": 0.5},
        ...     },
        ...     total_gross_dv01=20000,
        ... )
    """

    timestamp: datetime | pd.Timestamp
    positions: dict = field(default_factory=dict)
    strategy_contributions: dict = field(default_factory=dict)
    total_gross_dv01: float | None = None
    total_net_dv01: float | None = None
    total_gross_notional: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def instruments(self) -> list[str]:
        """List of all instruments in the portfolio."""
        return list(self.positions.keys())

    @property
    def n_instruments(self) -> int:
        """Number of instruments."""
        return len(self.positions)

    @property
    def n_strategies(self) -> int:
        """Number of contributing strategies."""
        return len(self.strategy_contributions)

    def get_position(self, bond_id: str) -> float:
        """Get target weight for an instrument."""
        return self.positions.get(bond_id, 0.0)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert positions to DataFrame."""
        records = [
            {"bond_id": bond_id, "weight": weight}
            for bond_id, weight in self.positions.items()
        ]
        return pd.DataFrame(records)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "positions": self.positions,
            "strategy_contributions": self.strategy_contributions,
            "total_gross_dv01": self.total_gross_dv01,
            "total_net_dv01": self.total_net_dv01,
            "total_gross_notional": self.total_gross_notional,
            "metadata": self.metadata,
        }


@dataclass
class StrategySignalBatch:
    """Collection of signals from multiple strategies at a single timestamp.

    Attributes:
        timestamp: Common timestamp for all signals.
        signals: List of StrategySignal objects.
    """

    timestamp: datetime | pd.Timestamp
    signals: list[StrategySignal] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    @property
    def strategy_ids(self) -> list[str]:
        """List of strategy IDs in the batch."""
        return [s.strategy_id for s in self.signals]

    def get_signal(self, strategy_id: str) -> StrategySignal | None:
        """Get signal for a specific strategy."""
        for s in self.signals:
            if s.strategy_id == strategy_id:
                return s
        return None

    def add_signal(self, signal: StrategySignal) -> None:
        """Add a signal to the batch."""
        if signal.timestamp != self.timestamp:
            raise ValueError(
                f"Signal timestamp {signal.timestamp} doesn't match "
                f"batch timestamp {self.timestamp}"
            )
        self.signals.append(signal)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per signal."""
        records = []
        for s in self.signals:
            record = {
                "timestamp": s.timestamp,
                "strategy_id": s.strategy_id,
                "direction": s.direction,
                "signal_value": s.signal_value,
                "target_gross_dv01": s.target_gross_dv01,
                "confidence": s.confidence,
                "n_legs": len(s.legs),
            }
            records.append(record)
        return pd.DataFrame(records)


def signals_to_dataframe(signals: list[StrategySignal]) -> pd.DataFrame:
    """Convert list of signals to DataFrame.

    Args:
        signals: List of StrategySignal objects.

    Returns:
        DataFrame with columns: timestamp, strategy_id, direction, signal_value,
        target_gross_dv01, confidence, and leg details.
    """
    records = []
    for s in signals:
        record = {
            "timestamp": s.timestamp,
            "strategy_id": s.strategy_id,
            "direction": s.direction,
            "signal_value": s.signal_value,
            "target_gross_dv01": s.target_gross_dv01,
            "confidence": s.confidence,
            "n_legs": len(s.legs),
            "leg_ids": ",".join(s.leg_ids),
        }
        # Add leg weights
        for i, leg in enumerate(s.legs):
            record[f"leg_{i}_id"] = leg.bond_id
            record[f"leg_{i}_weight"] = leg.weight
            record[f"leg_{i}_tenor"] = leg.tenor
        records.append(record)

    return pd.DataFrame(records)
