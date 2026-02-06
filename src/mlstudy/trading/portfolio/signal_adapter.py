"""Adapters to convert strategy outputs to standard StrategySignal format.

Provides functions to convert existing fly strategy outputs (signal DataFrames,
backtest results) to the StrategySignal interface for portfolio-level use.
"""

from __future__ import annotations

import pandas as pd

from mlstudy.trading.portfolio.portfolio_types import LegWeight, StrategySignal, StrategySignalBatch


def fly_name_to_strategy_id(
    front: float,
    belly: float,
    back: float,
    prefix: str = "fly",
) -> str:
    """Generate strategy ID from fly tenors.

    Args:
        front: Front leg tenor.
        belly: Belly leg tenor.
        back: Back leg tenor.
        prefix: Prefix for strategy ID.

    Returns:
        Strategy ID like "fly_2y5y10y".
    """

    def tenor_str(t: float) -> str:
        if t < 1:
            return f"{int(t * 12)}m"
        elif t == int(t):
            return f"{int(t)}y"
        else:
            return f"{t}y"

    return f"{prefix}_{tenor_str(front)}{tenor_str(belly)}{tenor_str(back)}"


def create_fly_legs(
    front_id: str,
    belly_id: str,
    back_id: str,
    front_weight: float = 1.0,
    belly_weight: float = -2.0,
    back_weight: float = 1.0,
    front_tenor: float | None = None,
    belly_tenor: float | None = None,
    back_tenor: float | None = None,
) -> list[LegWeight]:
    """Create list of LegWeight objects for a fly.

    Args:
        front_id: Front leg bond ID.
        belly_id: Belly leg bond ID.
        back_id: Back leg bond ID.
        front_weight: Front leg weight (default: 1.0).
        belly_weight: Belly leg weight (default: -2.0).
        back_weight: Back leg weight (default: 1.0).
        front_tenor: Front leg tenor in years.
        belly_tenor: Belly leg tenor in years.
        back_tenor: Back leg tenor in years.

    Returns:
        List of LegWeight objects.
    """
    return [
        LegWeight(bond_id=front_id, weight=front_weight, tenor=front_tenor),
        LegWeight(bond_id=belly_id, weight=belly_weight, tenor=belly_tenor),
        LegWeight(bond_id=back_id, weight=back_weight, tenor=back_tenor),
    ]


def signal_df_to_strategy_signals(
    signal_df: pd.DataFrame,
    front_id: str,
    belly_id: str,
    back_id: str,
    strategy_id: str | None = None,
    tenors: tuple[float, float, float] | None = None,
    weights: tuple[float, float, float] | None = None,
    datetime_col: str = "datetime",
    signal_col: str = "signal",
    zscore_col: str = "zscore",
    strength_col: str | None = "strength",
    target_gross_dv01: float | None = None,
) -> list[StrategySignal]:
    """Convert signal DataFrame to list of StrategySignal objects.

    Args:
        signal_df: DataFrame from build_signal_dataframe.
        front_id: Front leg bond ID.
        belly_id: Belly leg bond ID.
        back_id: Back leg bond ID.
        strategy_id: Strategy identifier. If None, generated from tenors.
        tenors: (front, belly, back) tenors for strategy ID and leg info.
        weights: (front, belly, back) weights. Default: (1, -2, 1).
        datetime_col: Datetime column name.
        signal_col: Signal column name.
        zscore_col: Z-score column name.
        strength_col: Strength column name (None to skip).
        target_gross_dv01: Target gross DV01 for all signals.

    Returns:
        List of StrategySignal objects, one per row.

    Example:
        >>> signals = signal_df_to_strategy_signals(
        ...     signal_df,
        ...     front_id="UST_2Y",
        ...     belly_id="UST_5Y",
        ...     back_id="UST_10Y",
        ...     tenors=(2, 5, 10),
        ... )
    """
    if weights is None:
        weights = (1.0, -2.0, 1.0)

    if strategy_id is None:
        if tenors is not None:
            strategy_id = fly_name_to_strategy_id(tenors[0], tenors[1], tenors[2])
        else:
            strategy_id = f"fly_{front_id}_{belly_id}_{back_id}"

    # Create legs
    front_tenor = tenors[0] if tenors else None
    belly_tenor = tenors[1] if tenors else None
    back_tenor = tenors[2] if tenors else None

    legs = create_fly_legs(
        front_id=front_id,
        belly_id=belly_id,
        back_id=back_id,
        front_weight=weights[0],
        belly_weight=weights[1],
        back_weight=weights[2],
        front_tenor=front_tenor,
        belly_tenor=belly_tenor,
        back_tenor=back_tenor,
    )

    # Convert each row to StrategySignal
    signals = []
    for _, row in signal_df.iterrows():
        direction = int(row[signal_col])
        zscore = row.get(zscore_col)
        confidence = row.get(strength_col) if strength_col else None

        signal = StrategySignal(
            timestamp=row[datetime_col],
            strategy_id=strategy_id,
            legs=legs,
            direction=direction,
            signal_value=zscore if pd.notna(zscore) else None,
            target_gross_dv01=target_gross_dv01,
            confidence=confidence if pd.notna(confidence) else None,
            metadata={"zscore": zscore},
        )
        signals.append(signal)

    return signals


def pnl_df_to_strategy_signals(
    pnl_df: pd.DataFrame,
    strategy_id: str | None = None,
    tenors: tuple[float, float, float] | None = None,
    datetime_col: str = "datetime",
    position_col: str = "position",
    signal_col: str = "signal",
    zscore_col: str = "zscore",
    front_id_col: str = "front_id",
    belly_id_col: str = "belly_id",
    back_id_col: str = "back_id",
    target_gross_dv01: float | None = None,
) -> list[StrategySignal]:
    """Convert backtest pnl_df to list of StrategySignal objects.

    Works with output from backtest_fly_intraday or backtest_fly.

    Args:
        pnl_df: DataFrame from backtest engine.
        strategy_id: Strategy identifier.
        tenors: (front, belly, back) tenors.
        datetime_col: Datetime column name.
        position_col: Position column name.
        signal_col: Raw signal column name.
        zscore_col: Z-score column name.
        front_id_col: Front leg ID column name.
        belly_id_col: Belly leg ID column name.
        back_id_col: Back leg ID column name.
        target_gross_dv01: Target gross DV01.

    Returns:
        List of StrategySignal objects.
    """
    if strategy_id is None:
        if tenors is not None:
            strategy_id = fly_name_to_strategy_id(tenors[0], tenors[1], tenors[2])
        else:
            strategy_id = "fly_strategy"

    signals = []

    for _, row in pnl_df.iterrows():
        # Get leg IDs (may vary by row for intraday with daily selection)
        front_id = row.get(front_id_col, "front")
        belly_id = row.get(belly_id_col, "belly")
        back_id = row.get(back_id_col, "back")

        # Get weights if available
        w_front = row.get("w_front", 1.0)
        w_belly = row.get("w_belly", -2.0)
        w_back = row.get("w_back", 1.0)

        front_tenor = tenors[0] if tenors else None
        belly_tenor = tenors[1] if tenors else None
        back_tenor = tenors[2] if tenors else None

        legs = create_fly_legs(
            front_id=str(front_id) if pd.notna(front_id) else "front",
            belly_id=str(belly_id) if pd.notna(belly_id) else "belly",
            back_id=str(back_id) if pd.notna(back_id) else "back",
            front_weight=w_front if pd.notna(w_front) else 1.0,
            belly_weight=w_belly if pd.notna(w_belly) else -2.0,
            back_weight=w_back if pd.notna(w_back) else 1.0,
            front_tenor=front_tenor,
            belly_tenor=belly_tenor,
            back_tenor=back_tenor,
        )

        # Get direction from position (lagged signal) or signal
        direction = int(row.get(position_col, row.get(signal_col, 0)))
        zscore = row.get(zscore_col)

        signal = StrategySignal(
            timestamp=row[datetime_col],
            strategy_id=strategy_id,
            legs=legs,
            direction=direction,
            signal_value=zscore if pd.notna(zscore) else None,
            target_gross_dv01=target_gross_dv01,
            metadata={
                "zscore": zscore,
                "gross_dv01": row.get("gross_dv01"),
                "net_dv01": row.get("net_dv01"),
            },
        )
        signals.append(signal)

    return signals


def intraday_result_to_signals(
    result,  # IntradayBacktestResult
    strategy_id: str | None = None,
    tenors: tuple[float, float, float] | None = None,
    target_gross_dv01: float | None = None,
) -> list[StrategySignal]:
    """Convert IntradayBacktestResult to StrategySignal list.

    Args:
        result: IntradayBacktestResult from backtest_fly_intraday.
        strategy_id: Strategy identifier.
        tenors: (front, belly, back) tenors.
        target_gross_dv01: Target gross DV01.

    Returns:
        List of StrategySignal objects.
    """
    return pnl_df_to_strategy_signals(
        result.pnl_df,
        strategy_id=strategy_id,
        tenors=tenors,
        target_gross_dv01=target_gross_dv01 or result.config.dv01_target,
    )


def batch_signals_by_timestamp(
    signals: list[StrategySignal],
) -> list[StrategySignalBatch]:
    """Group signals by timestamp into batches.

    Args:
        signals: List of StrategySignal from potentially multiple strategies.

    Returns:
        List of StrategySignalBatch, one per unique timestamp.
    """
    # Group by timestamp
    by_timestamp: dict = {}
    for signal in signals:
        ts = signal.timestamp
        if ts not in by_timestamp:
            by_timestamp[ts] = []
        by_timestamp[ts].append(signal)

    # Create batches
    batches = []
    for ts in sorted(by_timestamp.keys()):
        batch = StrategySignalBatch(timestamp=ts, signals=by_timestamp[ts])
        batches.append(batch)

    return batches


def filter_signals_by_strategy(
    signals: list[StrategySignal],
    strategy_ids: list[str],
) -> list[StrategySignal]:
    """Filter signals to specific strategies.

    Args:
        signals: List of StrategySignal.
        strategy_ids: List of strategy IDs to keep.

    Returns:
        Filtered list of StrategySignal.
    """
    return [s for s in signals if s.strategy_id in strategy_ids]


def filter_signals_by_direction(
    signals: list[StrategySignal],
    direction: int,
) -> list[StrategySignal]:
    """Filter signals by direction.

    Args:
        signals: List of StrategySignal.
        direction: Direction to keep (-1, 0, or 1).

    Returns:
        Filtered list of StrategySignal.
    """
    return [s for s in signals if s.direction == direction]


def get_active_signals(
    signals: list[StrategySignal],
) -> list[StrategySignal]:
    """Get non-flat signals (direction != 0).

    Args:
        signals: List of StrategySignal.

    Returns:
        List of signals with direction != 0.
    """
    return [s for s in signals if not s.is_flat]
