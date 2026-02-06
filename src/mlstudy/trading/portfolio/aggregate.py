"""Aggregate multiple strategy signals into bond-level portfolio targets.

Combines signals from multiple strategies, applies sizing rules and constraints,
and produces a single set of bond-level DV01 targets for execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from mlstudy.trading.portfolio.portfolio_types import PortfolioTarget, StrategySignal


@dataclass
class SizingRules:
    """Rules for sizing and aggregating strategy signals.

    Attributes:
        strategy_weights: How to weight strategies. Options:
            - "equal": Equal weight across all strategies (default)
            - dict: {strategy_id: weight} for fixed weights
            - callable: fn(strategy_id, timestamp) -> weight for time-varying
        default_target_dv01: Default target gross DV01 per strategy if not specified
            in the signal. Used when signal.target_gross_dv01 is None.
        gross_dv01_budget: Maximum total gross DV01 (sum of absolute DV01s).
            Portfolio is scaled down if exceeded.
        net_dv01_target: Target net DV01 (sum of signed DV01s). If specified,
            positions are adjusted to achieve this target.
        per_bond_dv01_cap: Maximum absolute DV01 per bond. Individual positions
            are clipped before other constraints.
        min_signal_confidence: Minimum confidence to include a signal (0-1).
            Signals with lower confidence are excluded.

    Example:
        >>> rules = SizingRules(
        ...     strategy_weights={"fly_2y5y10y": 0.6, "fly_5y10y30y": 0.4},
        ...     gross_dv01_budget=50000,
        ...     per_bond_dv01_cap=20000,
        ... )
    """

    strategy_weights: str | dict | Callable = "equal"
    default_target_dv01: float = 10000.0
    gross_dv01_budget: float | None = None
    net_dv01_target: float | None = None
    per_bond_dv01_cap: float | None = None
    min_signal_confidence: float | None = None


@dataclass
class AggregationResult:
    """Result of aggregating signals into bond-level targets.

    Attributes:
        target_dv01: Series indexed by bond_id with target DV01 values.
        strategy_contributions: Dict of {strategy_id: {bond_id: dv01}} showing
            each strategy's contribution before constraints.
        gross_dv01: Total gross DV01 (sum of absolute values).
        net_dv01: Total net DV01 (sum of signed values).
        scale_factor: Factor applied to meet gross_dv01_budget (1.0 if no scaling).
        constraints_applied: List of constraints that were binding.
        excluded_strategies: Strategies excluded due to confidence filter.
    """

    target_dv01: pd.Series
    strategy_contributions: dict = field(default_factory=dict)
    gross_dv01: float = 0.0
    net_dv01: float = 0.0
    scale_factor: float = 1.0
    constraints_applied: list = field(default_factory=list)
    excluded_strategies: list = field(default_factory=list)

    @property
    def n_bonds(self) -> int:
        """Number of bonds with non-zero targets."""
        return (self.target_dv01.abs() > 1e-10).sum()

    @property
    def n_strategies(self) -> int:
        """Number of contributing strategies."""
        return len(self.strategy_contributions)

    def to_portfolio_target(self, timestamp: pd.Timestamp) -> PortfolioTarget:
        """Convert to PortfolioTarget."""
        return PortfolioTarget(
            timestamp=timestamp,
            positions=self.target_dv01.to_dict(),
            strategy_contributions=self.strategy_contributions,
            total_gross_dv01=self.gross_dv01,
            total_net_dv01=self.net_dv01,
            metadata={
                "scale_factor": self.scale_factor,
                "constraints_applied": self.constraints_applied,
            },
        )


def _get_strategy_weight(
    strategy_id: str,
    timestamp: pd.Timestamp | None,
    weights: str | dict | Callable,
    n_strategies: int,
) -> float:
    """Get weight for a strategy.

    Args:
        strategy_id: Strategy identifier.
        timestamp: Signal timestamp (for time-varying weights).
        weights: Weight specification.
        n_strategies: Total number of strategies (for equal weighting).

    Returns:
        Weight for the strategy.
    """
    if weights == "equal":
        return 1.0 / n_strategies if n_strategies > 0 else 1.0
    elif isinstance(weights, dict):
        return weights.get(strategy_id, 0.0)
    elif callable(weights):
        return weights(strategy_id, timestamp)
    else:
        raise ValueError(f"Invalid weights specification: {weights}")


def _compute_signal_dv01(
    signal: StrategySignal,
    default_target_dv01: float,
) -> dict:
    """Compute DV01 contribution per bond from a signal.

    Args:
        signal: Strategy signal.
        default_target_dv01: Default target if not in signal.

    Returns:
        Dict of {bond_id: dv01} for this signal.
    """
    if signal.is_flat:
        return {}

    target_dv01 = signal.target_gross_dv01 or default_target_dv01

    # Get scaled weights (direction * leg weights)
    # For a fly with weights [1, -2, 1] and direction=1:
    #   scaled = {front: 1, belly: -2, back: 1}
    # For direction=-1:
    #   scaled = {front: -1, belly: 2, back: -1}
    scaled_weights = signal.scaled_weights()

    # Normalize weights to sum to target_dv01 in gross terms
    # Gross = sum of absolute weights
    gross_weight = sum(abs(w) for w in scaled_weights.values())
    if gross_weight == 0:
        return {}

    # Scale each leg's DV01 so gross DV01 = target
    scale = target_dv01 / gross_weight

    return {bond_id: w * scale for bond_id, w in scaled_weights.items()}


def _apply_per_bond_cap(
    bond_dv01: pd.Series,
    cap: float,
) -> tuple[pd.Series, bool]:
    """Apply per-bond DV01 cap.

    Args:
        bond_dv01: Series of DV01 by bond.
        cap: Maximum absolute DV01 per bond.

    Returns:
        Tuple of (capped Series, whether cap was binding).
    """
    capped = bond_dv01.clip(lower=-cap, upper=cap)
    was_binding = (bond_dv01.abs() > cap).any()
    return capped, was_binding


def _apply_gross_budget(
    bond_dv01: pd.Series,
    budget: float,
) -> tuple[pd.Series, float, bool]:
    """Scale down to meet gross DV01 budget.

    Args:
        bond_dv01: Series of DV01 by bond.
        budget: Maximum gross DV01.

    Returns:
        Tuple of (scaled Series, scale factor, whether budget was binding).
    """
    gross = bond_dv01.abs().sum()
    if gross <= budget:
        return bond_dv01, 1.0, False

    scale = budget / gross
    return bond_dv01 * scale, scale, True


def _apply_net_target(
    bond_dv01: pd.Series,
    net_target: float,
) -> tuple[pd.Series, bool]:
    """Adjust positions to achieve net DV01 target.

    Applies a uniform shift to all positions to hit the target.
    This is a simple approach; more sophisticated methods could
    minimize tracking error or respect other constraints.

    Args:
        bond_dv01: Series of DV01 by bond.
        net_target: Target net DV01.

    Returns:
        Tuple of (adjusted Series, whether adjustment was made).
    """
    current_net = bond_dv01.sum()
    diff = net_target - current_net

    if abs(diff) < 1e-10:
        return bond_dv01, False

    # Distribute the adjustment across all bonds proportionally
    # to their absolute weights
    abs_weights = bond_dv01.abs()
    total_abs = abs_weights.sum()

    if total_abs == 0:
        return bond_dv01, False

    # Add adjustment proportional to each bond's share
    adjustment = (abs_weights / total_abs) * diff
    return bond_dv01 + adjustment, True


def signals_to_bond_targets(
    signals: list[StrategySignal],
    sizing_rules: SizingRules | None = None,
) -> AggregationResult:
    """Aggregate multiple strategy signals into bond-level DV01 targets.

    Combines signals from multiple strategies by:
    1. Computing each strategy's bond-level DV01 contribution
    2. Applying strategy weights
    3. Summing across strategies (netting happens naturally)
    4. Applying constraints (per-bond cap, gross budget, net target)

    Args:
        signals: List of StrategySignal from multiple strategies.
            All signals should be for the same timestamp.
        sizing_rules: Rules for weighting and constraints.
            If None, uses default rules.

    Returns:
        AggregationResult with target DV01 by bond and metadata.

    Example:
        >>> signals = [signal_2y5y10y, signal_5y10y30y]
        >>> rules = SizingRules(gross_dv01_budget=50000)
        >>> result = signals_to_bond_targets(signals, rules)
        >>> print(result.target_dv01)
        UST_2Y     5000.0
        UST_5Y   -15000.0
        UST_10Y   10000.0
        UST_30Y    5000.0
        dtype: float64
    """
    if sizing_rules is None:
        sizing_rules = SizingRules()

    # Filter by confidence if specified
    excluded = []
    if sizing_rules.min_signal_confidence is not None:
        filtered = []
        for s in signals:
            if s.confidence is None or s.confidence >= sizing_rules.min_signal_confidence:
                filtered.append(s)
            else:
                excluded.append(s.strategy_id)
        signals = filtered

    # Filter out flat signals
    active_signals = [s for s in signals if not s.is_flat]

    if not active_signals:
        return AggregationResult(
            target_dv01=pd.Series(dtype=float),
            excluded_strategies=excluded,
        )

    # Get unique strategies for equal weighting
    strategy_ids = list({s.strategy_id for s in active_signals})
    n_strategies = len(strategy_ids)

    # Get timestamp (assume all same, use first)
    timestamp = active_signals[0].timestamp

    # Compute weighted DV01 contribution per strategy
    strategy_contributions: dict = {}
    all_bond_dv01: dict = {}

    for signal in active_signals:
        # Get strategy weight
        weight = _get_strategy_weight(
            signal.strategy_id,
            timestamp,
            sizing_rules.strategy_weights,
            n_strategies,
        )

        if weight == 0:
            continue

        # Compute raw DV01 per bond for this signal
        signal_dv01 = _compute_signal_dv01(signal, sizing_rules.default_target_dv01)

        # Apply strategy weight
        weighted_dv01 = {bond: dv01 * weight for bond, dv01 in signal_dv01.items()}

        # Store strategy contribution
        strategy_contributions[signal.strategy_id] = weighted_dv01

        # Accumulate into total (netting happens here)
        for bond_id, dv01 in weighted_dv01.items():
            all_bond_dv01[bond_id] = all_bond_dv01.get(bond_id, 0.0) + dv01

    # Convert to Series
    bond_dv01 = pd.Series(all_bond_dv01, dtype=float)

    if bond_dv01.empty:
        return AggregationResult(
            target_dv01=pd.Series(dtype=float),
            strategy_contributions=strategy_contributions,
            excluded_strategies=excluded,
        )

    # Apply constraints
    constraints_applied = []
    scale_factor = 1.0

    # 1. Per-bond cap (applied first)
    if sizing_rules.per_bond_dv01_cap is not None:
        bond_dv01, was_binding = _apply_per_bond_cap(
            bond_dv01, sizing_rules.per_bond_dv01_cap
        )
        if was_binding:
            constraints_applied.append("per_bond_dv01_cap")

    # 2. Gross DV01 budget
    if sizing_rules.gross_dv01_budget is not None:
        bond_dv01, scale_factor, was_binding = _apply_gross_budget(
            bond_dv01, sizing_rules.gross_dv01_budget
        )
        if was_binding:
            constraints_applied.append("gross_dv01_budget")

    # 3. Net DV01 target (applied last)
    if sizing_rules.net_dv01_target is not None:
        bond_dv01, was_adjusted = _apply_net_target(
            bond_dv01, sizing_rules.net_dv01_target
        )
        if was_adjusted:
            constraints_applied.append("net_dv01_target")

    # Compute final metrics
    gross_dv01 = bond_dv01.abs().sum()
    net_dv01 = bond_dv01.sum()

    return AggregationResult(
        target_dv01=bond_dv01,
        strategy_contributions=strategy_contributions,
        gross_dv01=gross_dv01,
        net_dv01=net_dv01,
        scale_factor=scale_factor,
        constraints_applied=constraints_applied,
        excluded_strategies=excluded,
    )


def aggregate_signal_batch(
    signals: list[StrategySignal],
    sizing_rules: SizingRules | None = None,
) -> PortfolioTarget:
    """Aggregate signals into a PortfolioTarget.

    Convenience wrapper around signals_to_bond_targets that returns
    a PortfolioTarget directly.

    Args:
        signals: List of StrategySignal (should be same timestamp).
        sizing_rules: Sizing rules to apply.

    Returns:
        PortfolioTarget with aggregated positions.
    """
    if not signals:
        raise ValueError("No signals to aggregate")

    result = signals_to_bond_targets(signals, sizing_rules)
    timestamp = signals[0].timestamp

    return result.to_portfolio_target(timestamp)


def aggregate_batches(
    batches: list,  # list[StrategySignalBatch]
    sizing_rules: SizingRules | None = None,
) -> list[PortfolioTarget]:
    """Aggregate multiple batches into portfolio targets.

    Args:
        batches: List of StrategySignalBatch.
        sizing_rules: Sizing rules to apply (same for all batches).

    Returns:
        List of PortfolioTarget, one per batch.
    """
    targets = []
    for batch in batches:
        if len(batch.signals) > 0:
            target = aggregate_signal_batch(batch.signals, sizing_rules)
            targets.append(target)
    return targets


def compute_turnover(
    current: pd.Series,
    previous: pd.Series,
) -> float:
    """Compute turnover between two position snapshots.

    Args:
        current: Current target DV01 by bond.
        previous: Previous target DV01 by bond.

    Returns:
        Total turnover (sum of absolute changes).
    """
    # Align indices
    all_bonds = current.index.union(previous.index)
    curr_aligned = current.reindex(all_bonds, fill_value=0.0)
    prev_aligned = previous.reindex(all_bonds, fill_value=0.0)

    return (curr_aligned - prev_aligned).abs().sum()


def compute_position_changes(
    current: pd.Series,
    previous: pd.Series,
) -> pd.DataFrame:
    """Compute detailed position changes between snapshots.

    Args:
        current: Current target DV01 by bond.
        previous: Previous target DV01 by bond.

    Returns:
        DataFrame with columns: bond_id, previous, current, change, pct_change
    """
    all_bonds = current.index.union(previous.index)
    curr_aligned = current.reindex(all_bonds, fill_value=0.0)
    prev_aligned = previous.reindex(all_bonds, fill_value=0.0)

    changes = curr_aligned - prev_aligned
    pct_changes = changes / prev_aligned.abs().replace(0, float("nan"))

    return pd.DataFrame({
        "bond_id": all_bonds,
        "previous": prev_aligned.values,
        "current": curr_aligned.values,
        "change": changes.values,
        "pct_change": pct_changes.values,
    }).set_index("bond_id")


def create_sizing_rules_with_weighting(
    weighting_config,  # WeightingConfig
    strategy_pnl: pd.DataFrame | None = None,
    default_target_dv01: float = 10000.0,
    gross_dv01_budget: float | None = None,
    net_dv01_target: float | None = None,
    per_bond_dv01_cap: float | None = None,
    min_signal_confidence: float | None = None,
) -> SizingRules:
    """Create SizingRules with advanced weighting from WeightingConfig.

    Convenience function that integrates WeightingConfig with SizingRules.
    For time-varying weights (e.g., inverse_vol), creates a callable
    that recomputes weights at each timestamp.

    Args:
        weighting_config: WeightingConfig specifying the weighting method.
        strategy_pnl: Strategy P&L DataFrame for inverse_vol weighting.
            Required if weighting_config.method is INVERSE_VOL.
        default_target_dv01: Default target gross DV01 per strategy.
        gross_dv01_budget: Maximum total gross DV01.
        net_dv01_target: Target net DV01.
        per_bond_dv01_cap: Maximum absolute DV01 per bond.
        min_signal_confidence: Minimum confidence threshold.

    Returns:
        SizingRules configured with the weighting function.

    Example:
        >>> from mlstudy.trading.portfolio import WeightingConfig, WeightingMethod
        >>> config = WeightingConfig(
        ...     method=WeightingMethod.INVERSE_VOL,
        ...     vol_lookback=20,
        ...     max_weight=0.5,
        ... )
        >>> rules = create_sizing_rules_with_weighting(
        ...     config,
        ...     strategy_pnl=pnl_df,
        ...     gross_dv01_budget=50000,
        ... )
    """
    from mlstudy.trading.portfolio.weighting import (
        WeightingMethod,
        create_weighting_function,
    )

    # For equal weights, use the string shortcut
    if weighting_config.method == WeightingMethod.EQUAL:
        strategy_weights = "equal"
    else:
        # Create callable for time-varying weights
        strategy_weights = create_weighting_function(weighting_config, strategy_pnl)

    return SizingRules(
        strategy_weights=strategy_weights,
        default_target_dv01=default_target_dv01,
        gross_dv01_budget=gross_dv01_budget,
        net_dv01_target=net_dv01_target,
        per_bond_dv01_cap=per_bond_dv01_cap,
        min_signal_confidence=min_signal_confidence,
    )
