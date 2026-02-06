"""Strategy weighting schemes for combining multiple strategies.

Provides methods for computing strategy weights:
- Equal weight: Simple 1/N allocation
- Inverse volatility (risk parity): Weight inversely proportional to trailing vol
- Capped weights: Apply min/max caps with renormalization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class WeightingMethod(Enum):
    """Available weighting methods."""

    EQUAL = "equal"
    INVERSE_VOL = "inverse_vol"
    CUSTOM = "custom"


@dataclass
class WeightingConfig:
    """Configuration for strategy weighting.

    Attributes:
        method: Weighting method to use.
        vol_lookback: Lookback period for volatility calculation (inverse_vol).
        min_weight: Minimum weight per strategy (floor).
        max_weight: Maximum weight per strategy (cap).
        floor_vol: Minimum volatility to avoid division by zero.
        normalize: Whether to normalize weights to sum to 1.
        custom_weights: Dict of {strategy_id: weight} for CUSTOM method.

    Example:
        >>> config = WeightingConfig(
        ...     method=WeightingMethod.INVERSE_VOL,
        ...     vol_lookback=20,
        ...     max_weight=0.4,  # No strategy > 40%
        ... )
    """

    method: WeightingMethod = WeightingMethod.EQUAL
    vol_lookback: int = 20
    min_weight: float = 0.0
    max_weight: float = 1.0
    floor_vol: float = 1e-6
    normalize: bool = True
    custom_weights: dict | None = None


@dataclass
class WeightingResult:
    """Result of weight computation.

    Attributes:
        weights: Dict of {strategy_id: weight}.
        raw_weights: Weights before caps/normalization.
        volatilities: Strategy volatilities (for inverse_vol).
        caps_applied: Whether caps were binding.
        method: Method used.
    """

    weights: dict
    raw_weights: dict | None = None
    volatilities: dict | None = None
    caps_applied: bool = False
    method: str = "equal"

    @property
    def weight_sum(self) -> float:
        """Sum of all weights."""
        return sum(self.weights.values())

    def to_series(self) -> pd.Series:
        """Convert weights to Series."""
        return pd.Series(self.weights)


def compute_equal_weights(strategy_ids: list[str]) -> dict[str, float]:
    """Compute equal weights across strategies.

    Args:
        strategy_ids: List of strategy identifiers.

    Returns:
        Dict of {strategy_id: weight} with equal weights summing to 1.

    Example:
        >>> compute_equal_weights(["strat_A", "strat_B", "strat_C"])
        {'strat_A': 0.333..., 'strat_B': 0.333..., 'strat_C': 0.333...}
    """
    if not strategy_ids:
        return {}

    n = len(strategy_ids)
    weight = 1.0 / n
    return dict.fromkeys(strategy_ids, weight)


def compute_inverse_vol_weights(
    strategy_pnl: pd.DataFrame,
    strategy_ids: list[str] | None = None,
    lookback: int = 20,
    floor_vol: float = 1e-6,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute inverse volatility (risk parity) weights.

    Weights each strategy inversely proportional to its trailing volatility,
    so lower-vol strategies get higher weights.

    Args:
        strategy_pnl: DataFrame with strategy P&L. Columns are strategy IDs,
            index is timestamps. Uses the last `lookback` rows.
        strategy_ids: Optional list of strategies to include. If None, uses
            all columns in strategy_pnl.
        lookback: Number of periods for volatility calculation.
        floor_vol: Minimum volatility to avoid division by zero.

    Returns:
        Tuple of (weights dict, volatilities dict).

    Example:
        >>> pnl_df = pd.DataFrame({
        ...     "strat_A": [100, -50, 80, -30, 60],
        ...     "strat_B": [10, -5, 8, -3, 6],  # Lower vol
        ... })
        >>> weights, vols = compute_inverse_vol_weights(pnl_df)
        >>> # strat_B gets higher weight due to lower vol
    """
    if strategy_ids is None:
        strategy_ids = list(strategy_pnl.columns)

    # Use last lookback rows
    pnl_window = strategy_pnl[strategy_ids].iloc[-lookback:]

    # Compute volatility for each strategy
    volatilities = {}
    for sid in strategy_ids:
        vol = pnl_window[sid].std()
        # Apply floor to avoid division by zero
        volatilities[sid] = max(vol, floor_vol) if pd.notna(vol) else floor_vol

    # Inverse volatility weights
    inv_vols = {sid: 1.0 / vol for sid, vol in volatilities.items()}
    total_inv_vol = sum(inv_vols.values())

    if total_inv_vol == 0:
        # Fallback to equal weights
        return compute_equal_weights(strategy_ids), volatilities

    # Normalize to sum to 1
    weights = {sid: iv / total_inv_vol for sid, iv in inv_vols.items()}

    return weights, volatilities


def apply_weight_caps(
    weights: dict[str, float],
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    normalize: bool = True,
    max_iterations: int = 10,
) -> tuple[dict[str, float], bool]:
    """Apply min/max caps to weights and optionally renormalize.

    Uses iterative clipping: clip to bounds, renormalize uncapped weights,
    repeat until convergence or max iterations.

    Args:
        weights: Dict of {strategy_id: weight}.
        min_weight: Minimum weight (floor).
        max_weight: Maximum weight (cap).
        normalize: Whether to renormalize after clipping.
        max_iterations: Maximum iterations for convergence.

    Returns:
        Tuple of (capped weights, whether caps were binding).

    Example:
        >>> weights = {"A": 0.6, "B": 0.3, "C": 0.1}
        >>> capped, was_binding = apply_weight_caps(weights, max_weight=0.4)
        >>> # A capped to 0.4, B and C scaled up proportionally
    """
    if not weights:
        return {}, False

    n_strategies = len(weights)
    result = dict(weights)
    caps_applied = False

    # Check if constraints are feasible
    # min_weight * n <= 1 and max_weight * n >= 1
    if min_weight * n_strategies > 1.0:
        # Infeasible floor, use equal weights
        return compute_equal_weights(list(weights.keys())), True

    for _iteration in range(max_iterations):
        # Identify strategies at bounds vs free
        at_cap = {}
        at_floor = {}
        free = {}

        for sid, w in result.items():
            if w >= max_weight - 1e-10:
                at_cap[sid] = max_weight
                if w > max_weight + 1e-10:
                    caps_applied = True
            elif w <= min_weight + 1e-10:
                at_floor[sid] = min_weight
                if w < min_weight - 1e-10:
                    caps_applied = True
            else:
                free[sid] = w

        # Combine capped
        capped = {**at_cap, **at_floor}

        # If nothing free or no normalization needed, we're done
        if not free or not normalize:
            result = {**capped, **free}
            break

        # Compute remaining weight to distribute among free strategies
        capped_total = sum(capped.values())
        remaining = 1.0 - capped_total

        if remaining <= 0:
            # All weight consumed by capped strategies
            # Give free strategies their floor
            for sid in free:
                free[sid] = min_weight
            result = {**capped, **free}
            # Renormalize final result
            total = sum(result.values())
            if total > 0:
                result = {sid: w / total for sid, w in result.items()}
            break

        # Scale free weights to fill remaining
        free_total = sum(free.values())
        if free_total > 0:
            scale = remaining / free_total
            free = {sid: w * scale for sid, w in free.items()}

        result = {**capped, **free}

        # Check if any newly scaled weights exceed caps
        all_within_bounds = all(
            min_weight - 1e-10 <= w <= max_weight + 1e-10 for w in result.values()
        )
        if all_within_bounds:
            break

    # Final normalization to ensure sum is exactly 1
    if normalize:
        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 1e-10:
            result = {sid: w / total for sid, w in result.items()}

    return result, caps_applied


def compute_strategy_weights(
    strategy_ids: list[str],
    config: WeightingConfig | None = None,
    strategy_pnl: pd.DataFrame | None = None,
) -> WeightingResult:
    """Compute strategy weights using specified method.

    Main entry point for weight computation. Handles method selection,
    volatility calculation, and cap application.

    Args:
        strategy_ids: List of strategy identifiers.
        config: Weighting configuration. If None, uses equal weights.
        strategy_pnl: DataFrame with strategy P&L for inverse_vol method.
            Required if method is INVERSE_VOL.

    Returns:
        WeightingResult with computed weights and metadata.

    Raises:
        ValueError: If inverse_vol requested but no P&L data provided.

    Example:
        >>> config = WeightingConfig(
        ...     method=WeightingMethod.INVERSE_VOL,
        ...     max_weight=0.5,
        ... )
        >>> result = compute_strategy_weights(
        ...     ["fly_2y5y10y", "fly_5y10y30y"],
        ...     config=config,
        ...     strategy_pnl=pnl_df,
        ... )
        >>> print(result.weights)
    """
    if config is None:
        config = WeightingConfig()

    if not strategy_ids:
        return WeightingResult(weights={}, method=config.method.value)

    # Compute raw weights based on method
    volatilities = None

    if config.method == WeightingMethod.EQUAL:
        raw_weights = compute_equal_weights(strategy_ids)

    elif config.method == WeightingMethod.INVERSE_VOL:
        if strategy_pnl is None:
            raise ValueError("strategy_pnl required for inverse_vol weighting")

        # Filter to strategies that exist in P&L data
        available = [sid for sid in strategy_ids if sid in strategy_pnl.columns]
        if not available:
            raise ValueError("No strategies found in strategy_pnl columns")

        raw_weights, volatilities = compute_inverse_vol_weights(
            strategy_pnl,
            strategy_ids=available,
            lookback=config.vol_lookback,
            floor_vol=config.floor_vol,
        )

        # Add zero weights for strategies not in P&L
        for sid in strategy_ids:
            if sid not in raw_weights:
                raw_weights[sid] = 0.0

    elif config.method == WeightingMethod.CUSTOM:
        if config.custom_weights is None:
            raise ValueError("custom_weights required for CUSTOM method")

        raw_weights = {}
        for sid in strategy_ids:
            raw_weights[sid] = config.custom_weights.get(sid, 0.0)

        # Normalize if requested
        if config.normalize:
            total = sum(raw_weights.values())
            if total > 0:
                raw_weights = {sid: w / total for sid, w in raw_weights.items()}

    else:
        raise ValueError(f"Unknown weighting method: {config.method}")

    # Apply caps
    weights, caps_applied = apply_weight_caps(
        raw_weights,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        normalize=config.normalize,
    )

    return WeightingResult(
        weights=weights,
        raw_weights=raw_weights,
        volatilities=volatilities,
        caps_applied=caps_applied,
        method=config.method.value,
    )


def create_weighting_function(
    config: WeightingConfig,
    strategy_pnl: pd.DataFrame | None = None,
):
    """Create a callable weighting function for use with SizingRules.

    Returns a function that can be passed as strategy_weights to SizingRules
    for time-varying weights.

    Args:
        config: Weighting configuration.
        strategy_pnl: Strategy P&L DataFrame (will be updated over time).

    Returns:
        Callable (strategy_id, timestamp) -> weight.

    Example:
        >>> config = WeightingConfig(method=WeightingMethod.INVERSE_VOL)
        >>> weight_fn = create_weighting_function(config, pnl_df)
        >>> rules = SizingRules(strategy_weights=weight_fn)
    """
    # Cache for computed weights
    _cache: dict = {"weights": None, "timestamp": None}

    def weight_function(strategy_id: str, timestamp) -> float:
        # Recompute weights if timestamp changed (new batch)
        if _cache["timestamp"] != timestamp:
            # Get all strategies from P&L if available
            strategy_ids = list(strategy_pnl.columns) if strategy_pnl is not None else [strategy_id]

            result = compute_strategy_weights(
                strategy_ids=strategy_ids,
                config=config,
                strategy_pnl=strategy_pnl,
            )
            _cache["weights"] = result.weights
            _cache["timestamp"] = timestamp

        return _cache["weights"].get(strategy_id, 0.0)

    return weight_function


def compute_rolling_weights(
    strategy_pnl: pd.DataFrame,
    config: WeightingConfig,
) -> pd.DataFrame:
    """Compute rolling weights over time.

    Useful for analyzing how weights would have evolved historically.

    Args:
        strategy_pnl: DataFrame with strategy P&L (columns = strategies).
        config: Weighting configuration.

    Returns:
        DataFrame with weights over time (same index as strategy_pnl).

    Example:
        >>> config = WeightingConfig(
        ...     method=WeightingMethod.INVERSE_VOL,
        ...     vol_lookback=20,
        ... )
        >>> weight_history = compute_rolling_weights(pnl_df, config)
    """
    strategy_ids = list(strategy_pnl.columns)
    records = []

    for i in range(len(strategy_pnl)):
        # Use data up to current row
        pnl_slice = strategy_pnl.iloc[: i + 1]

        if len(pnl_slice) < config.vol_lookback and config.method == WeightingMethod.INVERSE_VOL:
            # Not enough data, use equal weights
            weights = compute_equal_weights(strategy_ids)
        else:
            result = compute_strategy_weights(
                strategy_ids=strategy_ids,
                config=config,
                strategy_pnl=pnl_slice,
            )
            weights = result.weights

        record = {"timestamp": strategy_pnl.index[i], **weights}
        records.append(record)

    return pd.DataFrame(records).set_index("timestamp")


def validate_weights(
    weights: dict[str, float],
    tolerance: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Validate that weights are well-formed.

    Checks:
    - All weights are non-negative
    - Weights sum to 1 (within tolerance)
    - No NaN values

    Args:
        weights: Dict of {strategy_id: weight}.
        tolerance: Tolerance for sum-to-1 check.

    Returns:
        Tuple of (is_valid, list of error messages).

    Example:
        >>> is_valid, errors = validate_weights({"A": 0.5, "B": 0.5})
        >>> assert is_valid
    """
    errors = []

    if not weights:
        return True, []

    # Check for NaN
    nan_strategies = [sid for sid, w in weights.items() if pd.isna(w)]
    if nan_strategies:
        errors.append(f"NaN weights for: {nan_strategies}")

    # Check non-negative
    negative = [sid for sid, w in weights.items() if w < 0]
    if negative:
        errors.append(f"Negative weights for: {negative}")

    # Check sum to 1
    weight_sum = sum(w for w in weights.values() if pd.notna(w))
    if abs(weight_sum - 1.0) > tolerance:
        errors.append(f"Weights sum to {weight_sum:.6f}, not 1.0")

    return len(errors) == 0, errors
