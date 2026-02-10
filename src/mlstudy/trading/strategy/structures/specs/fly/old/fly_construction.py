"""Fly construction methods optimized for mean reversion.

This module provides multiple methods to construct butterfly spread ("fly")
series from three yield series (front, belly, back) with weights chosen
to enhance mean-reverting behavior.

Key methods:
1. Levels residual (cointegration-style OLS)
2. Changes residual (OLS on differences)
3. Direct optimization for most mean-reverting fly

All methods support both static (full-sample) and rolling estimation.

Usage:
    >>> from mlstudy.trading.strategy.fly import fly_levels_residual
    >>> fly, weights_df = fly_levels_residual(front, belly, back, window=252)
    >>> # Use fly series for signal generation
    >>> # Use weights_df for execution sizing

Note:
    Core construction logic is provided by
    mlstudy.trading.strategy.structures.construction.
    This module provides fly-specific wrappers and additional utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

# Import shared construction primitives
from mlstudy.trading.strategy.structures.construction.construction import (
    _validate_3leg,
    changes_residual_3leg,
    enforce_dv01_neutral,
    fit_ar1_half_life,
    levels_residual_3leg,
    optimize_stationarity_3leg,
)


# =============================================================================
# FlyWeights dataclass
# =============================================================================


@dataclass
class FlyWeights:
    """Weights for butterfly spread construction.

    Attributes:
        w_front: Weight on front leg (e.g., 2Y yield).
        w_belly: Weight on belly leg (e.g., 5Y yield).
        w_back: Weight on back leg (e.g., 10Y yield).
        method: Construction method used (e.g., "levels_residual", "optimize").
        window: Rolling window size if applicable, None for static.
        meta: Additional metadata (e.g., objective value, net_dv01).
    """

    w_front: float
    w_belly: float
    w_back: float
    method: str
    window: Optional[int] = None
    meta: Dict = field(default_factory=dict)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return weights as (front, belly, back) tuple."""
        return (self.w_front, self.w_belly, self.w_back)

    def compute_fly(
        self,
        front: Union[float, np.ndarray, pd.Series],
        belly: Union[float, np.ndarray, pd.Series],
        back: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Compute fly value using these weights."""
        return self.w_front * front + self.w_belly * belly + self.w_back * back


# =============================================================================
# Re-export shared functions with fly-specific names
# =============================================================================

# Re-export enforce_dv01_neutral as-is
__all__ = [
    "FlyWeights",
    "enforce_dv01_neutral",
    "fit_ar1_half_life",
    "fly_changes_residual",
    "fly_levels_residual",
    "fly_optimize_mean_reversion",
    "validate_inputs",
    "compute_fly_from_weights",
    "equal_weight_fly",
]


def validate_inputs(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Align and validate three yield series for fly construction.

    Args:
        front: Front leg yield series (e.g., 2Y).
        belly: Belly leg yield series (e.g., 5Y).
        back: Back leg yield series (e.g., 10Y).
        min_obs: Minimum observations required after alignment and NaN removal.

    Returns:
        DataFrame with aligned 'front', 'belly', 'back' columns, NaNs dropped.

    Raises:
        ValueError: If fewer than min_obs valid observations remain.
    """
    return _validate_3leg(front, belly, back, min_obs=min_obs)


# =============================================================================
# Fly-specific wrappers
# =============================================================================


def fly_levels_residual(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int = 252,
    min_obs: int = 60,
    rolling: bool = True,
    add_intercept: bool = True,
    ridge_lambda: Optional[float] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Construct fly as regression residual on yield levels.

    Regresses belly on front and back yields:
        belly_t = c + beta_front * front_t + beta_back * back_t + e_t

    The fly series is the residual e_t, which should be stationary if
    the yields are cointegrated.

    Args:
        front: Front leg yield series (e.g., 2Y).
        belly: Belly leg yield series (e.g., 5Y).
        back: Back leg yield series (e.g., 10Y).
        window: Rolling window size for coefficient estimation.
        min_obs: Minimum observations for validation.
        rolling: If True, use rolling window; if False, use full sample.
        add_intercept: If True, include intercept in regression.
        ridge_lambda: Ridge regularization parameter (optional).

    Returns:
        Tuple of:
        - fly: pd.Series of residuals (the fly series)
        - weights_df: pd.DataFrame with columns:
            - beta_front: Coefficient on front leg
            - beta_back: Coefficient on back leg
            - intercept: Intercept (if add_intercept=True)

    Notes:
        - Weights are TIME-VARYING when rolling=True.
        - NO LOOKAHEAD: coefficients at time t use data up to and including t.
    """
    fly, weights_df = levels_residual_3leg(
        front,
        belly,
        back,
        window=window,
        rolling=rolling,
        min_obs=min_obs,
        add_intercept=add_intercept,
        ridge_lambda=ridge_lambda,
    )
    # Rename for fly-specific output
    fly = fly.rename("fly_levels_residual")
    return fly, weights_df


def fly_changes_residual(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int = 252,
    min_obs: int = 60,
    rolling: bool = True,
    add_intercept: bool = True,
    ridge_lambda: Optional[float] = None,
    integrate: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Construct fly from regression on yield changes (differences).

    Regresses change in belly on changes in front and back:
        Δbelly_t = c + beta_front * Δfront_t + beta_back * Δback_t + e_t

    The fly series is either the cumulative residual (if integrate=True)
    or the residual changes themselves.

    Args:
        front: Front leg yield series (e.g., 2Y).
        belly: Belly leg yield series (e.g., 5Y).
        back: Back leg yield series (e.g., 10Y).
        window: Rolling window size for coefficient estimation.
        min_obs: Minimum observations for validation.
        rolling: If True, use rolling window; if False, use full sample.
        add_intercept: If True, include intercept in regression.
        ridge_lambda: Ridge regularization parameter (optional).
        integrate: If True, return cumulative sum of residuals (level);
            if False, return residual changes.

    Returns:
        Tuple of:
        - fly: pd.Series of (integrated) residuals
        - weights_df: pd.DataFrame with columns:
            - beta_front: Coefficient on Δfront
            - beta_back: Coefficient on Δback
            - intercept: Intercept (if add_intercept=True)

    Notes:
        - Weights are TIME-VARYING when rolling=True.
        - NO LOOKAHEAD: coefficients at time t use data up to t.
    """
    fly, weights_df = changes_residual_3leg(
        front,
        belly,
        back,
        window=window,
        rolling=rolling,
        min_obs=min_obs,
        add_intercept=add_intercept,
        ridge_lambda=ridge_lambda,
        integrate=integrate,
    )
    # Rename for fly-specific output
    if integrate:
        fly = fly.rename("fly_changes_residual_integrated")
    else:
        fly = fly.rename("fly_changes_residual")
    return fly, weights_df


def fly_optimize_mean_reversion(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int = 252,
    min_obs: int = 120,
    rolling: bool = True,
    w_belly: float = 1.0,
    bounds_front: Tuple[float, float] = (-3.0, 3.0),
    bounds_back: Tuple[float, float] = (-3.0, 3.0),
    objective: Literal["half_life", "ar1_b", "adf_pvalue"] = "half_life",
    grid_size: int = 41,
    dv01: Optional[Tuple[float, float, float]] = None,
    dv01_tolerance: float = 1e-6,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Optimize fly weights to maximize mean reversion.

    Searches over a grid of front and back weights to find the fly
    construction that minimizes the specified mean-reversion objective.

    Args:
        front: Front leg yield series (e.g., 2Y).
        belly: Belly leg yield series (e.g., 5Y).
        back: Back leg yield series (e.g., 10Y).
        window: Rolling window size for optimization.
        min_obs: Minimum observations for objective evaluation.
        rolling: If True, re-optimize for each time point.
        w_belly: Fixed weight on belly leg (typically 1.0 or -2.0).
        bounds_front: (min, max) bounds for front leg weight.
        bounds_back: (min, max) bounds for back leg weight.
        objective: Optimization objective:
            - "half_life": Minimize half-life (faster mean reversion)
            - "ar1_b": Minimize AR(1) coefficient (prefer 0 < b < 1)
            - "adf_pvalue": Minimize ADF p-value (requires statsmodels)
        grid_size: Number of grid points per dimension.
        dv01: Optional (dv01_front, dv01_belly, dv01_back) for DV01 constraint.
        dv01_tolerance: Maximum allowed absolute net DV01.

    Returns:
        Tuple of:
        - fly: pd.Series of optimized fly values
        - weights_df: pd.DataFrame with columns:
            - w_front, w_belly, w_back: Weights
            - objective_value: Achieved objective value
            - net_dv01: Net DV01 (if dv01 provided)

    Notes:
        - Weights are TIME-VARYING when rolling=True.
        - NO LOOKAHEAD: optimization at time t uses data up to t only.
    """
    # For ar1_b and adf_pvalue objectives, use local implementation
    # The shared layer only supports half_life
    if objective in ("ar1_b", "adf_pvalue"):
        return _fly_optimize_extended(
            front,
            belly,
            back,
            window=window,
            min_obs=min_obs,
            rolling=rolling,
            w_belly=w_belly,
            bounds_front=bounds_front,
            bounds_back=bounds_back,
            objective=objective,
            grid_size=grid_size,
            dv01=dv01,
            dv01_tolerance=dv01_tolerance,
        )

    # Use shared implementation for half_life
    fly, weights_df = optimize_stationarity_3leg(
        front,
        belly,
        back,
        window=window,
        rolling=rolling,
        min_obs=min_obs,
        w_belly=w_belly,
        bounds_front=bounds_front,
        bounds_back=bounds_back,
        grid_size=grid_size,
        objective="half_life",
    )

    # Add net_dv01 column if dv01 provided
    if dv01 is not None:
        dv01_front, dv01_belly, dv01_back = dv01
        weights_df["net_dv01"] = (
            weights_df["w_front"] * dv01_front
            + weights_df["w_belly"] * dv01_belly
            + weights_df["w_back"] * dv01_back
        )
    else:
        weights_df["net_dv01"] = np.nan

    fly = fly.rename("fly_optimized")
    return fly, weights_df


def _fly_optimize_extended(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int,
    min_obs: int,
    rolling: bool,
    w_belly: float,
    bounds_front: Tuple[float, float],
    bounds_back: Tuple[float, float],
    objective: str,
    grid_size: int,
    dv01: Optional[Tuple[float, float, float]],
    dv01_tolerance: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Extended optimization with ar1_b and adf_pvalue objectives."""
    df = validate_inputs(front, belly, back, min_obs=min_obs)
    n = len(df)
    idx = df.index

    front_vals = df["front"].values
    belly_vals = df["belly"].values
    back_vals = df["back"].values

    w_front_grid = np.linspace(bounds_front[0], bounds_front[1], grid_size)
    w_back_grid = np.linspace(bounds_back[0], bounds_back[1], grid_size)

    if dv01 is not None:
        dv01_front, dv01_belly, dv01_back = dv01

    def evaluate_objective(fly_values: np.ndarray) -> float:
        """Evaluate objective on fly values."""
        LARGE_PENALTY = 1e10
        clean = fly_values[~np.isnan(fly_values)]
        if len(clean) < 30:
            return LARGE_PENALTY

        if objective == "ar1_b":
            try:
                result = fit_ar1_half_life(pd.Series(clean), min_obs=30)
                b = result["b"]
                if 0 < b < 1:
                    return b
                else:
                    return LARGE_PENALTY
            except (ValueError, np.linalg.LinAlgError):
                return LARGE_PENALTY

        elif objective == "adf_pvalue":
            try:
                from statsmodels.tsa.stattools import adfuller
            except ImportError:
                raise RuntimeError(
                    "statsmodels is required for objective='adf_pvalue'. "
                    "Install with: pip install statsmodels"
                )

            try:
                result = adfuller(clean, maxlag=None, regression="c", autolag="AIC")
                return result[1]
            except Exception:
                return LARGE_PENALTY

        return LARGE_PENALTY

    def find_best_weights(
        front_win: np.ndarray,
        belly_win: np.ndarray,
        back_win: np.ndarray,
    ) -> Tuple[float, float, float, Optional[float]]:
        best_obj = 1e10
        best_w_front = w_front_grid[len(w_front_grid) // 2]
        best_w_back = w_back_grid[len(w_back_grid) // 2]
        best_net_dv01 = None

        min_dv01_violation = float("inf")
        min_dv01_weights = (best_w_front, best_w_back)

        for w_f in w_front_grid:
            for w_b in w_back_grid:
                if dv01 is not None:
                    net = w_f * dv01_front + w_belly * dv01_belly + w_b * dv01_back
                    abs_net = abs(net)

                    if abs_net < min_dv01_violation:
                        min_dv01_violation = abs_net
                        min_dv01_weights = (w_f, w_b)

                    if abs_net > dv01_tolerance:
                        continue

                fly_win = w_f * front_win + w_belly * belly_win + w_b * back_win
                obj = evaluate_objective(fly_win)

                if obj < best_obj:
                    best_obj = obj
                    best_w_front = w_f
                    best_w_back = w_b
                    if dv01 is not None:
                        best_net_dv01 = (
                            w_f * dv01_front + w_belly * dv01_belly + w_b * dv01_back
                        )

        if dv01 is not None and best_obj >= 1e10:
            best_w_front, best_w_back = min_dv01_weights
            best_net_dv01 = (
                best_w_front * dv01_front
                + w_belly * dv01_belly
                + best_w_back * dv01_back
            )
            fly_win = (
                best_w_front * front_win + w_belly * belly_win + best_w_back * back_win
            )
            best_obj = evaluate_objective(fly_win)

        return best_w_front, best_w_back, best_obj, best_net_dv01

    weights_arr = np.full((n, 5), np.nan)

    if rolling:
        if window > n:
            raise ValueError(f"Window ({window}) exceeds observations ({n})")

        for t in range(window - 1, n):
            start = t - window + 1
            end = t + 1

            w_f, w_b, obj_val, net_dv01 = find_best_weights(
                front_vals[start:end],
                belly_vals[start:end],
                back_vals[start:end],
            )

            weights_arr[t, 0] = w_f
            weights_arr[t, 1] = w_belly
            weights_arr[t, 2] = w_b
            weights_arr[t, 3] = obj_val
            weights_arr[t, 4] = net_dv01 if net_dv01 is not None else np.nan
    else:
        w_f, w_b, obj_val, net_dv01 = find_best_weights(
            front_vals, belly_vals, back_vals
        )

        weights_arr[:, 0] = w_f
        weights_arr[:, 1] = w_belly
        weights_arr[:, 2] = w_b
        weights_arr[:, 3] = obj_val
        weights_arr[:, 4] = net_dv01 if net_dv01 is not None else np.nan

    weights_df = pd.DataFrame(
        weights_arr,
        index=idx,
        columns=["w_front", "w_belly", "w_back", "objective_value", "net_dv01"],
    )

    fly_values = (
        weights_arr[:, 0] * front_vals
        + weights_arr[:, 1] * belly_vals
        + weights_arr[:, 2] * back_vals
    )
    fly = pd.Series(fly_values, index=idx, name="fly_optimized")

    return fly, weights_df


# =============================================================================
# Convenience functions
# =============================================================================


def compute_fly_from_weights(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    weights_df: pd.DataFrame,
) -> pd.Series:
    """Compute fly series from time-varying weights DataFrame.

    Args:
        front: Front leg yield series.
        belly: Belly leg yield series.
        back: Back leg yield series.
        weights_df: DataFrame with w_front, w_belly, w_back columns.

    Returns:
        Fly series computed using weights at each time point.
    """
    df = validate_inputs(front, belly, back, min_obs=1)
    df = df.join(weights_df[["w_front", "w_belly", "w_back"]], how="inner")

    fly = (
        df["w_front"] * df["front"]
        + df["w_belly"] * df["belly"]
        + df["w_back"] * df["back"]
    )
    fly.name = "fly_from_weights"
    return fly


def equal_weight_fly(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    w_front: float = 1.0,
    w_belly: float = -2.0,
    w_back: float = 1.0,
) -> Tuple[pd.Series, FlyWeights]:
    """Construct equal-weight (or custom fixed-weight) fly.

    This is a baseline for comparison with optimized methods.

    Args:
        front: Front leg yield series.
        belly: Belly leg yield series.
        back: Back leg yield series.
        w_front: Weight on front leg (default 1.0).
        w_belly: Weight on belly leg (default -2.0).
        w_back: Weight on back leg (default 1.0).

    Returns:
        Tuple of:
        - fly: pd.Series of fly values
        - weights: FlyWeights dataclass with constant weights

    Notes:
        - Weights are STATIC (not time-varying).
        - NO LOOKAHEAD: this is a pure fixed-weight construction.
        - Standard butterfly: (1, -2, 1) means long wings, short belly.
    """
    df = validate_inputs(front, belly, back, min_obs=1)

    fly = w_front * df["front"] + w_belly * df["belly"] + w_back * df["back"]
    fly.name = "fly_equal_weight"

    weights = FlyWeights(
        w_front=w_front,
        w_belly=w_belly,
        w_back=w_back,
        method="equal_weight",
        window=None,
        meta={},
    )

    return fly, weights
