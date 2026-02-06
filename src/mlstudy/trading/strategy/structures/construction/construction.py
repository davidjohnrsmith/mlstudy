"""Shared construction primitives for spread trading.

This module provides low-level construction functions for multi-leg
spreads (butterflies, etc.) with weights optimized for stationarity.

These are shared utilities used by fly_construction and potentially
other spread construction modules.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd


# =============================================================================
# Input validation
# =============================================================================


def _validate_3leg(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Align and validate three series for 3-leg construction.

    Args:
        front: Front leg series.
        belly: Belly leg series.
        back: Back leg series.
        min_obs: Minimum observations required.

    Returns:
        DataFrame with aligned 'front', 'belly', 'back' columns, NaNs dropped.

    Raises:
        ValueError: If fewer than min_obs valid observations.
    """
    df = pd.DataFrame(
        {"front": front, "belly": belly, "back": back}
    ).dropna()

    n_obs = len(df)
    if n_obs < min_obs:
        raise ValueError(
            f"Insufficient observations after alignment: {n_obs} < {min_obs}"
        )

    return df


# =============================================================================
# OLS helpers
# =============================================================================


def _rolling_ols(
    y: np.ndarray,
    X: np.ndarray,
    window: int,
    ridge_lambda: Optional[float] = None,
) -> np.ndarray:
    """Compute rolling OLS coefficients.

    No lookahead: coefficients at time t use data from t-window+1 to t.
    """
    n, k = X.shape
    coeffs = np.full((n, k), np.nan)

    for t in range(window - 1, n):
        start = t - window + 1
        end = t + 1
        y_win = y[start:end]
        X_win = X[start:end]

        XtX = X_win.T @ X_win
        Xty = X_win.T @ y_win

        if ridge_lambda is not None:
            XtX = XtX + ridge_lambda * np.eye(k)

        try:
            coeffs[t] = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coeffs[t] = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    return coeffs


def _static_ols(
    y: np.ndarray,
    X: np.ndarray,
    ridge_lambda: Optional[float] = None,
) -> np.ndarray:
    """Compute static (full-sample) OLS coefficients."""
    XtX = X.T @ X
    Xty = X.T @ y

    if ridge_lambda is not None:
        XtX = XtX + ridge_lambda * np.eye(X.shape[1])

    try:
        return np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(XtX, Xty, rcond=None)[0]


# =============================================================================
# AR(1) half-life
# =============================================================================


def fit_ar1_half_life(
    series: pd.Series,
    *,
    min_obs: int = 60,
) -> Dict[str, Optional[float]]:
    """Fit AR(1) model and compute half-life for mean reversion.

    Fits: x_t = a + b * x_{t-1} + e_t using OLS.

    Args:
        series: Time series to analyze.
        min_obs: Minimum observations required.

    Returns:
        Dict with:
        - b: AR(1) coefficient (persistence parameter)
        - half_life: -ln(2)/ln(b) if 0 < b < 1, else None
        - residual_std: Standard deviation of residuals

    Raises:
        ValueError: If insufficient observations.
    """
    clean = series.dropna().values
    n_obs = len(clean)

    if n_obs < min_obs:
        raise ValueError(f"Insufficient observations: {n_obs} < {min_obs}")

    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")

    y = clean[1:]
    x_lag = clean[:-1]
    n = len(y)

    x_std = np.std(x_lag)
    if x_std < 1e-12:
        return {"b": 0.0, "half_life": None, "residual_std": 0.0}

    X = np.column_stack([np.ones(n), x_lag])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs[0], coeffs[1]

    y_pred = a + b * x_lag
    residuals = y - y_pred
    residual_std = float(np.std(residuals))

    b = float(b)
    if 0 < b < 1:
        half_life = float(-np.log(2) / np.log(b))
    else:
        half_life = None

    return {"b": b, "half_life": half_life, "residual_std": residual_std}


# =============================================================================
# DV01-neutral enforcement
# =============================================================================


def enforce_dv01_neutral(
    w_front: float,
    w_belly: float,
    w_back: float,
    dv01_front: float,
    dv01_belly: float,
    dv01_back: float,
    *,
    adjust_leg: Literal["front", "back"] = "back",
) -> Tuple[float, float, float]:
    """Adjust one leg weight to achieve DV01 neutrality.

    Solves: w_front*dv01_front + w_belly*dv01_belly + w_back*dv01_back = 0

    Args:
        w_front: Weight on front leg.
        w_belly: Weight on belly leg.
        w_back: Weight on back leg.
        dv01_front: DV01 of front leg.
        dv01_belly: DV01 of belly leg.
        dv01_back: DV01 of back leg.
        adjust_leg: Which leg to adjust ("front" or "back").

    Returns:
        Tuple (w_front_adj, w_belly, w_back_adj) with adjusted weights.

    Raises:
        ValueError: If the adjustment leg has near-zero DV01.
    """
    EPS = 1e-12

    if adjust_leg == "back":
        if abs(dv01_back) < EPS:
            raise ValueError(
                f"Cannot adjust back leg: dv01_back ({dv01_back}) is near zero."
            )
        w_back_adj = -(w_front * dv01_front + w_belly * dv01_belly) / dv01_back
        return (w_front, w_belly, w_back_adj)

    elif adjust_leg == "front":
        if abs(dv01_front) < EPS:
            raise ValueError(
                f"Cannot adjust front leg: dv01_front ({dv01_front}) is near zero."
            )
        w_front_adj = -(w_belly * dv01_belly + w_back * dv01_back) / dv01_front
        return (w_front_adj, w_belly, w_back)

    else:
        raise ValueError(f"adjust_leg must be 'front' or 'back', got '{adjust_leg}'")


# =============================================================================
# Levels residual (cointegration-style OLS)
# =============================================================================


def levels_residual_3leg(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int = 252,
    rolling: bool = True,
    min_obs: int = 60,
    add_intercept: bool = True,
    ridge_lambda: Optional[float] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Construct spread as regression residual on levels.

    Regresses belly on front and back:
        belly_t = c + beta_front * front_t + beta_back * back_t + e_t

    Args:
        front: Front leg series.
        belly: Belly leg series.
        back: Back leg series.
        window: Rolling window size.
        rolling: If True, use rolling window; if False, use full sample.
        min_obs: Minimum observations for validation.
        add_intercept: If True, include intercept in regression.
        ridge_lambda: Ridge regularization parameter (optional).

    Returns:
        Tuple of:
        - fly: pd.Series of residuals
        - weights_df: pd.DataFrame with columns: beta_front, beta_back, intercept

    Notes:
        - NO LOOKAHEAD: weights at time t use data up to t only.
    """
    df = _validate_3leg(front, belly, back, min_obs=min_obs)
    n = len(df)
    idx = df.index

    y = df["belly"].values
    front_vals = df["front"].values
    back_vals = df["back"].values

    if add_intercept:
        X = np.column_stack([np.ones(n), front_vals, back_vals])
        col_names = ["intercept", "beta_front", "beta_back"]
    else:
        X = np.column_stack([front_vals, back_vals])
        col_names = ["beta_front", "beta_back"]

    if rolling:
        if window > n:
            raise ValueError(f"Window ({window}) exceeds observations ({n})")
        coeffs = _rolling_ols(y, X, window, ridge_lambda)
    else:
        static_coeffs = _static_ols(y, X, ridge_lambda)
        coeffs = np.tile(static_coeffs, (n, 1))

    weights_df = pd.DataFrame(coeffs, index=idx, columns=col_names)

    y_pred = np.sum(coeffs * X, axis=1)
    fly_values = y - y_pred
    fly = pd.Series(fly_values, index=idx, name="levels_residual")

    return fly, weights_df


# =============================================================================
# Changes residual (OLS on differences)
# =============================================================================


def changes_residual_3leg(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int = 252,
    rolling: bool = True,
    min_obs: int = 60,
    add_intercept: bool = True,
    ridge_lambda: Optional[float] = None,
    integrate: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Construct spread from regression on changes (differences).

    Regresses change in belly on changes in front and back:
        Δbelly_t = c + beta_front * Δfront_t + beta_back * Δback_t + e_t

    Args:
        front: Front leg series.
        belly: Belly leg series.
        back: Back leg series.
        window: Rolling window size.
        rolling: If True, use rolling window; if False, use full sample.
        min_obs: Minimum observations for validation.
        add_intercept: If True, include intercept in regression.
        ridge_lambda: Ridge regularization parameter (optional).
        integrate: If True, return cumulative sum of residuals.

    Returns:
        Tuple of:
        - fly: pd.Series of (integrated) residuals
        - weights_df: pd.DataFrame with columns: beta_front, beta_back, intercept

    Notes:
        - NO LOOKAHEAD: weights at time t use data up to t only.
    """
    df = _validate_3leg(front, belly, back, min_obs=min_obs)

    df_diff = df.diff().dropna()
    n = len(df_diff)
    idx = df_diff.index

    if n < min_obs:
        raise ValueError(f"Insufficient observations after diff: {n} < {min_obs}")

    y = df_diff["belly"].values
    front_vals = df_diff["front"].values
    back_vals = df_diff["back"].values

    if add_intercept:
        X = np.column_stack([np.ones(n), front_vals, back_vals])
        col_names = ["intercept", "beta_front", "beta_back"]
    else:
        X = np.column_stack([front_vals, back_vals])
        col_names = ["beta_front", "beta_back"]

    if rolling:
        if window > n:
            raise ValueError(f"Window ({window}) exceeds observations ({n})")
        coeffs = _rolling_ols(y, X, window, ridge_lambda)
    else:
        static_coeffs = _static_ols(y, X, ridge_lambda)
        coeffs = np.tile(static_coeffs, (n, 1))

    weights_df = pd.DataFrame(coeffs, index=idx, columns=col_names)

    y_pred = np.sum(coeffs * X, axis=1)
    residual_changes = y - y_pred

    if integrate:
        fly_values = np.nancumsum(np.nan_to_num(residual_changes, nan=0.0))
        fly_values = np.where(np.isnan(coeffs[:, 0]), np.nan, fly_values)
        fly_name = "changes_residual_integrated"
    else:
        fly_values = residual_changes
        fly_name = "changes_residual"

    fly = pd.Series(fly_values, index=idx, name=fly_name)

    return fly, weights_df


# =============================================================================
# Stationarity optimization
# =============================================================================


def _evaluate_half_life(
    fly_values: np.ndarray,
    min_obs: int = 30,
) -> float:
    """Evaluate half-life objective on a fly series."""
    LARGE_PENALTY = 1e10

    clean = fly_values[~np.isnan(fly_values)]
    if len(clean) < min_obs:
        return LARGE_PENALTY

    try:
        result = fit_ar1_half_life(pd.Series(clean), min_obs=min_obs)
        if result["half_life"] is None:
            return LARGE_PENALTY
        return result["half_life"]
    except (ValueError, np.linalg.LinAlgError):
        return LARGE_PENALTY


def optimize_stationarity_3leg(
    front: pd.Series,
    belly: pd.Series,
    back: pd.Series,
    *,
    window: int = 252,
    rolling: bool = False,
    min_obs: int = 120,
    w_belly: float = 1.0,
    bounds_front: Tuple[float, float] = (-3.0, 3.0),
    bounds_back: Tuple[float, float] = (-3.0, 3.0),
    grid_size: int = 21,
    objective: Literal["half_life"] = "half_life",
) -> Tuple[pd.Series, pd.DataFrame]:
    """Optimize weights to minimize half-life (maximize stationarity).

    Grid-searches over front and back weights with fixed belly weight.

    Args:
        front: Front leg series.
        belly: Belly leg series.
        back: Back leg series.
        window: Rolling window size for optimization.
        rolling: If True, re-optimize for each time point.
        min_obs: Minimum observations for objective evaluation.
        w_belly: Fixed weight on belly leg.
        bounds_front: (min, max) bounds for front leg weight.
        bounds_back: (min, max) bounds for back leg weight.
        grid_size: Number of grid points per dimension.
        objective: Optimization objective (currently only "half_life").

    Returns:
        Tuple of:
        - fly: pd.Series of optimized spread values
        - weights_df: pd.DataFrame with columns: w_front, w_belly, w_back, objective_value

    Notes:
        - NO LOOKAHEAD: optimization at time t uses data up to t only.
    """
    if objective != "half_life":
        raise ValueError(f"Unsupported objective: {objective}")

    df = _validate_3leg(front, belly, back, min_obs=min_obs)
    n = len(df)
    idx = df.index

    front_vals = df["front"].values
    belly_vals = df["belly"].values
    back_vals = df["back"].values

    w_front_grid = np.linspace(bounds_front[0], bounds_front[1], grid_size)
    w_back_grid = np.linspace(bounds_back[0], bounds_back[1], grid_size)

    def find_best_weights(
        front_win: np.ndarray,
        belly_win: np.ndarray,
        back_win: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Find best weights for a window of data."""
        best_obj = 1e10
        best_w_front = w_front_grid[len(w_front_grid) // 2]
        best_w_back = w_back_grid[len(w_back_grid) // 2]

        for w_f in w_front_grid:
            for w_b in w_back_grid:
                fly_win = w_f * front_win + w_belly * belly_win + w_b * back_win
                obj = _evaluate_half_life(fly_win, min_obs=30)

                if obj < best_obj:
                    best_obj = obj
                    best_w_front = w_f
                    best_w_back = w_b

        return best_w_front, best_w_back, best_obj

    weights_arr = np.full((n, 4), np.nan)  # w_front, w_belly, w_back, obj

    if rolling:
        if window > n:
            raise ValueError(f"Window ({window}) exceeds observations ({n})")

        for t in range(window - 1, n):
            start = t - window + 1
            end = t + 1

            w_f, w_b, obj_val = find_best_weights(
                front_vals[start:end],
                belly_vals[start:end],
                back_vals[start:end],
            )

            weights_arr[t, 0] = w_f
            weights_arr[t, 1] = w_belly
            weights_arr[t, 2] = w_b
            weights_arr[t, 3] = obj_val
    else:
        w_f, w_b, obj_val = find_best_weights(front_vals, belly_vals, back_vals)
        weights_arr[:, 0] = w_f
        weights_arr[:, 1] = w_belly
        weights_arr[:, 2] = w_b
        weights_arr[:, 3] = obj_val

    weights_df = pd.DataFrame(
        weights_arr,
        index=idx,
        columns=["w_front", "w_belly", "w_back", "objective_value"],
    )

    fly_values = (
        weights_arr[:, 0] * front_vals
        + weights_arr[:, 1] * belly_vals
        + weights_arr[:, 2] * back_vals
    )
    fly = pd.Series(fly_values, index=idx, name="optimized_spread")

    return fly, weights_df
