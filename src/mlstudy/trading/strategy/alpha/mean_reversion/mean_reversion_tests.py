"""Mean reversion statistical tests and diagnostics.

This module provides statistical tests to assess whether a time series
(e.g., fly yield, fly residual, or any financial spread) exhibits
mean-reverting behavior.

Key concepts:
- ADF test: Null hypothesis is unit root (non-stationary). Rejecting suggests stationarity.
- KPSS test: Null hypothesis is stationarity. Rejecting suggests non-stationarity.
- AR(1) half-life: Measures how quickly deviations from mean decay.

Typical usage:
    >>> from mlstudy.trading.strategy.mean_reversion import mean_reversion_diagnostics
    >>> result = mean_reversion_diagnostics(fly_yield_series)
    >>> print(f"Half-life: {result['ar1']['half_life']:.1f} bars")
    >>> print(f"Evidence: {result['mr_evidence']}")
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def fit_ar1(
    series: pd.Series,
    *,
    min_obs: int = 30,
) -> Dict[str, Union[int, float, bool, None]]:
    """Fit AR(1) model and estimate mean-reversion parameters.

    Fits the model: x_t = a + b * x_{t-1} + e_t using OLS regression.

    Args:
        series: Time series to analyze (e.g., fly yield).
        min_obs: Minimum observations required after dropping NaNs.

    Returns:
        Dict with:
        - n_obs: Number of observations used
        - a: Intercept
        - b: AR(1) coefficient (persistence)
        - sigma: Standard deviation of residuals
        - t_stat_b: T-statistic for b (None if not computed)
        - half_life: Half-life in bars (None if not mean-reverting)
        - is_mean_reverting: True if 0 < b < 1

    Raises:
        ValueError: If fewer than min_obs valid observations.

    Notes:
        Half-life formula: -ln(2) / ln(b), valid only when 0 < b < 1.
        A shorter half-life indicates faster mean reversion.

    Example:
        >>> result = fit_ar1(fly_yield, min_obs=30)
        >>> if result["is_mean_reverting"]:
        ...     print(f"Half-life: {result['half_life']:.1f} bars")
    """
    # Clean NaNs
    clean = series.dropna().values
    n_obs = len(clean)

    if n_obs < min_obs:
        raise ValueError(
            f"Insufficient observations: {n_obs} < {min_obs} required. "
            f"Series has {len(series)} total values, {len(series) - n_obs} NaNs."
        )

    # Need at least 2 points for AR(1)
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")

    # Prepare lagged data: y = x[1:], x_lag = x[:-1]
    y = clean[1:]
    x_lag = clean[:-1]
    n = len(y)

    # Handle near-constant series
    x_std = np.std(x_lag)
    if x_std < 1e-12:
        # Near-constant series: b is undefined, not mean-reverting
        return {
            "n_obs": n_obs,
            "a": float(np.mean(y)),
            "b": 0.0,
            "sigma": float(np.std(y)) if np.std(y) > 0 else 0.0,
            "t_stat_b": None,
            "half_life": None,
            "is_mean_reverting": False,
        }

    # OLS regression: y = a + b * x_lag
    # Using normal equations: [a, b] = (X'X)^-1 X'y
    # where X = [1, x_lag]
    X = np.column_stack([np.ones(n), x_lag])
    XtX = X.T @ X
    Xty = X.T @ y

    # Solve using numpy's lstsq for numerical stability
    coeffs, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
    a, b = coeffs[0], coeffs[1]

    # Compute residuals and sigma
    y_pred = a + b * x_lag
    residuals = y - y_pred
    sigma = float(np.std(residuals, ddof=2))  # ddof=2 for 2 estimated params

    # Compute t-statistic for b
    try:
        # Variance of b: sigma^2 * (X'X)^-1[1,1]
        XtX_inv = np.linalg.inv(XtX)
        var_b = (sigma ** 2) * XtX_inv[1, 1]
        se_b = np.sqrt(max(var_b, 1e-20))  # Prevent sqrt of negative
        t_stat_b = float(b / se_b) if se_b > 1e-12 else None
    except np.linalg.LinAlgError:
        t_stat_b = None

    # Determine mean reversion
    is_mean_reverting = 0 < b < 1

    # Compute half-life
    if is_mean_reverting:
        # half_life = -ln(2) / ln(b)
        half_life = float(-np.log(2) / np.log(b))
    else:
        half_life = None

    return {
        "n_obs": n_obs,
        "a": float(a),
        "b": float(b),
        "sigma": sigma,
        "t_stat_b": t_stat_b,
        "half_life": half_life,
        "is_mean_reverting": is_mean_reverting,
    }


def adf_test(
    series: pd.Series,
    *,
    maxlag: Optional[int] = None,
    regression: str = "c",
    autolag: str = "AIC",
    min_obs: int = 30,
) -> Dict[str, Union[bool, int, float, str, Dict[str, float], None]]:
    """Perform Augmented Dickey-Fuller test for stationarity.

    The ADF test null hypothesis is that the series has a unit root
    (is non-stationary). Rejecting the null suggests stationarity,
    which is consistent with mean reversion.

    Args:
        series: Time series to test.
        maxlag: Maximum lag to consider. If None, uses automatic selection.
        regression: Regression type: "c" (constant), "ct" (constant+trend),
            "ctt" (constant+linear+quadratic trend), "n" (no constant).
        autolag: Lag selection method: "AIC", "BIC", "t-stat", or None.
        min_obs: Minimum observations required.

    Returns:
        Dict with:
        - available: True if statsmodels is installed
        - reason: Error message if not available
        - n_obs: Number of observations
        - test_stat: ADF test statistic
        - p_value: P-value for the test
        - used_lag: Number of lags used
        - critical_values: Dict of critical values at 1%, 5%, 10%
        - reject_unit_root_5pct: True if we reject unit root at 5% level

    Notes:
        - Rejecting the null (low p-value) suggests mean reversion
        - Common threshold: p_value < 0.05 to reject unit root

    Example:
        >>> result = adf_test(fly_yield)
        >>> if result["available"] and result["reject_unit_root_5pct"]:
        ...     print("Evidence of stationarity (mean reversion)")
    """
    # Clean NaNs
    clean = series.dropna().values
    n_obs = len(clean)

    if n_obs < min_obs:
        return {
            "available": False,
            "reason": f"Insufficient observations: {n_obs} < {min_obs}",
            "n_obs": n_obs,
        }

    # Try to import statsmodels
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {
            "available": False,
            "reason": "statsmodels not installed",
            "n_obs": n_obs,
        }

    # Handle near-constant series
    if np.std(clean) < 1e-12:
        return {
            "available": True,
            "n_obs": n_obs,
            "test_stat": float("-inf"),
            "p_value": 0.0,
            "used_lag": 0,
            "critical_values": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
            "reject_unit_root_5pct": True,  # Constant is stationary
        }

    # Run ADF test
    try:
        result = adfuller(
            clean,
            maxlag=maxlag,
            regression=regression,
            autolag=autolag,
        )
        test_stat = float(result[0])
        p_value = float(result[1])
        used_lag = int(result[2])
        critical_values = {k: float(v) for k, v in result[4].items()}

        # Reject unit root if test stat < critical value at 5%
        reject_5pct = test_stat < critical_values.get("5%", -2.86)

        return {
            "available": True,
            "n_obs": n_obs,
            "test_stat": test_stat,
            "p_value": p_value,
            "used_lag": used_lag,
            "critical_values": critical_values,
            "reject_unit_root_5pct": reject_5pct,
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"ADF test failed: {str(e)}",
            "n_obs": n_obs,
        }


def kpss_test(
    series: pd.Series,
    *,
    regression: str = "c",
    nlags: Union[str, int] = "auto",
    min_obs: int = 30,
) -> Dict[str, Union[bool, int, float, str, Dict[str, float], None]]:
    """Perform KPSS test for stationarity.

    The KPSS test null hypothesis is that the series is stationary.
    Rejecting the null suggests non-stationarity (e.g., unit root or trend),
    which is inconsistent with mean reversion.

    Args:
        series: Time series to test.
        regression: "c" for level stationarity, "ct" for trend stationarity.
        nlags: Number of lags for HAC variance. "auto" or "legacy" for automatic.
        min_obs: Minimum observations required.

    Returns:
        Dict with:
        - available: True if statsmodels is installed
        - reason: Error message if not available
        - n_obs: Number of observations
        - test_stat: KPSS test statistic
        - p_value: P-value for the test
        - used_lags: Number of lags used
        - critical_values: Dict of critical values at 10%, 5%, 2.5%, 1%
        - reject_stationarity_5pct: True if we reject stationarity at 5%

    Notes:
        - NOT rejecting the null (high p-value) suggests mean reversion
        - Rejecting suggests the series is non-stationary

    Example:
        >>> result = kpss_test(fly_yield)
        >>> if result["available"] and not result["reject_stationarity_5pct"]:
        ...     print("Evidence of stationarity (mean reversion)")
    """
    # Clean NaNs
    clean = series.dropna().values
    n_obs = len(clean)

    if n_obs < min_obs:
        return {
            "available": False,
            "reason": f"Insufficient observations: {n_obs} < {min_obs}",
            "n_obs": n_obs,
        }

    # Try to import statsmodels
    try:
        from statsmodels.tsa.stattools import kpss
    except ImportError:
        return {
            "available": False,
            "reason": "statsmodels not installed",
            "n_obs": n_obs,
        }

    # Handle near-constant series
    if np.std(clean) < 1e-12:
        return {
            "available": True,
            "n_obs": n_obs,
            "test_stat": 0.0,
            "p_value": 0.1,  # Above typical thresholds
            "used_lags": 0,
            "critical_values": {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
            "reject_stationarity_5pct": False,  # Constant is stationary
        }

    # Run KPSS test
    try:
        # Suppress the interpolation warning from statsmodels
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(clean, regression=regression, nlags=nlags)

        test_stat = float(result[0])
        p_value = float(result[1])
        used_lags = int(result[2])
        critical_values = {k: float(v) for k, v in result[3].items()}

        # Reject stationarity if test stat > critical value at 5%
        reject_5pct = test_stat > critical_values.get("5%", 0.463)

        return {
            "available": True,
            "n_obs": n_obs,
            "test_stat": test_stat,
            "p_value": p_value,
            "used_lags": used_lags,
            "critical_values": critical_values,
            "reject_stationarity_5pct": reject_5pct,
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"KPSS test failed: {str(e)}",
            "n_obs": n_obs,
        }


def mean_reversion_diagnostics(
    series: pd.Series,
    *,
    min_obs: int = 60,
) -> Dict[str, Union[int, str, Dict]]:
    """Run comprehensive mean reversion diagnostics on a time series.

    Combines AR(1) fitting with ADF and KPSS stationarity tests to provide
    an overall assessment of mean reversion evidence.

    Args:
        series: Time series to analyze (e.g., fly yield or spread).
        min_obs: Minimum observations for valid analysis.

    Returns:
        Dict with:
        - n_obs: Number of clean observations
        - ar1: Results from fit_ar1()
        - adf: Results from adf_test()
        - kpss: Results from kpss_test()
        - mr_evidence: Summary assessment:
            - "strong": AR1 mean-reverting AND (ADF rejects unit root OR KPSS doesn't reject stationarity)
            - "mixed": AR1 mean-reverting but stationarity tests disagree/unavailable
            - "weak": AR1 not mean-reverting or stationarity tests suggest non-stationarity
            - "insufficient_data": Not enough observations

    Example:
        >>> result = mean_reversion_diagnostics(fly_yield)
        >>> print(f"Mean reversion evidence: {result['mr_evidence']}")
        >>> if result['ar1']['half_life']:
        ...     print(f"Half-life: {result['ar1']['half_life']:.1f} bars")
    """
    # Clean NaNs
    clean = series.dropna()
    n_obs = len(clean)

    # Check minimum observations
    if n_obs < min_obs:
        return {
            "n_obs": n_obs,
            "ar1": {"error": f"Insufficient observations: {n_obs} < {min_obs}"},
            "adf": {"available": False, "reason": "Insufficient observations"},
            "kpss": {"available": False, "reason": "Insufficient observations"},
            "mr_evidence": "insufficient_data",
        }

    # Run AR(1) fit
    try:
        ar1_result = fit_ar1(series, min_obs=min_obs)
    except ValueError as e:
        return {
            "n_obs": n_obs,
            "ar1": {"error": str(e)},
            "adf": {"available": False, "reason": "AR1 failed"},
            "kpss": {"available": False, "reason": "AR1 failed"},
            "mr_evidence": "insufficient_data",
        }

    # Run stationarity tests
    adf_result = adf_test(series, min_obs=min_obs)
    kpss_result = kpss_test(series, min_obs=min_obs)

    # Determine evidence level
    ar1_mr = ar1_result.get("is_mean_reverting", False)

    adf_available = adf_result.get("available", False)
    adf_rejects = adf_result.get("reject_unit_root_5pct", False)

    kpss_available = kpss_result.get("available", False)
    kpss_rejects = kpss_result.get("reject_stationarity_5pct", True)

    # Determine mr_evidence
    if not ar1_mr:
        # AR1 coefficient >= 1 or <= 0: not mean-reverting
        mr_evidence = "weak"
    elif adf_available and kpss_available:
        # Both tests available
        if adf_rejects and not kpss_rejects:
            # Both tests agree: strong evidence
            mr_evidence = "strong"
        elif adf_rejects or not kpss_rejects:
            # One test supports, one doesn't
            mr_evidence = "mixed"
        else:
            # Neither test supports stationarity
            mr_evidence = "weak"
    elif adf_available:
        # Only ADF available
        mr_evidence = "strong" if adf_rejects else "mixed"
    elif kpss_available:
        # Only KPSS available
        mr_evidence = "strong" if not kpss_rejects else "mixed"
    else:
        # No stationarity tests available, rely on AR1
        mr_evidence = "mixed"

    return {
        "n_obs": n_obs,
        "ar1": ar1_result,
        "adf": adf_result,
        "kpss": kpss_result,
        "mr_evidence": mr_evidence,
    }


def compute_half_life(series: pd.Series, *, min_obs: int = 30) -> Optional[float]:
    """Convenience function to compute mean reversion half-life.

    Args:
        series: Time series to analyze.
        min_obs: Minimum observations required.

    Returns:
        Half-life in bars, or None if not mean-reverting or insufficient data.

    Example:
        >>> hl = compute_half_life(fly_yield)
        >>> if hl:
        ...     print(f"Deviations decay by half in {hl:.1f} bars")
    """
    try:
        result = fit_ar1(series, min_obs=min_obs)
        return result.get("half_life")
    except ValueError:
        return None


def is_mean_reverting(
    series: pd.Series,
    *,
    min_obs: int = 60,
    require_strong: bool = False,
) -> bool:
    """Quick check if series shows mean-reverting behavior.

    Args:
        series: Time series to test.
        min_obs: Minimum observations required.
        require_strong: If True, requires "strong" evidence; otherwise
            accepts "strong" or "mixed".

    Returns:
        True if mean reversion evidence meets threshold.

    Example:
        >>> if is_mean_reverting(fly_yield):
        ...     print("Series appears mean-reverting")
    """
    try:
        result = mean_reversion_diagnostics(series, min_obs=min_obs)
        evidence = result.get("mr_evidence", "insufficient_data")

        if require_strong:
            return evidence == "strong"
        else:
            return evidence in ("strong", "mixed")
    except Exception:
        return False
