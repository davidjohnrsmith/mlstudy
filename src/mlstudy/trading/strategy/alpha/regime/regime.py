"""Regime detection for mean-reversion strategies.

Provides rolling statistical tests to classify whether a series is in a
mean-reverting, trending, or uncertain regime. Used to gate mean-reversion
entries and manage exits during regime transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Regime(Enum):
    """Detected regime type."""

    MEAN_REVERT = "mean_revert"
    TREND = "trend"
    UNCERTAIN = "uncertain"


@dataclass
class RegimeConfig:
    """Configuration for regime detection.

    Attributes:
        window: Rolling window size for all tests.
        adf_threshold: P-value threshold for ADF test. Below this = stationary.
        kpss_threshold: P-value threshold for KPSS test. Above this = stationary.
        half_life_max: Maximum half-life (days) to consider mean-reverting.
        half_life_min: Minimum half-life to avoid noise.
        trend_r2_threshold: R² threshold for trend detection.
        trend_slope_threshold: Minimum absolute slope for trend detection.
        use_adf: Whether to use ADF test.
        use_kpss: Whether to use KPSS test.
        use_half_life: Whether to use half-life criterion.
        use_trend: Whether to use trend detection.
        require_all: If True, all enabled tests must agree for classification.
            If False, majority voting is used.

    Example:
        >>> config = RegimeConfig(
        ...     window=60,
        ...     adf_threshold=0.05,
        ...     half_life_max=20,
        ... )
    """

    window: int = 60
    adf_threshold: float = 0.05
    kpss_threshold: float = 0.05
    half_life_max: float = 30.0
    half_life_min: float = 1.0
    trend_r2_threshold: float = 0.3
    trend_slope_threshold: float = 0.0
    use_adf: bool = True
    use_kpss: bool = False  # Often redundant with ADF
    use_half_life: bool = True
    use_trend: bool = True
    require_all: bool = False


@dataclass
class RegimeResult:
    """Result of regime classification.

    Attributes:
        regime: Detected regime.
        adf_pvalue: ADF test p-value (if computed).
        kpss_pvalue: KPSS test p-value (if computed).
        half_life: Estimated half-life in periods (if computed).
        trend_slope: Linear trend slope (if computed).
        trend_r2: R² of linear fit (if computed).
        scores: Dict of individual test scores/votes.
    """

    regime: Regime
    adf_pvalue: float | None = None
    kpss_pvalue: float | None = None
    half_life: float | None = None
    trend_slope: float | None = None
    trend_r2: float | None = None
    scores: dict | None = None


def _adf_test(series: np.ndarray) -> float:
    """Compute ADF test p-value.

    Uses a simplified ADF test without external dependencies.
    For production, consider using statsmodels.tsa.stattools.adfuller.

    Args:
        series: Time series array.

    Returns:
        Approximate p-value (lower = more stationary).
    """
    n = len(series)
    if n < 10:
        return 1.0

    # Compute first differences
    diff = np.diff(series)
    lagged = series[:-1]

    # Simple OLS: diff = alpha + beta * lagged + error
    # ADF tests if beta < 0 (mean-reverting)
    X = np.column_stack([np.ones(len(lagged)), lagged])
    try:
        beta = np.linalg.lstsq(X, diff, rcond=None)[0]
        residuals = diff - X @ beta

        # Estimate standard error
        mse = np.sum(residuals**2) / (n - 3)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(mse * XtX_inv[1, 1])

        # t-statistic for beta (coefficient on lagged level)
        t_stat = beta[1] / se_beta if se_beta > 0 else 0

        # Approximate p-value using critical values
        # ADF critical values (approximate): -3.5 (1%), -2.9 (5%), -2.6 (10%)
        if t_stat < -3.5:
            return 0.01
        elif t_stat < -2.9:
            return 0.05
        elif t_stat < -2.6:
            return 0.10
        elif t_stat < -1.95:
            return 0.20
        else:
            return 0.50

    except (np.linalg.LinAlgError, ValueError):
        return 1.0


def _kpss_test(series: np.ndarray) -> float:
    """Compute KPSS test p-value.

    Simplified KPSS test. Null hypothesis is stationarity.

    Args:
        series: Time series array.

    Returns:
        Approximate p-value (higher = more stationary).
    """
    n = len(series)
    if n < 10:
        return 0.0

    # Demean the series
    demeaned = series - np.mean(series)

    # Compute cumulative sum of residuals
    cumsum = np.cumsum(demeaned)

    # KPSS statistic
    s2 = np.sum(demeaned**2) / n

    # Long-run variance estimate (simple)
    lrv = s2  # Simplified; proper KPSS uses Newey-West

    if lrv == 0:
        return 0.0

    kpss_stat = np.sum(cumsum**2) / (n**2 * lrv)

    # Approximate p-value using critical values
    # KPSS critical values: 0.347 (10%), 0.463 (5%), 0.739 (1%)
    if kpss_stat > 0.739:
        return 0.01
    elif kpss_stat > 0.463:
        return 0.05
    elif kpss_stat > 0.347:
        return 0.10
    else:
        return 0.50


def _estimate_ou_half_life(series: np.ndarray) -> float:
    """Estimate Ornstein-Uhlenbeck half-life.

    Fits AR(1) model: x_t = c + phi * x_{t-1} + epsilon
    Half-life = -log(2) / log(phi)

    Args:
        series: Time series array.

    Returns:
        Estimated half-life in periods. Returns inf if not mean-reverting.
    """
    n = len(series)
    if n < 10:
        return float("inf")

    # AR(1) regression
    y = series[1:]
    x = series[:-1]

    # OLS: y = c + phi * x
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = coeffs[1]

        # Half-life calculation
        if phi <= 0 or phi >= 1:
            return float("inf")

        half_life = -np.log(2) / np.log(phi)
        return max(half_life, 0.0)

    except (np.linalg.LinAlgError, ValueError):
        return float("inf")


def _compute_trend_score(series: np.ndarray) -> tuple[float, float]:
    """Compute linear trend slope and R².

    Args:
        series: Time series array.

    Returns:
        Tuple of (slope, r_squared).
    """
    n = len(series)
    if n < 3:
        return 0.0, 0.0

    # Time index
    t = np.arange(n)

    # Linear regression: series = a + b * t
    X = np.column_stack([np.ones(n), t])
    try:
        coeffs = np.linalg.lstsq(X, series, rcond=None)[0]
        slope = coeffs[1]

        # Compute R²
        predicted = X @ coeffs
        ss_res = np.sum((series - predicted) ** 2)
        ss_tot = np.sum((series - np.mean(series)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        return slope, r_squared

    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0


def rolling_adf_pvalue(
    series: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling ADF test p-value.

    Args:
        series: Time series.
        window: Rolling window size.

    Returns:
        Series of p-values (lower = more stationary/mean-reverting).
    """
    result = pd.Series(index=series.index, dtype=float)

    values = series.values
    for i in range(window - 1, len(values)):
        window_data = values[i - window + 1 : i + 1]
        result.iloc[i] = _adf_test(window_data)

    return result


def rolling_kpss_pvalue(
    series: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling KPSS test p-value.

    Args:
        series: Time series.
        window: Rolling window size.

    Returns:
        Series of p-values (higher = more stationary).
    """
    result = pd.Series(index=series.index, dtype=float)

    values = series.values
    for i in range(window - 1, len(values)):
        window_data = values[i - window + 1 : i + 1]
        result.iloc[i] = _kpss_test(window_data)

    return result


def rolling_ou_half_life(
    series: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling Ornstein-Uhlenbeck half-life.

    Args:
        series: Time series.
        window: Rolling window size.

    Returns:
        Series of half-life estimates in periods.
    """
    result = pd.Series(index=series.index, dtype=float)

    values = series.values
    for i in range(window - 1, len(values)):
        window_data = values[i - window + 1 : i + 1]
        result.iloc[i] = _estimate_ou_half_life(window_data)

    return result


def rolling_trend_score(
    series: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Compute rolling linear trend slope and R².

    Args:
        series: Time series.
        window: Rolling window size.

    Returns:
        DataFrame with columns ['slope', 'r_squared'].
    """
    slopes = pd.Series(index=series.index, dtype=float)
    r_squareds = pd.Series(index=series.index, dtype=float)

    values = series.values
    for i in range(window - 1, len(values)):
        window_data = values[i - window + 1 : i + 1]
        slope, r2 = _compute_trend_score(window_data)
        slopes.iloc[i] = slope
        r_squareds.iloc[i] = r2

    return pd.DataFrame({"slope": slopes, "r_squared": r_squareds})


def classify_regime_single(
    series: np.ndarray,
    config: RegimeConfig | None = None,
) -> RegimeResult:
    """Classify regime for a single window of data.

    Args:
        series: Time series array (single window).
        config: Regime detection configuration.

    Returns:
        RegimeResult with classification and test results.
    """
    if config is None:
        config = RegimeConfig()

    scores = {}
    mr_votes = 0
    trend_votes = 0
    total_tests = 0

    # ADF test
    adf_pvalue = None
    if config.use_adf:
        adf_pvalue = _adf_test(series)
        scores["adf"] = "mean_revert" if adf_pvalue < config.adf_threshold else "uncertain"
        if adf_pvalue < config.adf_threshold:
            mr_votes += 1
        total_tests += 1

    # KPSS test
    kpss_pvalue = None
    if config.use_kpss:
        kpss_pvalue = _kpss_test(series)
        scores["kpss"] = "mean_revert" if kpss_pvalue > config.kpss_threshold else "uncertain"
        if kpss_pvalue > config.kpss_threshold:
            mr_votes += 1
        total_tests += 1

    # Half-life
    half_life = None
    if config.use_half_life:
        half_life = _estimate_ou_half_life(series)
        if config.half_life_min <= half_life <= config.half_life_max:
            scores["half_life"] = "mean_revert"
            mr_votes += 1
        elif half_life > config.half_life_max * 2:
            scores["half_life"] = "trend"
            trend_votes += 1
        else:
            scores["half_life"] = "uncertain"
        total_tests += 1

    # Trend detection
    trend_slope = None
    trend_r2 = None
    if config.use_trend:
        trend_slope, trend_r2 = _compute_trend_score(series)
        # Normalize slope by series std for comparability
        series_std = np.std(series)
        norm_slope = abs(trend_slope) / series_std if series_std > 0 else 0

        if trend_r2 > config.trend_r2_threshold and norm_slope > config.trend_slope_threshold:
            scores["trend"] = "trend"
            trend_votes += 1
        elif trend_r2 < config.trend_r2_threshold * 0.5:
            scores["trend"] = "mean_revert"
            mr_votes += 1
        else:
            scores["trend"] = "uncertain"
        total_tests += 1

    # Determine final regime
    if total_tests == 0:
        regime = Regime.UNCERTAIN
    elif config.require_all:
        # All tests must agree
        if mr_votes == total_tests:
            regime = Regime.MEAN_REVERT
        elif trend_votes == total_tests:
            regime = Regime.TREND
        else:
            regime = Regime.UNCERTAIN
    else:
        # Majority voting
        if mr_votes > trend_votes and mr_votes > total_tests / 2:
            regime = Regime.MEAN_REVERT
        elif trend_votes > mr_votes and trend_votes > total_tests / 2:
            regime = Regime.TREND
        else:
            regime = Regime.UNCERTAIN

    return RegimeResult(
        regime=regime,
        adf_pvalue=adf_pvalue,
        kpss_pvalue=kpss_pvalue,
        half_life=half_life,
        trend_slope=trend_slope,
        trend_r2=trend_r2,
        scores=scores,
    )


def classify_regime(
    series: pd.Series,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Classify regime over rolling windows.

    Args:
        series: Time series.
        config: Regime detection configuration.

    Returns:
        Series of Regime enum values.
    """
    if config is None:
        config = RegimeConfig()

    result = pd.Series(index=series.index, dtype=object)

    values = series.values
    for i in range(config.window - 1, len(values)):
        window_data = values[i - config.window + 1 : i + 1]
        regime_result = classify_regime_single(window_data, config)
        result.iloc[i] = regime_result.regime

    # Fill early values with UNCERTAIN
    result.iloc[: config.window - 1] = Regime.UNCERTAIN

    return result


def compute_regime_features(
    series: pd.Series,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Compute all regime-related features.

    Args:
        series: Time series.
        config: Regime detection configuration.

    Returns:
        DataFrame with columns: regime, adf_pvalue, half_life, trend_slope, trend_r2.
    """
    if config is None:
        config = RegimeConfig()

    records = []
    values = series.values

    for i in range(len(values)):
        if i < config.window - 1:
            records.append({
                "regime": Regime.UNCERTAIN,
                "adf_pvalue": None,
                "half_life": None,
                "trend_slope": None,
                "trend_r2": None,
            })
        else:
            window_data = values[i - config.window + 1 : i + 1]
            result = classify_regime_single(window_data, config)
            records.append({
                "regime": result.regime,
                "adf_pvalue": result.adf_pvalue,
                "half_life": result.half_life,
                "trend_slope": result.trend_slope,
                "trend_r2": result.trend_r2,
            })

    df = pd.DataFrame(records, index=series.index)
    return df


def apply_regime_gate(
    signal: pd.Series,
    regime: pd.Series,
    allow_regimes: list[Regime] | None = None,
    exit_on_regime_change: bool = False,
) -> pd.Series:
    """Apply regime gate to trading signal.

    Gates entries based on regime classification. Optionally forces
    exits when regime changes while in a position.

    Args:
        signal: Trading signal series (-1, 0, 1).
        regime: Regime classification series.
        allow_regimes: List of regimes that allow entries.
            Default: [Regime.MEAN_REVERT].
        exit_on_regime_change: If True, force exit when regime
            changes to non-allowed while in position.

    Returns:
        Gated signal series.
    """
    if allow_regimes is None:
        allow_regimes = [Regime.MEAN_REVERT]

    gated = signal.copy()
    position = 0

    for i in range(len(signal)):
        current_regime = regime.iloc[i]
        current_signal = signal.iloc[i]

        # Check if regime allows entries
        regime_allows = current_regime in allow_regimes

        if position == 0:
            # Not in position - only enter if regime allows
            if regime_allows and current_signal != 0:
                gated.iloc[i] = current_signal
                position = current_signal
            else:
                gated.iloc[i] = 0
        else:
            # In position
            if exit_on_regime_change and not regime_allows:
                # Force exit due to regime change
                gated.iloc[i] = 0
                position = 0
            elif current_signal == 0:
                # Signal says exit
                gated.iloc[i] = 0
                position = 0
            elif current_signal != position:
                # Signal reversal
                if regime_allows:
                    gated.iloc[i] = current_signal
                    position = current_signal
                else:
                    # Exit but don't enter new position
                    gated.iloc[i] = 0
                    position = 0
            else:
                # Continue holding
                gated.iloc[i] = position

    return gated


def generate_ou_process(
    n: int,
    theta: float = 0.1,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic Ornstein-Uhlenbeck process.

    dx = theta * (mu - x) * dt + sigma * dW

    Args:
        n: Number of observations.
        theta: Mean-reversion speed (higher = faster reversion).
        mu: Long-term mean.
        sigma: Volatility.
        x0: Initial value.
        seed: Random seed for reproducibility.

    Returns:
        Array of OU process values.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0
    x = np.zeros(n)
    x[0] = x0

    for i in range(1, n):
        dx = theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        x[i] = x[i - 1] + dx

    return x


def generate_random_walk_with_drift(
    n: int,
    drift: float = 0.01,
    sigma: float = 1.0,
    x0: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random walk with drift (trending process).

    dx = drift * dt + sigma * dW

    Args:
        n: Number of observations.
        drift: Drift per period.
        sigma: Volatility.
        x0: Initial value.
        seed: Random seed for reproducibility.

    Returns:
        Array of random walk values.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.zeros(n)
    x[0] = x0

    for i in range(1, n):
        x[i] = x[i - 1] + drift + sigma * np.random.randn()

    return x
