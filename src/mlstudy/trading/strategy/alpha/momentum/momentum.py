"""Momentum (trend-following) signals for fly trading.

Provides momentum-based trading signals as a complement to mean-reversion
signals. Signals are designed for intraday session trading with overnight
holding capability.

Signal convention:
- Positive score = bullish momentum (go long)
- Negative score = bearish momentum (go short)
- Signal at bar t executes at bar t+1 (no lookahead)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class TrendMethod(Enum):
    """Method for computing trend strength."""

    R2 = "r2"  # R-squared of linear fit
    TSTAT = "tstat"  # T-statistic of slope


@dataclass
class MomentumConfig:
    """Configuration for momentum signal generation.

    Attributes:
        lookback_bars: Lookback period for momentum calculation.
        smooth_bars: Optional smoothing period (EMA).
        fast_span: Fast EMA span for crossover signals.
        slow_span: Slow EMA span for crossover signals.
        breakout_window: Window for breakout detection.
        threshold: Threshold for converting score to position.
        clip: Optional clipping for continuous positions.

    Example:
        >>> config = MomentumConfig(
        ...     lookback_bars=20,
        ...     fast_span=5,
        ...     slow_span=20,
        ...     threshold=0.0,
        ... )
    """

    lookback_bars: int = 20
    smooth_bars: int | None = None
    fast_span: int = 5
    slow_span: int = 20
    breakout_window: int = 20
    threshold: float = 0.0
    clip: float | None = None


def ts_momentum_signal(
    series: pd.Series,
    lookback_bars: int,
    smooth: int | None = None,
    use_log_return: bool = False,
) -> pd.Series:
    """Compute time-series momentum signal.

    Measures the direction and magnitude of price movement over
    a lookback period. Positive values indicate upward momentum.

    Args:
        series: Input time series (e.g., fly yield).
        lookback_bars: Number of bars for momentum calculation.
        smooth: Optional EMA smoothing period for the signal.
        use_log_return: If True, use log returns instead of simple diff.

    Returns:
        Series of momentum scores. Positive = bullish, negative = bearish.

    Example:
        >>> mom = ts_momentum_signal(fly_yield, lookback_bars=20)
        >>> mom_smooth = ts_momentum_signal(fly_yield, lookback_bars=20, smooth=5)
    """
    if use_log_return:
        # Log return over lookback period
        score = np.log(series / series.shift(lookback_bars))
    else:
        # Simple difference
        score = series.diff(lookback_bars)

    if smooth is not None and smooth > 1:
        score = score.ewm(span=smooth, min_periods=1).mean()

    score.name = "ts_momentum"
    return score


def ema_crossover_signal(
    series: pd.Series,
    fast_span: int,
    slow_span: int,
    normalize: bool = True,
) -> pd.Series:
    """Compute EMA crossover signal.

    Signal is the difference between fast and slow EMAs.
    Positive when fast EMA is above slow EMA (bullish).

    Args:
        series: Input time series.
        fast_span: Span for fast (short-term) EMA.
        slow_span: Span for slow (long-term) EMA.
        normalize: If True, normalize by slow EMA std for comparability.

    Returns:
        Series of crossover scores. Positive = bullish crossover.

    Example:
        >>> xover = ema_crossover_signal(fly_yield, fast_span=5, slow_span=20)
    """
    if fast_span >= slow_span:
        raise ValueError(f"fast_span ({fast_span}) must be < slow_span ({slow_span})")

    ema_fast = series.ewm(span=fast_span, min_periods=fast_span).mean()
    ema_slow = series.ewm(span=slow_span, min_periods=slow_span).mean()

    score = ema_fast - ema_slow

    if normalize:
        # Normalize by rolling std of slow EMA for scale-invariance
        rolling_std = series.rolling(window=slow_span, min_periods=slow_span).std()
        score = score / rolling_std.replace(0, np.nan)

    score.name = "ema_crossover"
    return score


def breakout_signal(
    series: pd.Series,
    window_bars: int,
    use_close: bool = True,
) -> pd.Series:
    """Compute breakout signal based on rolling high/low.

    Returns +1 when series breaks above rolling high, -1 when below
    rolling low, 0 otherwise (Donchian channel breakout style).

    Args:
        series: Input time series.
        window_bars: Lookback window for high/low calculation.
        use_close: If True, compare current value to previous high/low
            (excludes current bar from channel calculation).

    Returns:
        Series of breakout signals: +1 (upside breakout), -1 (downside), 0 (none).

    Example:
        >>> breakout = breakout_signal(fly_yield, window_bars=20)
    """
    if use_close:
        # Exclude current bar from channel calculation (shift by 1)
        rolling_high = series.shift(1).rolling(window=window_bars, min_periods=window_bars).max()
        rolling_low = series.shift(1).rolling(window=window_bars, min_periods=window_bars).min()
    else:
        rolling_high = series.rolling(window=window_bars, min_periods=window_bars).max()
        rolling_low = series.rolling(window=window_bars, min_periods=window_bars).min()

    # Breakout detection
    signal = pd.Series(0, index=series.index, dtype=int)
    signal[series > rolling_high] = 1
    signal[series < rolling_low] = -1

    signal.name = "breakout"
    return signal


def trend_strength(
    series: pd.Series,
    window_bars: int,
    method: str = "r2",
) -> pd.Series:
    """Compute rolling trend strength measure.

    Quantifies how strongly the series is trending (vs mean-reverting).
    Higher values indicate stronger trend.

    Args:
        series: Input time series.
        window_bars: Rolling window for trend calculation.
        method: Method for trend strength:
            - "r2": R-squared of linear fit (0 to 1)
            - "tstat": T-statistic of slope (unbounded, can be negative)

    Returns:
        Series of trend strength values.

    Example:
        >>> strength = trend_strength(fly_yield, window_bars=60, method="r2")
    """
    method = method.lower()
    if method not in ("r2", "tstat"):
        raise ValueError(f"method must be 'r2' or 'tstat', got '{method}'")

    result = pd.Series(index=series.index, dtype=float)
    values = series.values
    n = len(values)

    for i in range(window_bars - 1, n):
        window_data = values[i - window_bars + 1 : i + 1]

        if np.any(np.isnan(window_data)):
            continue

        # Linear regression: y = a + b*t
        t = np.arange(window_bars)
        X = np.column_stack([np.ones(window_bars), t])

        try:
            coeffs = np.linalg.lstsq(X, window_data, rcond=None)[0]
            slope = coeffs[1]

            # Compute R² and t-stat
            predicted = X @ coeffs
            ss_res = np.sum((window_data - predicted) ** 2)
            ss_tot = np.sum((window_data - np.mean(window_data)) ** 2)

            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                r_squared = max(0.0, min(1.0, r_squared))
            else:
                r_squared = 0.0

            if method == "r2":
                # R² with sign of slope
                result.iloc[i] = r_squared * np.sign(slope) if slope != 0 else 0.0
            else:  # tstat
                # T-statistic of slope
                mse = ss_res / (window_bars - 2) if window_bars > 2 else ss_res
                var_t = np.sum((t - np.mean(t)) ** 2)
                se_slope = np.sqrt(mse / var_t) if var_t > 0 else 0.0
                t_stat = slope / se_slope if se_slope > 0 else 0.0
                result.iloc[i] = t_stat

        except (np.linalg.LinAlgError, ValueError):
            continue

    result.name = f"trend_strength_{method}"
    return result


def signal_to_position(
    score: pd.Series,
    threshold: float = 0.0,
    clip: float | None = None,
    discrete: bool = True,
) -> pd.Series:
    """Convert raw signal score to trading position.

    Args:
        score: Raw signal score series.
        threshold: Minimum absolute score to take a position.
            Score > threshold -> long, score < -threshold -> short.
        clip: If set, clip continuous positions to [-clip, +clip].
        discrete: If True, return discrete positions {-1, 0, 1}.
            If False, return continuous positions (scaled score).

    Returns:
        Series of positions. If discrete: -1 (short), 0 (flat), 1 (long).
        If continuous: scaled score values.

    Example:
        >>> pos = signal_to_position(momentum_score, threshold=0.5)
        >>> pos_continuous = signal_to_position(score, threshold=0.0, discrete=False, clip=1.0)
    """
    if discrete:
        position = pd.Series(0, index=score.index, dtype=int)
        position[score > threshold] = 1
        position[score < -threshold] = -1
    else:
        # Continuous position
        position = score.copy()
        # Apply dead zone around zero
        position[(score > -threshold) & (score < threshold)] = 0.0

        if clip is not None:
            position = position.clip(-clip, clip)

    position.name = "position"
    return position


def macd_signal(
    series: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence).

    MACD is a trend-following momentum indicator that shows the
    relationship between two EMAs.

    Args:
        series: Input time series.
        fast_span: Span for fast EMA (default: 12).
        slow_span: Span for slow EMA (default: 26).
        signal_span: Span for signal line EMA (default: 9).

    Returns:
        DataFrame with columns:
        - macd: MACD line (fast EMA - slow EMA)
        - signal: Signal line (EMA of MACD)
        - histogram: MACD histogram (MACD - signal)

    Example:
        >>> macd_df = macd_signal(fly_yield)
        >>> buy_signal = macd_df["histogram"] > 0
    """
    ema_fast = series.ewm(span=fast_span, min_periods=fast_span).mean()
    ema_slow = series.ewm(span=slow_span, min_periods=slow_span).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, min_periods=signal_span).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=series.index)


def rsi_signal(
    series: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Args:
        series: Input time series.
        window: Lookback window for RSI calculation.

    Returns:
        Series of RSI values (0 to 100).
        RSI > 70 typically indicates overbought.
        RSI < 30 typically indicates oversold.

    Example:
        >>> rsi = rsi_signal(fly_yield, window=14)
        >>> oversold = rsi < 30
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=window, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi.name = "rsi"
    return rsi


def adx_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Compute Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Higher ADX indicates stronger trend.

    Note: This requires high/low/close data. For single series like
    fly_yield, consider using trend_strength() instead.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Lookback window.

    Returns:
        DataFrame with columns:
        - adx: Average Directional Index (0 to 100)
        - plus_di: Positive Directional Indicator
        - minus_di: Negative Directional Indicator

    Example:
        >>> adx_df = adx_signal(high, low, close)
        >>> strong_trend = adx_df["adx"] > 25
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed averages
    atr = tr.ewm(span=window, min_periods=window).mean()
    plus_dm_smooth = plus_dm.ewm(span=window, min_periods=window).mean()
    minus_dm_smooth = minus_dm.ewm(span=window, min_periods=window).mean()

    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=window, min_periods=window).mean()

    return pd.DataFrame({
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }, index=close.index)


def combine_momentum_signals(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    method: str = "weighted_avg",
) -> pd.Series:
    """Combine multiple momentum signals into a single score.

    Args:
        signals: Dict mapping signal names to signal series.
        weights: Optional dict mapping signal names to weights.
            If None, equal weights are used.
        method: Combination method:
            - "weighted_avg": Weighted average of signals
            - "vote": Majority vote (sum of signs)

    Returns:
        Combined signal score series.

    Example:
        >>> signals = {
        ...     "ts_mom": ts_momentum_signal(fly_yield, 20),
        ...     "ema_xover": ema_crossover_signal(fly_yield, 5, 20),
        ... }
        >>> combined = combine_momentum_signals(signals)
    """
    if not signals:
        raise ValueError("signals dict must not be empty")

    # Align all signals to common index
    df = pd.DataFrame(signals)

    if weights is None:
        weights = dict.fromkeys(signals, 1.0)

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    if method == "weighted_avg":
        # Weighted average
        combined = pd.Series(0.0, index=df.index)
        for name, series in signals.items():
            w = weights.get(name, 0.0)
            combined += w * series.fillna(0)

    elif method == "vote":
        # Sum of signs (majority vote)
        combined = pd.Series(0.0, index=df.index)
        for name, series in signals.items():
            w = weights.get(name, 1.0)
            combined += w * np.sign(series.fillna(0))

    else:
        raise ValueError(f"Unknown method: {method}")

    combined.name = "combined_momentum"
    return combined


def build_momentum_dataframe(
    fly_yield: pd.Series,
    lookback_bars: int = 20,
    fast_span: int = 5,
    slow_span: int = 20,
    breakout_window: int = 20,
    trend_window: int = 60,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Build complete momentum signal DataFrame from fly yield.

    Computes multiple momentum indicators and combines them into
    a single DataFrame for analysis or trading.

    Args:
        fly_yield: Fly yield series with datetime index.
        lookback_bars: Lookback for time-series momentum.
        fast_span: Fast EMA span for crossover.
        slow_span: Slow EMA span for crossover.
        breakout_window: Window for breakout detection.
        trend_window: Window for trend strength.
        threshold: Threshold for position conversion.

    Returns:
        DataFrame with columns:
        - datetime: Index as column
        - fly_yield: Input fly yield
        - ts_momentum: Time-series momentum score
        - ema_crossover: EMA crossover score
        - breakout: Breakout signal
        - trend_strength: Trend strength (R²)
        - combined_score: Combined momentum score
        - position: Trading position (-1/0/1)

    Example:
        >>> mom_df = build_momentum_dataframe(fly_yield, lookback_bars=20)
    """
    # Compute individual signals
    ts_mom = ts_momentum_signal(fly_yield, lookback_bars=lookback_bars)
    ema_xover = ema_crossover_signal(fly_yield, fast_span=fast_span, slow_span=slow_span)
    breakout = breakout_signal(fly_yield, window_bars=breakout_window)
    trend = trend_strength(fly_yield, window_bars=trend_window, method="r2")

    # Normalize ts_momentum for combination
    ts_mom_std = ts_mom.rolling(window=slow_span, min_periods=slow_span).std()
    ts_mom_norm = ts_mom / ts_mom_std.replace(0, np.nan)

    # Combine signals
    signals = {
        "ts_momentum": ts_mom_norm,
        "ema_crossover": ema_xover,
        "breakout": breakout.astype(float),
    }
    combined = combine_momentum_signals(signals, method="weighted_avg")

    # Convert to position
    position = signal_to_position(combined, threshold=threshold)

    # Build DataFrame
    result = pd.DataFrame({
        "fly_yield": fly_yield.values,
        "ts_momentum": ts_mom.values,
        "ema_crossover": ema_xover.values,
        "breakout": breakout.values,
        "trend_strength": trend.values,
        "combined_score": combined.values,
        "position": position.values,
    }, index=fly_yield.index)

    # Reset index to get datetime as column
    result = result.reset_index()
    if result.columns[0] != "datetime":
        result = result.rename(columns={result.columns[0]: "datetime"})

    return result
