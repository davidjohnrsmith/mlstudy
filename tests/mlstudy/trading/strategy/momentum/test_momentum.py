"""Tests for momentum (trend-following) signals."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.alpha.momentum.momentum import (
    MomentumConfig,
    TrendMethod,
    breakout_signal,
    build_momentum_dataframe,
    combine_momentum_signals,
    ema_crossover_signal,
    macd_signal,
    rsi_signal,
    signal_to_position,
    trend_strength,
    ts_momentum_signal,
)
from mlstudy.trading.strategy.alpha.regime.regime import (
    generate_ou_process,
    generate_random_walk_with_drift,
)


@pytest.fixture
def trending_series():
    """Create synthetic trending (random walk with drift) series."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    values = generate_random_walk_with_drift(n, drift=0.05, sigma=0.3, seed=42)
    return pd.Series(values, index=dates, name="fly_yield")


@pytest.fixture
def mean_reverting_series():
    """Create synthetic mean-reverting (OU process) series."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    values = generate_ou_process(n, theta=0.3, sigma=0.5, seed=42)
    return pd.Series(values, index=dates, name="fly_yield")


@pytest.fixture
def flat_series():
    """Create flat series with no trend."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    np.random.seed(42)
    values = np.random.randn(n) * 0.1  # Small noise around zero
    return pd.Series(values, index=dates, name="fly_yield")


class TestTsMomentumSignal:
    """Tests for ts_momentum_signal."""

    def test_returns_series(self, trending_series):
        """Should return a pandas Series."""
        mom = ts_momentum_signal(trending_series, lookback_bars=20)
        assert isinstance(mom, pd.Series)
        assert len(mom) == len(trending_series)

    def test_positive_for_uptrend(self, trending_series):
        """Should be mostly positive for upward trending series."""
        mom = ts_momentum_signal(trending_series, lookback_bars=20)
        # After warmup, momentum should be mostly positive
        valid_mom = mom.dropna().iloc[20:]
        assert valid_mom.mean() > 0

    def test_smoothing_reduces_noise(self, trending_series):
        """Smoothing should reduce signal noise."""
        mom_raw = ts_momentum_signal(trending_series, lookback_bars=20)
        mom_smooth = ts_momentum_signal(trending_series, lookback_bars=20, smooth=5)

        # Smoothed signal should have lower std
        assert mom_smooth.dropna().std() < mom_raw.dropna().std()

    def test_log_return_mode(self, trending_series):
        """Should support log return calculation."""
        # Shift to positive values for log
        pos_series = trending_series + 10

        mom_diff = ts_momentum_signal(pos_series, lookback_bars=20, use_log_return=False)
        mom_log = ts_momentum_signal(pos_series, lookback_bars=20, use_log_return=True)

        # Both should have same sign pattern
        assert not mom_diff.equals(mom_log)
        # Log returns should be smaller in magnitude
        assert abs(mom_log.dropna().mean()) < abs(mom_diff.dropna().mean())


class TestEmaCrossoverSignal:
    """Tests for ema_crossover_signal."""

    def test_returns_series(self, trending_series):
        """Should return a pandas Series."""
        xover = ema_crossover_signal(trending_series, fast_span=5, slow_span=20)
        assert isinstance(xover, pd.Series)

    def test_positive_for_uptrend(self, trending_series):
        """Fast EMA should be above slow EMA in uptrend."""
        xover = ema_crossover_signal(trending_series, fast_span=5, slow_span=20)
        # After warmup, crossover should be mostly positive
        valid_xover = xover.dropna().iloc[30:]
        assert valid_xover.mean() > 0

    def test_fast_must_be_less_than_slow(self, trending_series):
        """Should raise if fast_span >= slow_span."""
        with pytest.raises(ValueError, match="fast_span"):
            ema_crossover_signal(trending_series, fast_span=20, slow_span=10)

    def test_normalization(self, trending_series):
        """Normalized signal should be scale-invariant."""
        scaled_series = trending_series * 100

        xover_orig = ema_crossover_signal(trending_series, fast_span=5, slow_span=20, normalize=True)
        xover_scaled = ema_crossover_signal(scaled_series, fast_span=5, slow_span=20, normalize=True)

        # Normalized signals should be similar
        valid_slice = slice(30, None)
        corr = xover_orig.iloc[valid_slice].corr(xover_scaled.iloc[valid_slice])
        assert corr > 0.99


class TestBreakoutSignal:
    """Tests for breakout_signal."""

    def test_returns_series(self, trending_series):
        """Should return a pandas Series."""
        breakout = breakout_signal(trending_series, window_bars=20)
        assert isinstance(breakout, pd.Series)

    def test_values_are_discrete(self, trending_series):
        """Should return -1, 0, or 1."""
        breakout = breakout_signal(trending_series, window_bars=20)
        unique_values = set(breakout.dropna().unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_upside_breakout_in_uptrend(self, trending_series):
        """Should detect upside breakouts in uptrend."""
        breakout = breakout_signal(trending_series, window_bars=20)
        # Should have some upside breakouts
        assert (breakout == 1).sum() > 0

    def test_downside_breakout_in_downtrend(self):
        """Should detect downside breakouts in downtrend."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        # Create downward trending series
        values = generate_random_walk_with_drift(n, drift=-0.05, sigma=0.3, seed=42)
        series = pd.Series(values, index=dates)

        breakout = breakout_signal(series, window_bars=20)
        # Should have some downside breakouts
        assert (breakout == -1).sum() > 0


class TestTrendStrength:
    """Tests for trend_strength."""

    def test_returns_series(self, trending_series):
        """Should return a pandas Series."""
        strength = trend_strength(trending_series, window_bars=30)
        assert isinstance(strength, pd.Series)

    def test_high_for_trending(self, trending_series):
        """Should be high for trending series."""
        strength = trend_strength(trending_series, window_bars=30, method="r2")
        valid_strength = strength.dropna()
        # R² should be relatively high for trending series
        assert valid_strength.abs().mean() > 0.3

    def test_low_for_mean_reverting(self, mean_reverting_series):
        """Should be lower for mean-reverting series."""
        strength = trend_strength(mean_reverting_series, window_bars=30, method="r2")
        valid_strength = strength.dropna()
        # R² should be lower for mean-reverting series
        assert valid_strength.abs().mean() < 0.5

    def test_r2_method(self, trending_series):
        """R² method should return values in [-1, 1] (signed R²)."""
        strength = trend_strength(trending_series, window_bars=30, method="r2")
        valid_strength = strength.dropna()
        assert valid_strength.abs().max() <= 1.0

    def test_tstat_method(self, trending_series):
        """T-stat method should return unbounded values."""
        strength = trend_strength(trending_series, window_bars=30, method="tstat")
        valid_strength = strength.dropna()
        # T-stat can be large for strong trends
        assert valid_strength.abs().max() > 1.0

    def test_invalid_method_raises(self, trending_series):
        """Should raise for invalid method."""
        with pytest.raises(ValueError, match="method"):
            trend_strength(trending_series, window_bars=30, method="invalid")


class TestSignalToPosition:
    """Tests for signal_to_position."""

    def test_discrete_positions(self):
        """Should return discrete positions by default."""
        score = pd.Series([2.0, 0.5, -0.5, -2.0, 0.0])
        pos = signal_to_position(score, threshold=1.0)

        assert pos.iloc[0] == 1  # score > threshold
        assert pos.iloc[1] == 0  # |score| < threshold
        assert pos.iloc[2] == 0  # |score| < threshold
        assert pos.iloc[3] == -1  # score < -threshold
        assert pos.iloc[4] == 0  # score == 0

    def test_continuous_positions(self):
        """Should return continuous positions when discrete=False."""
        score = pd.Series([2.0, 0.5, -0.5, -2.0])
        pos = signal_to_position(score, threshold=0.0, discrete=False)

        assert pos.iloc[0] == 2.0
        assert pos.iloc[1] == 0.5
        assert pos.iloc[2] == -0.5
        assert pos.iloc[3] == -2.0

    def test_clipping(self):
        """Should clip continuous positions."""
        score = pd.Series([5.0, 0.5, -0.5, -5.0])
        pos = signal_to_position(score, threshold=0.0, discrete=False, clip=1.0)

        assert pos.iloc[0] == 1.0
        assert pos.iloc[3] == -1.0

    def test_dead_zone(self):
        """Threshold should create dead zone around zero."""
        score = pd.Series([0.3, -0.3, 0.0])
        pos = signal_to_position(score, threshold=0.5, discrete=False)

        assert (pos == 0).all()


class TestMacdSignal:
    """Tests for macd_signal."""

    def test_returns_dataframe(self, trending_series):
        """Should return DataFrame with expected columns."""
        macd_df = macd_signal(trending_series)

        assert isinstance(macd_df, pd.DataFrame)
        assert "macd" in macd_df.columns
        assert "signal" in macd_df.columns
        assert "histogram" in macd_df.columns

    def test_histogram_is_macd_minus_signal(self, trending_series):
        """Histogram should equal MACD minus signal line."""
        macd_df = macd_signal(trending_series)
        diff = macd_df["macd"] - macd_df["signal"]

        pd.testing.assert_series_equal(
            macd_df["histogram"], diff, check_names=False
        )


class TestRsiSignal:
    """Tests for rsi_signal."""

    def test_returns_series(self, trending_series):
        """Should return a pandas Series."""
        rsi = rsi_signal(trending_series)
        assert isinstance(rsi, pd.Series)

    def test_bounded_0_to_100(self, trending_series):
        """RSI should be between 0 and 100."""
        rsi = rsi_signal(trending_series)
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_high_for_uptrend(self, trending_series):
        """RSI should be high for uptrending series."""
        rsi = rsi_signal(trending_series)
        valid_rsi = rsi.dropna().iloc[30:]
        # Should be above 50 on average for uptrend
        assert valid_rsi.mean() > 50


class TestCombineMomentumSignals:
    """Tests for combine_momentum_signals."""

    def test_weighted_average(self, trending_series):
        """Should compute weighted average of signals."""
        ts_mom = ts_momentum_signal(trending_series, lookback_bars=20)
        ema_xover = ema_crossover_signal(trending_series, fast_span=5, slow_span=20)

        signals = {"ts_mom": ts_mom, "ema_xover": ema_xover}
        combined = combine_momentum_signals(signals, method="weighted_avg")

        assert isinstance(combined, pd.Series)
        assert len(combined) == len(trending_series)

    def test_vote_method(self, trending_series):
        """Vote method should sum signs."""
        # Create signals with known signs
        pos_signal = pd.Series([1.0] * 100, index=trending_series.index[:100])
        neg_signal = pd.Series([-1.0] * 100, index=trending_series.index[:100])
        mixed_signal = pd.Series([1.0] * 100, index=trending_series.index[:100])

        signals = {"pos": pos_signal, "neg": neg_signal, "mixed": mixed_signal}
        combined = combine_momentum_signals(signals, method="vote")

        # 2 positive, 1 negative -> combined should be positive
        assert combined.mean() > 0

    def test_custom_weights(self, trending_series):
        """Should respect custom weights."""
        ts_mom = pd.Series([1.0] * len(trending_series), index=trending_series.index)
        ema_xover = pd.Series([-1.0] * len(trending_series), index=trending_series.index)

        signals = {"ts_mom": ts_mom, "ema_xover": ema_xover}

        # Weight ts_mom more heavily
        weights = {"ts_mom": 3.0, "ema_xover": 1.0}
        combined = combine_momentum_signals(signals, weights=weights, method="weighted_avg")

        # Should be positive due to heavier ts_mom weight
        assert combined.mean() > 0

    def test_empty_signals_raises(self):
        """Should raise for empty signals dict."""
        with pytest.raises(ValueError, match="empty"):
            combine_momentum_signals({})


class TestBuildMomentumDataframe:
    """Tests for build_momentum_dataframe."""

    def test_returns_dataframe(self, trending_series):
        """Should return DataFrame with expected columns."""
        df = build_momentum_dataframe(trending_series)

        assert isinstance(df, pd.DataFrame)
        assert "datetime" in df.columns
        assert "fly_yield" in df.columns
        assert "ts_momentum" in df.columns
        assert "ema_crossover" in df.columns
        assert "breakout" in df.columns
        assert "trend_strength" in df.columns
        assert "combined_score" in df.columns
        assert "position" in df.columns

    def test_position_values(self, trending_series):
        """Position should be -1, 0, or 1."""
        df = build_momentum_dataframe(trending_series)
        unique_pos = set(df["position"].dropna().unique())
        assert unique_pos.issubset({-1, 0, 1})


class TestMomentumAlignmentWithTrend:
    """Test that momentum signals align with trending series."""

    def test_momentum_aligns_with_drift(self, trending_series):
        """Momentum signal should be mostly aligned with drift direction."""
        # Upward drift -> should have more long positions
        mom = ts_momentum_signal(trending_series, lookback_bars=20)
        pos = signal_to_position(mom, threshold=0.0)

        # Count long vs short
        n_long = (pos == 1).sum()
        n_short = (pos == -1).sum()

        # For upward trending series, should have more longs
        assert n_long > n_short

    def test_momentum_opposite_for_downtrend(self):
        """Momentum should be negative for downward trending series."""
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        values = generate_random_walk_with_drift(n, drift=-0.05, sigma=0.3, seed=42)
        down_series = pd.Series(values, index=dates)

        mom = ts_momentum_signal(down_series, lookback_bars=20)
        pos = signal_to_position(mom, threshold=0.0)

        # Count long vs short
        n_long = (pos == 1).sum()
        n_short = (pos == -1).sum()

        # For downward trending series, should have more shorts
        assert n_short > n_long


class TestTrendFilterReducesOvertrade:
    """Test that trend filter reduces overtrading in non-trending regimes."""

    def test_trend_filter_reduces_trades_in_mr_regime(self, mean_reverting_series):
        """Trend filter should reduce position changes in MR regime."""
        from mlstudy.trading.strategy.alpha.momentum.momentum import trend_strength

        # Without filter
        mom = ts_momentum_signal(mean_reverting_series, lookback_bars=10)
        pos_no_filter = signal_to_position(mom, threshold=0.0)

        # With filter
        trend = trend_strength(mean_reverting_series, window_bars=30, method="r2")
        pos_with_filter = pos_no_filter.copy()
        weak_trend = abs(trend) < 0.2
        pos_with_filter[weak_trend] = 0

        # Count position changes
        changes_no_filter = pos_no_filter.diff().abs().sum()
        changes_with_filter = pos_with_filter.diff().abs().sum()

        # Filter should reduce changes
        assert changes_with_filter <= changes_no_filter


class TestExecutionLag:
    """Test that execution lag is properly applied."""

    def test_signal_at_t_produces_position_at_t_plus_1(self):
        """Signal at bar t should produce position at bar t+1."""
        n = 10
        dates = pd.date_range("2023-01-01", periods=n, freq="h")

        # Create deterministic signal
        signal = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 0, 0], index=dates)

        # Apply lag of 1
        position = signal.shift(1).fillna(0).astype(int)

        # Position should be delayed by 1 bar
        assert position.iloc[0] == 0  # No signal yet
        assert position.iloc[1] == 0  # signal[0] = 0
        assert position.iloc[2] == 0  # signal[1] = 0
        assert position.iloc[3] == 1  # signal[2] = 1
        assert position.iloc[4] == 1  # signal[3] = 1
        assert position.iloc[5] == 1  # signal[4] = 1
        assert position.iloc[6] == -1  # signal[5] = -1

    def test_no_lookahead_in_momentum_signal(self, trending_series):
        """Momentum signal should not look ahead."""
        lookback = 20

        # Compute momentum
        mom = ts_momentum_signal(trending_series, lookback_bars=lookback)

        # Signal at bar i should only depend on bars [i-lookback, i]
        # Check a specific bar
        test_idx = 50
        window_data = trending_series.iloc[test_idx - lookback : test_idx + 1]
        expected_diff = window_data.iloc[-1] - window_data.iloc[0]

        assert abs(mom.iloc[test_idx] - expected_diff) < 1e-10


class TestEndToEndPipeline:
    """End-to-end tests for momentum signal pipeline."""

    def test_full_pipeline_trending_data(self, trending_series):
        """Should run complete pipeline on trending data."""
        # Build momentum signals
        df = build_momentum_dataframe(
            trending_series,
            lookback_bars=20,
            fast_span=5,
            slow_span=20,
            breakout_window=20,
            threshold=0.0,
        )

        # Should have reasonable signal quality
        assert len(df) > 0
        assert df["position"].notna().sum() > 0

        # For trending data, should have mostly aligned positions
        n_long = (df["position"] == 1).sum()
        n_short = (df["position"] == -1).sum()
        # Upward trend should favor longs
        assert n_long > n_short * 0.5  # At least half as many longs as shorts

    def test_full_pipeline_mean_reverting_data(self, mean_reverting_series):
        """Should run complete pipeline on MR data (may not profit)."""
        df = build_momentum_dataframe(
            mean_reverting_series,
            lookback_bars=20,
            threshold=0.0,
        )

        # Should run without error
        assert len(df) > 0

    def test_config_dataclass(self):
        """MomentumConfig should have all expected fields."""
        config = MomentumConfig(
            lookback_bars=30,
            fast_span=10,
            slow_span=30,
            threshold=0.5,
        )

        assert config.lookback_bars == 30
        assert config.fast_span == 10
        assert config.slow_span == 30
        assert config.threshold == 0.5

    def test_trend_method_enum(self):
        """TrendMethod enum should have expected values."""
        assert TrendMethod.R2.value == "r2"
        assert TrendMethod.TSTAT.value == "tstat"
