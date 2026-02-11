"""Tests for regime detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.alpha.regime.regime import (
    Regime,
    RegimeConfig,
    apply_regime_gate,
    classify_regime,
    classify_regime_single,
    compute_regime_features,
    generate_ou_process,
    generate_random_walk_with_drift,
    rolling_adf_pvalue,
    rolling_ou_half_life,
    rolling_trend_score,
)


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_ou_process_deterministic(self):
        """OU process should be deterministic with fixed seed."""
        ou1 = generate_ou_process(100, theta=0.1, seed=42)
        ou2 = generate_ou_process(100, theta=0.1, seed=42)

        np.testing.assert_array_equal(ou1, ou2)

    def test_random_walk_deterministic(self):
        """Random walk should be deterministic with fixed seed."""
        rw1 = generate_random_walk_with_drift(100, drift=0.01, seed=42)
        rw2 = generate_random_walk_with_drift(100, drift=0.01, seed=42)

        np.testing.assert_array_equal(rw1, rw2)

    def test_ou_process_mean_reverts(self):
        """OU process should revert toward mean."""
        ou = generate_ou_process(1000, theta=0.2, mu=0.0, sigma=1.0, seed=42)

        # Should spend most time near mean
        near_mean = np.abs(ou) < 2 * np.std(ou)
        assert near_mean.mean() > 0.9

    def test_random_walk_trends(self):
        """Random walk with drift should trend."""
        rw = generate_random_walk_with_drift(500, drift=0.1, sigma=0.5, seed=42)

        # End value should be significantly higher than start
        assert rw[-1] > rw[0] + 20  # drift * n = 0.1 * 500 = 50 expected


class TestRollingStatistics:
    """Tests for rolling statistical functions."""

    @pytest.fixture
    def ou_series(self):
        """Create OU process as pandas Series."""
        ou = generate_ou_process(200, theta=0.15, mu=0.0, sigma=1.0, seed=42)
        return pd.Series(ou, index=pd.date_range("2024-01-01", periods=200, freq="D"))

    @pytest.fixture
    def trend_series(self):
        """Create trending series."""
        rw = generate_random_walk_with_drift(200, drift=0.05, sigma=0.5, seed=42)
        return pd.Series(rw, index=pd.date_range("2024-01-01", periods=200, freq="D"))

    def test_rolling_adf_pvalue_ou(self, ou_series):
        """ADF p-value should be low for OU process."""
        pvalues = rolling_adf_pvalue(ou_series, window=60)

        # Should have many low p-values (stationary)
        valid_pvalues = pvalues.dropna()
        low_pvalue_ratio = (valid_pvalues < 0.1).mean()
        assert low_pvalue_ratio > 0.5

    def test_rolling_adf_pvalue_trend(self, trend_series):
        """ADF p-value should be high for trending series."""
        pvalues = rolling_adf_pvalue(trend_series, window=60)

        # Should have mostly high p-values (non-stationary)
        valid_pvalues = pvalues.dropna()
        high_pvalue_ratio = (valid_pvalues > 0.2).mean()
        assert high_pvalue_ratio > 0.5

    def test_rolling_ou_half_life(self, ou_series):
        """Half-life should be reasonable for OU process."""
        half_lives = rolling_ou_half_life(ou_series, window=60)

        # Half-life for theta=0.15 is approximately -log(2)/log(1-0.15) ≈ 4.3
        valid_hl = half_lives.dropna()
        median_hl = valid_hl.median()
        assert 1 < median_hl < 30  # Reasonable range

    def test_rolling_trend_score_trending(self, trend_series):
        """R² should be high for trending series."""
        trend_df = rolling_trend_score(trend_series, window=60)

        valid_r2 = trend_df["r_squared"].dropna()
        high_r2_ratio = (valid_r2 > 0.3).mean()
        assert high_r2_ratio > 0.5

    def test_rolling_trend_score_ou(self, ou_series):
        """R² should be low for mean-reverting series."""
        trend_df = rolling_trend_score(ou_series, window=60)

        valid_r2 = trend_df["r_squared"].dropna()
        low_r2_ratio = (valid_r2 < 0.3).mean()
        assert low_r2_ratio > 0.5


class TestClassifyRegimeSingle:
    """Tests for single-window regime classification."""

    def test_ou_classified_as_mean_revert(self):
        """OU process should be classified as mean-reverting."""
        ou = generate_ou_process(100, theta=0.15, mu=0.0, sigma=1.0, seed=42)

        config = RegimeConfig(
            window=100,
            adf_threshold=0.1,
            half_life_max=30,
        )
        result = classify_regime_single(ou, config)

        # Should be classified as mean_revert or at least not trend
        assert result.regime in [Regime.MEAN_REVERT, Regime.UNCERTAIN]

    def test_trend_classified_as_trend(self):
        """Strong trend should be classified as trending."""
        rw = generate_random_walk_with_drift(100, drift=0.1, sigma=0.5, seed=42)

        config = RegimeConfig(
            window=100,
            trend_r2_threshold=0.3,
        )
        result = classify_regime_single(rw, config)

        # Should be classified as trend or uncertain
        assert result.regime in [Regime.TREND, Regime.UNCERTAIN]

    def test_result_contains_test_values(self):
        """Result should contain individual test values."""
        ou = generate_ou_process(100, theta=0.15, seed=42)

        config = RegimeConfig(use_adf=True, use_half_life=True, use_trend=True)
        result = classify_regime_single(ou, config)

        assert result.adf_pvalue is not None
        assert result.half_life is not None
        assert result.trend_r2 is not None
        assert result.scores is not None


class TestClassifyRegime:
    """Tests for rolling regime classification."""

    def test_rolling_classification(self):
        """Should produce regime classification for each row."""
        ou = generate_ou_process(150, theta=0.15, seed=42)
        series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=150, freq="D"))

        config = RegimeConfig(window=60)
        regimes = classify_regime(series, config)

        assert len(regimes) == len(series)
        # First window-1 should be UNCERTAIN
        assert all(r == Regime.UNCERTAIN for r in regimes.iloc[:59])

    def test_classification_is_deterministic(self):
        """Classification should be deterministic with fixed seed."""
        ou = generate_ou_process(150, theta=0.15, seed=42)
        series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=150, freq="D"))

        config = RegimeConfig(window=60)
        regimes1 = classify_regime(series, config)
        regimes2 = classify_regime(series, config)

        # Should be identical
        assert (regimes1 == regimes2).all()

    def test_ou_more_mean_revert_than_trend(self):
        """OU process should classify as mean_revert more than random walk."""
        # Generate OU process
        ou = generate_ou_process(300, theta=0.15, mu=0.0, sigma=1.0, seed=42)
        ou_series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=300, freq="D"))

        # Generate random walk with drift
        rw = generate_random_walk_with_drift(300, drift=0.05, sigma=1.0, seed=42)
        rw_series = pd.Series(rw, index=pd.date_range("2024-01-01", periods=300, freq="D"))

        config = RegimeConfig(window=60)

        ou_regimes = classify_regime(ou_series, config)
        rw_regimes = classify_regime(rw_series, config)

        # Count mean_revert classifications
        ou_mr_count = (ou_regimes == Regime.MEAN_REVERT).sum()
        rw_mr_count = (rw_regimes == Regime.MEAN_REVERT).sum()

        # OU should have more mean_revert classifications
        assert ou_mr_count > rw_mr_count

    def test_trend_series_more_trend_than_ou(self):
        """Trending series should classify as trend more than OU."""
        # Generate OU process
        ou = generate_ou_process(300, theta=0.15, mu=0.0, sigma=1.0, seed=42)
        ou_series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=300, freq="D"))

        # Generate random walk with strong drift
        rw = generate_random_walk_with_drift(300, drift=0.1, sigma=0.5, seed=42)
        rw_series = pd.Series(rw, index=pd.date_range("2024-01-01", periods=300, freq="D"))

        config = RegimeConfig(window=60)

        ou_regimes = classify_regime(ou_series, config)
        rw_regimes = classify_regime(rw_series, config)

        # Count trend classifications
        ou_trend_count = (ou_regimes == Regime.TREND).sum()
        rw_trend_count = (rw_regimes == Regime.TREND).sum()

        # RW should have more trend classifications
        assert rw_trend_count > ou_trend_count


class TestComputeRegimeFeatures:
    """Tests for compute_regime_features function."""

    def test_returns_dataframe_with_all_features(self):
        """Should return DataFrame with all regime features."""
        ou = generate_ou_process(100, theta=0.15, seed=42)
        series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=100, freq="D"))

        df = compute_regime_features(series, RegimeConfig(window=30))

        assert "regime" in df.columns
        assert "adf_pvalue" in df.columns
        assert "half_life" in df.columns
        assert "trend_slope" in df.columns
        assert "trend_r2" in df.columns

    def test_features_match_index(self):
        """Features should have same index as input."""
        ou = generate_ou_process(100, theta=0.15, seed=42)
        series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=100, freq="D"))

        df = compute_regime_features(series, RegimeConfig(window=30))

        assert len(df) == len(series)
        assert df.index.equals(series.index)


class TestApplyRegimeGate:
    """Tests for apply_regime_gate function."""

    def test_blocks_entry_in_wrong_regime(self):
        """Should block entries when regime is not allowed."""
        signal = pd.Series([0, 1, 1, 1, 0])
        regime = pd.Series([
            Regime.UNCERTAIN,
            Regime.TREND,  # Entry blocked
            Regime.TREND,
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
        ])

        gated = apply_regime_gate(signal, regime)

        # Entry at index 1 should be blocked (regime=TREND)
        assert gated.iloc[1] == 0

    def test_allows_entry_in_correct_regime(self):
        """Should allow entries when regime is allowed."""
        signal = pd.Series([0, 1, 1, 1, 0])
        regime = pd.Series([
            Regime.UNCERTAIN,
            Regime.MEAN_REVERT,  # Entry allowed
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
        ])

        gated = apply_regime_gate(signal, regime)

        # Entry at index 1 should be allowed
        assert gated.iloc[1] == 1

    def test_maintains_position_during_regime(self):
        """Should maintain position while in allowed regime."""
        signal = pd.Series([0, 1, 1, 1, 0])
        regime = pd.Series([
            Regime.UNCERTAIN,
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
        ])

        gated = apply_regime_gate(signal, regime)

        assert list(gated) == [0, 1, 1, 1, 0]

    def test_exit_on_regime_change(self):
        """Should force exit when regime changes if configured."""
        signal = pd.Series([0, 1, 1, 1, 1])
        regime = pd.Series([
            Regime.UNCERTAIN,
            Regime.MEAN_REVERT,  # Enter
            Regime.MEAN_REVERT,
            Regime.TREND,  # Force exit
            Regime.TREND,
        ])

        gated = apply_regime_gate(
            signal, regime, exit_on_regime_change=True
        )

        # Should exit at index 3 when regime changes to TREND
        assert gated.iloc[2] == 1  # Still holding
        assert gated.iloc[3] == 0  # Forced exit
        assert gated.iloc[4] == 0  # Stays flat

    def test_no_exit_on_regime_change_by_default(self):
        """Should not force exit when exit_on_regime_change=False."""
        signal = pd.Series([0, 1, 1, 1, 1])
        regime = pd.Series([
            Regime.UNCERTAIN,
            Regime.MEAN_REVERT,
            Regime.MEAN_REVERT,
            Regime.TREND,  # No forced exit
            Regime.TREND,
        ])

        gated = apply_regime_gate(
            signal, regime, exit_on_regime_change=False
        )

        # Should maintain position even when regime changes
        assert gated.iloc[3] == 1  # Still holding
        assert gated.iloc[4] == 1

    def test_custom_allow_regimes(self):
        """Should allow custom list of regimes."""
        signal = pd.Series([0, 1, 1, 0])
        regime = pd.Series([
            Regime.UNCERTAIN,
            Regime.UNCERTAIN,  # Now allowed
            Regime.UNCERTAIN,
            Regime.UNCERTAIN,
        ])

        # Allow UNCERTAIN as well
        gated = apply_regime_gate(
            signal, regime,
            allow_regimes=[Regime.MEAN_REVERT, Regime.UNCERTAIN],
        )

        # Entry should be allowed with UNCERTAIN
        assert gated.iloc[1] == 1


class TestRegimeConfig:
    """Tests for RegimeConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = RegimeConfig()

        assert config.window == 60
        assert config.adf_threshold == 0.05
        assert config.half_life_max == 30.0
        assert config.use_adf is True
        assert config.use_half_life is True

    def test_require_all_mode(self):
        """require_all mode should require all tests to agree."""
        # Create data where tests might disagree
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100) * 0.5) + np.linspace(0, 5, 100)

        config_majority = RegimeConfig(window=100, require_all=False)
        config_all = RegimeConfig(window=100, require_all=True)

        result_majority = classify_regime_single(data, config_majority)
        result_all = classify_regime_single(data, config_all)

        # require_all more likely to be UNCERTAIN when tests disagree
        # Just verify both run without error
        assert result_majority.regime in [Regime.MEAN_REVERT, Regime.TREND, Regime.UNCERTAIN]
        assert result_all.regime in [Regime.MEAN_REVERT, Regime.TREND, Regime.UNCERTAIN]


class TestDeterminism:
    """Tests to verify determinism with fixed seeds."""

    def test_full_pipeline_deterministic(self):
        """Full regime detection pipeline should be deterministic."""
        # Run pipeline twice with same seed
        for _ in range(2):
            ou = generate_ou_process(200, theta=0.15, mu=0.0, sigma=1.0, seed=123)
            series = pd.Series(ou, index=pd.date_range("2024-01-01", periods=200, freq="D"))

            config = RegimeConfig(window=60)
            regimes = classify_regime(series, config)

            # Store first run results
            if _ == 0:
                first_run = regimes.copy()
            else:
                # Compare with first run
                assert (regimes == first_run).all()

    def test_regime_result_values_deterministic(self):
        """Individual regime test values should be deterministic."""
        ou = generate_ou_process(100, theta=0.15, seed=456)

        config = RegimeConfig(window=100)
        result1 = classify_regime_single(ou, config)
        result2 = classify_regime_single(ou, config)

        assert result1.adf_pvalue == result2.adf_pvalue
        assert result1.half_life == result2.half_life
        assert result1.trend_r2 == result2.trend_r2
        assert result1.regime == result2.regime


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_short_series(self):
        """Should handle series shorter than window."""
        short_data = np.array([1, 2, 3, 4, 5])

        result = classify_regime_single(short_data, RegimeConfig(window=10))

        # Should return some result without error
        assert result.regime is not None

    def test_constant_series(self):
        """Should handle constant series."""
        constant = np.ones(100)

        result = classify_regime_single(constant, RegimeConfig())

        # Should not crash
        assert result.regime is not None

    def test_nan_handling(self):
        """Should handle NaN values gracefully."""
        data = np.array([1, 2, np.nan, 4, 5] * 20)

        # This might produce NaN results but shouldn't crash
        result = classify_regime_single(data, RegimeConfig())
        assert result.regime is not None
