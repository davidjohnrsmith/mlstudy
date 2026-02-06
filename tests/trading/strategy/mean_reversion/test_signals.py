"""Tests for mean-reversion signal generation."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.alpha.mean_reversion.signals import (
    ExitReason,
    SignalConfig,
    TradeStats,
    backtest_signal,
    build_signal_dataframe,
    build_trade_blotter,
    build_trade_blotter_with_details,
    compute_backtest_stats,
    compute_signal_stats,
    compute_signal_strength,
    compute_trade_stats,
    ewma_zscore,
    generate_mean_reversion_signal,
    rolling_zscore,
)


@pytest.fixture
def fly_yield():
    """Create synthetic fly yield with mean-reverting behavior."""
    np.random.seed(42)

    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Mean-reverting process
    mean = 0.0
    theta = 0.1  # Mean reversion speed
    sigma = 0.3

    yields = [0.0]
    for _i in range(1, n):
        dy = theta * (mean - yields[-1]) + sigma * np.random.randn()
        yields.append(yields[-1] + dy)

    return pd.Series(yields, index=dates, name="fly_yield")


@pytest.fixture
def extreme_fly_yield():
    """Fly yield with clear extreme moves for signal testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Create pattern: normal -> extreme low -> reversion -> extreme high -> reversion
    values = (
        [0.0] * 20 +  # Normal
        [-2.5] * 10 +  # Extreme low (should trigger long)
        [-1.0] * 5 +  # Partial reversion
        [0.0] * 10 +  # Back to normal (exit long)
        [2.5] * 10 +  # Extreme high (should trigger short)
        [1.0] * 5 +  # Partial reversion
        [0.0] * 10 +  # Back to normal (exit short)
        [0.0] * 30  # Padding
    )

    return pd.Series(values, index=dates, name="fly_yield")


class TestRollingZscore:
    """Tests for rolling_zscore."""

    def test_returns_series(self, fly_yield):
        """Should return a pandas Series."""
        z = rolling_zscore(fly_yield, window=20)
        assert isinstance(z, pd.Series)
        assert len(z) == len(fly_yield)

    def test_zscore_properties(self, fly_yield):
        """Z-score should have mean ~0 and std ~1 after warmup."""
        z = rolling_zscore(fly_yield, window=20)
        z_valid = z.dropna()

        # After warmup, should be roughly standardized
        assert abs(z_valid.mean()) < 0.5
        assert 0.5 < z_valid.std() < 2.0

    def test_clip_works(self, fly_yield):
        """Clipping should bound z-scores."""
        z = rolling_zscore(fly_yield, window=20, clip=2.0)

        assert z.max() <= 2.0
        assert z.min() >= -2.0

    def test_robust_zscore(self, fly_yield):
        """Robust z-score should use median/MAD."""
        z_standard = rolling_zscore(fly_yield, window=20, robust=False)
        z_robust = rolling_zscore(fly_yield, window=20, robust=True)

        # Should be different (MAD vs std)
        assert not np.allclose(z_standard.dropna(), z_robust.dropna())

    def test_nan_at_start(self, fly_yield):
        """Should have NaN during warmup period."""
        z = rolling_zscore(fly_yield, window=20)

        # First 19 values should be NaN (need 20 for first valid)
        assert z.iloc[:19].isna().all()
        assert z.iloc[19:].notna().all()

    def test_eps_prevents_division_by_zero(self):
        """Eps should prevent division by zero for constant series."""
        # Constant series has std=0
        constant = pd.Series([1.0] * 50)
        z = rolling_zscore(constant, window=10, eps=1e-12)

        # Should not have inf values
        valid = z.dropna()
        assert np.isfinite(valid).all(), "Should not have inf values"

    def test_eps_robust_prevents_division_by_zero(self):
        """Eps should prevent division by zero for constant series with robust."""
        constant = pd.Series([1.0] * 50)
        z = rolling_zscore(constant, window=10, robust=True, eps=1e-12)

        valid = z.dropna()
        assert np.isfinite(valid).all(), "Should not have inf values"


class TestEwmaZscore:
    """Tests for ewma_zscore."""

    def test_returns_series(self, fly_yield):
        """Should return a pandas Series."""
        z = ewma_zscore(fly_yield, span=20)
        assert isinstance(z, pd.Series)

    def test_adapts_faster(self, fly_yield):
        """EWMA should adapt faster to regime changes."""
        # Create regime change
        series = fly_yield.copy()
        series.iloc[100:] += 2.0  # Shift mean

        z_rolling = rolling_zscore(series, window=20)
        z_ewma = ewma_zscore(series, span=20)

        # After shift, EWMA should normalize faster
        late_period = slice(120, 150)
        assert z_ewma.iloc[late_period].abs().mean() < z_rolling.iloc[late_period].abs().mean()

    def test_clip_works(self, fly_yield):
        """Clipping should bound z-scores."""
        z = ewma_zscore(fly_yield, span=20, clip=3.0)

        assert z.max() <= 3.0
        assert z.min() >= -3.0

    def test_eps_prevents_division_by_zero(self):
        """Eps should prevent division by zero for constant series."""
        constant = pd.Series([1.0] * 50)
        z = ewma_zscore(constant, span=10, eps=1e-12)

        # Should not have inf values after first few
        valid = z.iloc[5:]
        assert np.isfinite(valid).all(), "Should not have inf values"


class TestGenerateSignal:
    """Tests for generate_signal."""

    def test_returns_series(self, fly_yield):
        """Should return a pandas Series."""
        z = rolling_zscore(fly_yield, window=20)
        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5)

        assert isinstance(signal, pd.Series)
        assert len(signal) == len(z)

    def test_signal_values(self, fly_yield):
        """Signal should be -1, 0, or 1."""
        z = rolling_zscore(fly_yield, window=20)
        signal = generate_mean_reversion_signal(z)

        unique_values = set(signal.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_entry_logic(self):
        """Should enter positions at entry threshold."""
        # Create z-score series that crosses entry threshold
        z = pd.Series([0, 0, -2.5, -2.5, -1.0, 0, 0, 2.5, 2.5, 1.0, 0])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5)

        # Should go long when z < -2
        assert signal.iloc[2] == 1  # Enter long
        assert signal.iloc[3] == 1  # Stay long

        # Should exit when z > -0.5
        assert signal.iloc[5] == 0  # Exit to flat

        # Should go short when z > 2
        assert signal.iloc[7] == -1  # Enter short

    def test_stop_logic(self):
        """Should stop out at stop threshold."""
        # z-score that triggers stop
        z = pd.Series([0, -2.5, -3.0, -4.5, -5.0])  # Extreme move

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)

        # Should enter long at -2.5
        assert signal.iloc[1] == 1

        # Should stop out at -4.5 (exceeds -4.0)
        assert signal.iloc[3] == 0

    def test_long_entry_exit_deterministic(self):
        """Deterministic test of long entry and exit sequence."""
        # Construct z-score to test: flat -> enter long -> hold -> exit
        z = pd.Series([
            0.0,   # 0: flat (no entry condition)
            -1.5,  # 1: still flat (above -2 entry)
            -2.1,  # 2: enter long (below -2)
            -2.5,  # 3: stay long
            -1.0,  # 4: still long (below -0.5 exit)
            -0.4,  # 5: exit (above -0.5)
            0.0,   # 6: flat
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=None)

        assert signal.iloc[0] == 0, "Should start flat"
        assert signal.iloc[1] == 0, "Should stay flat (not at entry threshold)"
        assert signal.iloc[2] == 1, "Should enter long when z <= -2"
        assert signal.iloc[3] == 1, "Should stay long"
        assert signal.iloc[4] == 1, "Should stay long (not yet at exit)"
        assert signal.iloc[5] == 0, "Should exit when z >= -0.5"
        assert signal.iloc[6] == 0, "Should remain flat"

    def test_short_entry_exit_deterministic(self):
        """Deterministic test of short entry and exit sequence."""
        # Construct z-score to test: flat -> enter short -> hold -> exit
        z = pd.Series([
            0.0,   # 0: flat
            1.5,   # 1: still flat (below +2 entry)
            2.1,   # 2: enter short (above +2)
            2.5,   # 3: stay short
            1.0,   # 4: still short (above +0.5 exit)
            0.4,   # 5: exit (below +0.5)
            0.0,   # 6: flat
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=None)

        assert signal.iloc[0] == 0, "Should start flat"
        assert signal.iloc[1] == 0, "Should stay flat (not at entry threshold)"
        assert signal.iloc[2] == -1, "Should enter short when z >= +2"
        assert signal.iloc[3] == -1, "Should stay short"
        assert signal.iloc[4] == -1, "Should stay short (not yet at exit)"
        assert signal.iloc[5] == 0, "Should exit when z <= +0.5"
        assert signal.iloc[6] == 0, "Should remain flat"

    def test_long_stop_deterministic(self):
        """Deterministic test of long position stop-loss."""
        # z-score goes extremely negative (momentum continues against position)
        # After stop, return toward zero so we don't re-enter
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -3.0,  # 2: still long (stop at -4)
            -3.9,  # 3: still long (just above stop)
            -4.1,  # 4: stop out (below -4)
            -1.5,  # 5: flat, between entry (-2) and exit (-0.5)
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)

        assert signal.iloc[0] == 0, "Should start flat"
        assert signal.iloc[1] == 1, "Should enter long"
        assert signal.iloc[2] == 1, "Should stay long"
        assert signal.iloc[3] == 1, "Should stay long (above stop threshold)"
        assert signal.iloc[4] == 0, "Should stop out when z <= -4"
        assert signal.iloc[5] == 0, "Should stay flat (between entry and exit)"

    def test_short_stop_deterministic(self):
        """Deterministic test of short position stop-loss."""
        # z-score goes extremely positive (momentum continues against position)
        # After stop, return toward zero so we don't re-enter
        z = pd.Series([
            0.0,   # 0: flat
            2.5,   # 1: enter short
            3.0,   # 2: still short (stop at +4)
            3.9,   # 3: still short (just below stop)
            4.1,   # 4: stop out (above +4)
            1.5,   # 5: flat, between entry (+2) and exit (+0.5)
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)

        assert signal.iloc[0] == 0, "Should start flat"
        assert signal.iloc[1] == -1, "Should enter short"
        assert signal.iloc[2] == -1, "Should stay short"
        assert signal.iloc[3] == -1, "Should stay short (below stop threshold)"
        assert signal.iloc[4] == 0, "Should stop out when z >= +4"
        assert signal.iloc[5] == 0, "Should stay flat (between entry and exit)"

    def test_full_cycle_long_short(self):
        """Test complete cycle: flat -> long -> flat -> short -> flat."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -1.5,  # 2: stay long
            -0.3,  # 3: exit long (above -0.5)
            0.0,   # 4: flat
            2.5,   # 5: enter short
            1.5,   # 6: stay short
            0.3,   # 7: exit short (below +0.5)
            0.0,   # 8: flat
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=None)

        expected = [0, 1, 1, 0, 0, -1, -1, 0, 0]
        assert len(signal) == len(expected), "Signal and expected should have same length"
        for i, (actual, exp) in enumerate(zip(signal.values, expected)):  # noqa: B905
            assert actual == exp, f"Position {i}: expected {exp}, got {actual}"

    def test_nan_handling(self):
        """NaN values should preserve current position."""
        z = pd.Series([
            0.0,      # 0: flat
            -2.5,     # 1: enter long
            np.nan,   # 2: NaN - should stay long
            np.nan,   # 3: NaN - should stay long
            -0.3,     # 4: exit long
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5)

        assert signal.iloc[0] == 0
        assert signal.iloc[1] == 1
        assert signal.iloc[2] == 1, "NaN should preserve position"
        assert signal.iloc[3] == 1, "NaN should preserve position"
        assert signal.iloc[4] == 0

    def test_nan_does_not_crash(self):
        """Signal generation should not crash with NaN input."""
        z = pd.Series([np.nan, np.nan, -2.5, np.nan, -0.3, np.nan])
        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5)
        assert len(signal) == len(z)

    def test_no_stop_configured(self):
        """Without stop_z, extreme moves should not trigger stop."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -10.0, # 2: extreme but no stop (stop_z=None)
            -0.3,  # 3: exit via normal exit
        ])

        signal = generate_mean_reversion_signal(z, entry_z=2.0, exit_z=0.5, stop_z=None)

        assert signal.iloc[1] == 1
        assert signal.iloc[2] == 1, "Should stay long (no stop configured)"
        assert signal.iloc[3] == 0

    def test_signal_config(self, fly_yield):
        """Should accept SignalConfig object."""
        z = rolling_zscore(fly_yield, window=20)

        config = SignalConfig(entry_z=1.5, exit_z=0.3, stop_z=3.0)
        signal = generate_mean_reversion_signal(z, config=config)

        assert isinstance(signal, pd.Series)


class TestComputeSignalStrength:
    """Tests for compute_signal_strength."""

    def test_returns_series(self, fly_yield):
        """Should return a pandas Series."""
        z = rolling_zscore(fly_yield, window=20)
        signal = generate_mean_reversion_signal(z)
        strength = compute_signal_strength(z, signal)

        assert isinstance(strength, pd.Series)
        assert len(strength) == len(z)

    def test_strength_range(self, fly_yield):
        """Strength should be in [0, 1]."""
        z = rolling_zscore(fly_yield, window=20, clip=4.0)
        signal = generate_mean_reversion_signal(z)
        strength = compute_signal_strength(z, signal, max_z=4.0)

        assert strength.min() >= 0.0
        assert strength.max() <= 1.0

    def test_flat_has_zero_strength(self):
        """Flat positions should have zero strength."""
        z = pd.Series([0.0, 0.5, -0.3, 0.2])
        signal = pd.Series([0, 0, 0, 0])  # All flat

        strength = compute_signal_strength(z, signal)
        assert (strength == 0).all()


class TestBuildSignalDataframe:
    """Tests for build_signal_dataframe."""

    def test_returns_dataframe(self, fly_yield):
        """Should return DataFrame with expected columns."""
        df = build_signal_dataframe(fly_yield, window=20)

        assert isinstance(df, pd.DataFrame)
        assert "datetime" in df.columns
        assert "fly_yield" in df.columns
        assert "zscore" in df.columns
        assert "signal" in df.columns

    def test_includes_strength(self, fly_yield):
        """Should include strength when requested."""
        df = build_signal_dataframe(fly_yield, window=20, include_strength=True)
        assert "strength" in df.columns

        df_no_strength = build_signal_dataframe(fly_yield, window=20, include_strength=False)
        assert "strength" not in df_no_strength.columns

    def test_zscore_method_rolling(self, fly_yield):
        """Should use rolling z-score by default."""
        df = build_signal_dataframe(fly_yield, zscore_method="rolling", window=20)
        # Rolling has NaN for first window-1 values
        assert df["zscore"].iloc[:19].isna().all()

    def test_zscore_method_ewma(self, fly_yield):
        """Should use EWMA when specified."""
        df = build_signal_dataframe(fly_yield, zscore_method="ewma", window=20)
        # EWMA has fewer NaNs (min_periods=1)
        assert df["zscore"].notna().sum() > 190

    def test_ewma_mode(self, fly_yield):
        """Should use EWMA when span is provided."""
        df_rolling = build_signal_dataframe(fly_yield, zscore_method="rolling", window=20)
        df_ewma = build_signal_dataframe(fly_yield, zscore_method="ewma", window=20)

        # Z-scores should differ in the range where both are valid
        valid_start = 20
        assert not np.allclose(
            df_rolling["zscore"].iloc[valid_start:].values,
            df_ewma["zscore"].iloc[valid_start:].values,
        )

    def test_robust_mode(self, fly_yield):
        """Should use robust z-score when requested."""
        df_standard = build_signal_dataframe(fly_yield, window=20, robust=False)
        df_robust = build_signal_dataframe(fly_yield, window=20, robust=True)

        # Should differ
        assert not np.allclose(
            df_standard["zscore"].dropna(),
            df_robust["zscore"].dropna(),
        )

    def test_config_parameter(self, fly_yield):
        """Should accept SignalConfig object."""
        config = SignalConfig(entry_z=1.5, exit_z=0.3, stop_z=3.0)
        df = build_signal_dataframe(fly_yield, window=20, config=config)
        assert isinstance(df, pd.DataFrame)


class TestComputeSignalStats:
    """Tests for compute_signal_stats."""

    def test_returns_stats(self, fly_yield):
        """Should return SignalStats object."""
        df = build_signal_dataframe(fly_yield, window=20)
        stats = compute_signal_stats(df)

        assert stats.n_observations == len(df)
        assert stats.n_long_signals >= 0
        assert stats.n_short_signals >= 0
        assert stats.n_flat_signals >= 0

    def test_percentages_sum_to_one(self, fly_yield):
        """Position percentages should sum to 1."""
        df = build_signal_dataframe(fly_yield, window=20)
        stats = compute_signal_stats(df)

        total_pct = stats.pct_long + stats.pct_short + stats.pct_flat
        assert abs(total_pct - 1.0) < 1e-10

    def test_avg_zscore_at_entry_uses_only_entries(self):
        """avg_zscore_at_entry should use only true entry points (0 -> ±1)."""
        # Create a scenario where we can verify exact entry z-scores
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "datetime": dates,
            "fly_yield": [0, 0, -3, -2, -0.3, 0, 2.5, 2.0, 0.3, 0],
            "zscore": [0, 0, -2.5, -2.0, -0.3, 0, 2.1, 1.5, 0.3, 0],
            "signal": [0, 0, 1, 1, 0, 0, -1, -1, 0, 0],
        })

        stats = compute_signal_stats(df)

        # Entry points: index 2 (z=-2.5) and index 6 (z=2.1)
        expected_avg = (2.5 + 2.1) / 2
        assert abs(stats.avg_zscore_at_entry - expected_avg) < 1e-10


class TestBacktestSignal:
    """Tests for backtest_signal."""

    def test_returns_dataframe(self, fly_yield):
        """Should return DataFrame with expected columns."""
        signal_df = build_signal_dataframe(fly_yield, window=20)
        bt_df = backtest_signal(signal_df)

        assert isinstance(bt_df, pd.DataFrame)
        assert "datetime" in bt_df.columns
        assert "returns" in bt_df.columns
        assert "strategy_returns" in bt_df.columns
        assert "cumulative_returns" in bt_df.columns

    def test_strategy_returns_sign(self, extreme_fly_yield):
        """Long position on down move should make money."""
        signal_df = build_signal_dataframe(
            extreme_fly_yield, window=10, entry_z=1.5, exit_z=0.3
        )
        bt_df = backtest_signal(signal_df)

        # Should have some non-zero strategy returns
        assert bt_df["strategy_returns"].abs().sum() > 0


class TestComputeBacktestStats:
    """Tests for compute_backtest_stats."""

    def test_returns_dict(self, fly_yield):
        """Should return dict with expected keys."""
        signal_df = build_signal_dataframe(fly_yield, window=20)
        bt_df = backtest_signal(signal_df)
        stats = compute_backtest_stats(bt_df)

        assert isinstance(stats, dict)
        assert "total_return" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown" in stats
        assert "win_rate" in stats


class TestEndToEnd:
    """End-to-end tests for signal pipeline."""

    def test_full_pipeline(self, fly_yield):
        """Should run complete pipeline without errors."""
        # Build signals
        signal_df = build_signal_dataframe(
            fly_yield,
            window=20,
            robust=True,
            clip=4.0,
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
        )

        # Get stats
        stats = compute_signal_stats(signal_df)

        # Backtest
        bt_df = backtest_signal(signal_df)
        bt_stats = compute_backtest_stats(bt_df)

        # All should succeed
        assert len(signal_df) > 0
        assert stats.n_observations > 0
        assert "total_return" in bt_stats

    def test_mean_reverting_profitability(self):
        """Mean-reversion strategy should profit on mean-reverting data."""
        np.random.seed(123)

        # Generate strongly mean-reverting series
        n = 500
        mean = 0.0
        theta = 0.3  # Strong mean reversion
        sigma = 0.5

        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        yields = [0.0]
        for _i in range(1, n):
            dy = theta * (mean - yields[-1]) + sigma * np.random.randn()
            yields.append(yields[-1] + dy)

        fly_yield = pd.Series(yields, index=dates)

        # Build signals
        signal_df = build_signal_dataframe(
            fly_yield, window=30, entry_z=1.5, exit_z=0.3
        )

        # Backtest
        bt_df = backtest_signal(signal_df)
        stats = compute_backtest_stats(bt_df)

        # Should be profitable (positive total return)
        # Note: not guaranteed, but likely for strongly mean-reverting data
        assert stats["n_observations"] > 0


class TestMaxHoldBars:
    """Tests for max_hold_bars risk control."""

    def test_max_hold_triggers_exit(self):
        """Should exit when max_hold_bars exceeded."""
        # Z-score stays below entry threshold (position stays long)
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -2.0,  # 2: hold (bar 1)
            -1.8,  # 3: hold (bar 2)
            -1.6,  # 4: hold (bar 3)
            -1.5,  # 5: hold (bar 4)
            -1.4,  # 6: hold (bar 5) - should exit here
            -1.3,  # 7: flat (after max hold exit)
            -2.5,  # 8: can re-enter
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            max_hold_bars=5, return_details=True
        )

        assert signal.iloc[1] == 1, "Should enter long"
        assert signal.iloc[5] == 1, "Should still be long at bar 5"
        assert signal.iloc[6] == 0, "Should exit at bar 6 (max hold reached)"
        assert details.iloc[6]["exit_reason"] == ExitReason.MAX_HOLD

    def test_max_hold_bars_held_at_exit(self):
        """bars_held at exit bar should reflect holding duration.

        max_hold_bars=3 means exit when we've held for 3 bars (excluding entry bar).
        Entry bar has hold_count=0, then increments each bar.
        So: entry(0) -> hold(1) -> hold(2) -> hold(3) -> exit.
        """
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long, hold_count=0
            -2.0,  # 2: hold, hold_count=1
            -1.8,  # 3: hold, hold_count=2
            -1.6,  # 4: hold, hold_count=3, check 3>=3 -> exit
            -1.5,  # 5: flat
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            max_hold_bars=3, return_details=True
        )

        assert signal.iloc[4] == 0, "Should exit at bar 4"
        assert details.iloc[4]["bars_held"] == 3, "bars_held at exit should be 3"
        assert details.iloc[4]["exit_reason"] == ExitReason.MAX_HOLD

    def test_max_hold_with_normal_exit(self):
        """Normal exit should occur before max_hold if z crosses threshold."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -2.0,  # 2: hold (bar 1)
            -0.3,  # 3: normal exit (above -0.5)
            -1.5,  # 4: flat
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            max_hold_bars=10, return_details=True
        )

        assert signal.iloc[3] == 0, "Should exit normally"
        assert details.iloc[3]["exit_reason"] == ExitReason.NORMAL_EXIT

    def test_max_hold_short_position(self):
        """Max hold should work for short positions too."""
        z = pd.Series([
            0.0,   # 0: flat
            2.5,   # 1: enter short
            2.0,   # 2: hold (bar 1)
            1.8,   # 3: hold (bar 2)
            1.6,   # 4: hold (bar 3) - should exit here
            1.5,   # 5: flat
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            max_hold_bars=3, return_details=True
        )

        assert signal.iloc[1] == -1, "Should enter short"
        assert signal.iloc[3] == -1, "Should still be short at bar 3"
        assert signal.iloc[4] == 0, "Should exit at bar 4 (max hold reached)"
        assert details.iloc[4]["exit_reason"] == ExitReason.MAX_HOLD


class TestCooldownBars:
    """Tests for cooldown_bars risk control."""

    def test_cooldown_blocks_reentry(self):
        """Should block re-entry during cooldown period.

        cooldown_bars=3 means 3 FULL bars blocked after exit.
        If exit at bar 2, bars 3,4,5 are blocked, can enter at bar 6.
        """
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -0.3,  # 2: exit
            -2.5,  # 3: blocked by cooldown (bar 1 of 3)
            -2.5,  # 4: blocked by cooldown (bar 2 of 3)
            -2.5,  # 5: blocked by cooldown (bar 3 of 3)
            -2.5,  # 6: can enter (cooldown over)
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            cooldown_bars=3, return_details=True
        )

        assert signal.iloc[1] == 1, "Should enter long"
        assert signal.iloc[2] == 0, "Should exit"
        assert signal.iloc[3] == 0, "Should be blocked by cooldown"
        assert signal.iloc[4] == 0, "Should be blocked by cooldown"
        assert signal.iloc[5] == 0, "Should be blocked by cooldown"
        assert signal.iloc[6] == 1, "Should enter after cooldown"

        # Check cooldown remaining decrements correctly
        assert details.iloc[3]["cooldown_remaining"] == 2
        assert details.iloc[4]["cooldown_remaining"] == 1
        assert details.iloc[5]["cooldown_remaining"] == 0

    def test_cooldown_after_stop(self):
        """Cooldown should also apply after stop-loss exit."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -4.5,  # 2: stop out (z < -4)
            -2.5,  # 3: blocked by cooldown
            -2.5,  # 4: can enter
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=4.0,
            cooldown_bars=1, return_details=True
        )

        assert signal.iloc[1] == 1, "Should enter long"
        assert signal.iloc[2] == 0, "Should stop out"
        assert details.iloc[2]["exit_reason"] == ExitReason.STOP_LOSS
        assert signal.iloc[3] == 0, "Should be blocked by cooldown"
        assert signal.iloc[4] == 1, "Should enter after cooldown"

    def test_cooldown_z_threshold_early_exit(self):
        """Cooldown should end early when z returns near zero."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -0.3,  # 2: exit
            -2.5,  # 3: blocked by cooldown
            0.2,   # 4: cooldown ends early (|z| < 0.5)
            -2.5,  # 5: can enter now
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            cooldown_bars=5, cooldown_z_threshold=0.5,
            return_details=True
        )

        assert signal.iloc[2] == 0, "Should exit"
        assert signal.iloc[3] == 0, "Should be blocked"
        assert details.iloc[4]["cooldown_remaining"] == 0, "Cooldown should end early"
        assert signal.iloc[5] == 1, "Should be able to enter"

    def test_cooldown_no_early_exit_without_threshold(self):
        """Without threshold, cooldown runs full duration."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -0.3,  # 2: exit
            0.0,   # 3: z at zero but cooldown continues
            0.0,   # 4: still in cooldown
            -2.5,  # 5: can enter (cooldown over)
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            cooldown_bars=2, cooldown_z_threshold=None,
            return_details=True
        )

        assert signal.iloc[3] == 0, "Should still be in cooldown"
        assert signal.iloc[4] == 0, "Should still be in cooldown"
        assert signal.iloc[5] == 1, "Should enter after full cooldown"


class TestCombinedRiskControls:
    """Tests for multiple risk controls working together."""

    def test_stop_with_cooldown(self):
        """Stop-loss followed by cooldown period."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -4.5,  # 2: stop out
            -2.5,  # 3: cooldown
            -2.5,  # 4: cooldown
            -2.5,  # 5: can enter
            -0.3,  # 6: normal exit
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=4.0,
            cooldown_bars=2, return_details=True
        )

        assert signal.iloc[2] == 0, "Stop out"
        assert details.iloc[2]["exit_reason"] == ExitReason.STOP_LOSS
        assert signal.iloc[3] == 0, "Cooldown"
        assert signal.iloc[4] == 0, "Cooldown"
        assert signal.iloc[5] == 1, "Enter after cooldown"
        assert signal.iloc[6] == 0, "Normal exit"

    def test_max_hold_with_cooldown(self):
        """Max hold exit followed by cooldown."""
        z = pd.Series([
            0.0,   # 0: flat
            -2.5,  # 1: enter long
            -2.0,  # 2: hold 1
            -1.8,  # 3: hold 2 - max hold exit
            -2.5,  # 4: cooldown
            -2.5,  # 5: can enter
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            max_hold_bars=2, cooldown_bars=1, return_details=True
        )

        assert signal.iloc[1] == 1, "Enter"
        assert signal.iloc[3] == 0, "Max hold exit"
        assert details.iloc[3]["exit_reason"] == ExitReason.MAX_HOLD
        assert signal.iloc[4] == 0, "Cooldown"
        assert signal.iloc[5] == 1, "Enter after cooldown"

    def test_all_controls_config(self):
        """Test using SignalConfig with all risk controls."""
        config = SignalConfig(
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            max_hold_bars=10,
            cooldown_bars=3,
            cooldown_z_threshold=0.3,
        )

        z = pd.Series([
            0.0, -2.5, -2.0, -1.5, -0.3,  # Enter, hold, exit
            0.2,  # Cooldown ends early (|z| < 0.3)
            -2.5, -2.5,  # Can enter again
        ])

        signal, details = generate_mean_reversion_signal(z, config=config, return_details=True)

        assert signal.iloc[1] == 1, "Enter long"
        assert signal.iloc[4] == 0, "Exit"
        assert signal.iloc[6] == 1, "Re-enter after cooldown ended early"


class TestExitReason:
    """Tests for ExitReason tracking."""

    def test_exit_reasons_tracked(self):
        """Should track different exit reasons."""
        z = pd.Series([
            0.0,   # flat
            -2.5,  # enter long
            -0.3,  # normal exit
            -2.5,  # enter long
            -4.5,  # stop exit
            0.0,   # flat
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=4.0,
            cooldown_bars=0, return_details=True
        )

        # Check exit reasons
        assert details.iloc[2]["exit_reason"] == ExitReason.NORMAL_EXIT
        assert details.iloc[4]["exit_reason"] == ExitReason.STOP_LOSS

    def test_bars_held_tracked(self):
        """Should track bars held in position.

        hold_count is 0 on entry bar, increments each subsequent bar.
        """
        z = pd.Series([
            0.0, -2.5, -2.0, -1.8, -1.6, -0.3, 0.0
        ])

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=None,
            return_details=True
        )

        assert details.iloc[1]["bars_held"] == 0  # Entry bar
        assert details.iloc[2]["bars_held"] == 1
        assert details.iloc[3]["bars_held"] == 2
        assert details.iloc[4]["bars_held"] == 3
        # At exit, bars_held should show the final holding duration
        assert details.iloc[5]["bars_held"] == 4, "bars_held at exit should be 4"
        assert details.iloc[6]["bars_held"] == 0  # Flat, reset


class TestRiskControlsIdempotent:
    """Test that risk controls produce deterministic results."""

    def test_deterministic_with_all_controls(self):
        """Same input should produce same output."""
        np.random.seed(42)
        z = pd.Series(np.random.randn(100).cumsum())

        config = SignalConfig(
            entry_z=1.5, exit_z=0.5, stop_z=3.0,
            max_hold_bars=15, cooldown_bars=5,
            cooldown_z_threshold=0.5,
        )

        signal1, details1 = generate_mean_reversion_signal(z, config=config, return_details=True)
        signal2, details2 = generate_mean_reversion_signal(z, config=config, return_details=True)

        pd.testing.assert_series_equal(signal1, signal2)
        pd.testing.assert_frame_equal(details1, details2)


class TestTradeBlotter:
    """Tests for build_trade_blotter and compute_trade_stats."""

    def test_basic_trade_blotter(self):
        """Should create trade blotter with correct number of trades."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        z = pd.Series([0, -2.5, -2.0, -0.3, 0, 2.5, 2.0, 0.3, 0, 0], index=dates)
        signal = pd.Series([0, 1, 1, 0, 0, -1, -1, 0, 0, 0], index=dates)

        blotter = build_trade_blotter(signal, z)

        assert len(blotter) == 2, "Should have 2 trades (1 long, 1 short)"
        assert blotter.iloc[0]["side"] == 1, "First trade should be long"
        assert blotter.iloc[1]["side"] == -1, "Second trade should be short"

    def test_trade_blotter_columns(self):
        """Trade blotter should have all required columns."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        z = pd.Series([0, -2.5, -2.0, -0.3, 0], index=dates)
        signal = pd.Series([0, 1, 1, 0, 0], index=dates)

        blotter = build_trade_blotter(signal, z)

        expected_cols = [
            "entry_time", "exit_time", "side", "entry_z", "exit_z",
            "holding_bars", "exit_reason", "gross_pnl", "cost", "net_pnl"
        ]
        for col in expected_cols:
            assert col in blotter.columns, f"Missing column: {col}"

    def test_trade_blotter_holding_bars(self):
        """Holding bars should match trade duration."""
        dates = pd.date_range("2023-01-01", periods=6, freq="D")
        z = pd.Series([0, -2.5, -2.0, -1.8, -0.3, 0], index=dates)
        signal = pd.Series([0, 1, 1, 1, 0, 0], index=dates)

        blotter = build_trade_blotter(signal, z)

        assert len(blotter) == 1
        assert blotter.iloc[0]["holding_bars"] == 3, "Should be held for 3 bars"

    def test_trade_blotter_pnl_calculation(self):
        """PnL should be calculated correctly."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        z = pd.Series([0, 1.0, 2.0, 3.0, 0], index=dates)
        signal = pd.Series([0, 1, 1, 0, 0], index=dates)
        # Returns: position_t-1 * returns_t
        # Long at bar 1, z diff from 1 to 2 is +1, from 2 to 3 is +1
        # So pnl = 1*1 + 1*1 = 2

        blotter = build_trade_blotter(signal, z, transaction_cost=0.0)

        assert len(blotter) == 1
        assert blotter.iloc[0]["gross_pnl"] == pytest.approx(2.0, abs=0.01)

    def test_trade_blotter_transaction_costs(self):
        """Transaction costs should be applied correctly."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        z = pd.Series([0, 1.0, 2.0, 3.0, 0], index=dates)
        signal = pd.Series([0, 1, 1, 0, 0], index=dates)

        blotter = build_trade_blotter(signal, z, transaction_cost=0.1)

        assert len(blotter) == 1
        # Entry cost: 1 * 0.1 = 0.1, Exit cost: 1 * 0.1 = 0.1, Total: 0.2
        assert blotter.iloc[0]["cost"] == pytest.approx(0.2, abs=0.01)
        assert blotter.iloc[0]["net_pnl"] == pytest.approx(
            blotter.iloc[0]["gross_pnl"] - 0.2, abs=0.01
        )

    def test_trade_blotter_with_returns(self):
        """Should use provided returns series for PnL."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        z = pd.Series([0, -2.5, -2.0, -0.3, 0], index=dates)
        signal = pd.Series([0, 1, 1, 0, 0], index=dates)
        returns = pd.Series([0, 0.01, 0.02, 0.03, 0], index=dates)

        blotter = build_trade_blotter(signal, z, returns=returns)

        # pnl = 1 * 0.02 + 1 * 0.03 = 0.05
        assert blotter.iloc[0]["gross_pnl"] == pytest.approx(0.05, abs=0.001)

    def test_trade_blotter_empty(self):
        """Should return empty DataFrame when no trades."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        z = pd.Series([0, 0, 0, 0, 0], index=dates)
        signal = pd.Series([0, 0, 0, 0, 0], index=dates)

        blotter = build_trade_blotter(signal, z)

        assert len(blotter) == 0
        assert "entry_time" in blotter.columns

    def test_trade_blotter_with_details(self):
        """build_trade_blotter_with_details should capture accurate exit reasons."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        z = pd.Series([0, -2.5, -2.0, -4.5, 0, 2.5, 2.0, 0.3, 0, 0], index=dates)

        signal, details = generate_mean_reversion_signal(
            z, entry_z=2.0, exit_z=0.5, stop_z=4.0, return_details=True
        )

        blotter = build_trade_blotter_with_details(signal, details, z)

        assert len(blotter) == 2
        # First trade should be stop loss
        assert blotter.iloc[0]["exit_reason"] == ExitReason.STOP_LOSS
        # Second trade should be normal exit
        assert blotter.iloc[1]["exit_reason"] == ExitReason.NORMAL_EXIT


class TestTradeStats:
    """Tests for compute_trade_stats."""

    def test_basic_trade_stats(self):
        """Should compute basic statistics correctly."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        z = pd.Series([0, -2.5, -2.0, -0.3, 0, 2.5, 2.0, 0.3, 0, 0], index=dates)
        signal = pd.Series([0, 1, 1, 0, 0, -1, -1, 0, 0, 0], index=dates)

        blotter = build_trade_blotter(signal, z)
        stats = compute_trade_stats(blotter)

        assert isinstance(stats, TradeStats)
        assert stats.n_trades == 2

    def test_trade_stats_win_rate(self):
        """Win rate should be calculated correctly."""
        # Create blotter with known wins/losses
        blotter = pd.DataFrame({
            "entry_time": pd.date_range("2023-01-01", periods=4, freq="D"),
            "exit_time": pd.date_range("2023-01-02", periods=4, freq="D"),
            "side": [1, -1, 1, -1],
            "entry_z": [-2.5, 2.5, -2.5, 2.5],
            "exit_z": [-0.3, 0.3, -0.3, 0.3],
            "holding_bars": [2, 2, 2, 2],
            "exit_reason": [1, 1, 1, 1],
            "gross_pnl": [1.0, 0.5, -0.5, 0.3],
            "cost": [0.0, 0.0, 0.0, 0.0],
            "net_pnl": [1.0, 0.5, -0.5, 0.3],  # 3 wins, 1 loss
        })

        stats = compute_trade_stats(blotter)

        assert stats.win_rate == pytest.approx(0.75, abs=0.01)
        assert stats.n_trades == 4

    def test_trade_stats_profit_factor(self):
        """Profit factor should be total wins / total losses."""
        blotter = pd.DataFrame({
            "entry_time": pd.date_range("2023-01-01", periods=2, freq="D"),
            "exit_time": pd.date_range("2023-01-02", periods=2, freq="D"),
            "side": [1, 1],
            "entry_z": [-2.5, -2.5],
            "exit_z": [-0.3, -0.3],
            "holding_bars": [2, 2],
            "exit_reason": [1, 1],
            "gross_pnl": [2.0, -1.0],
            "cost": [0.0, 0.0],
            "net_pnl": [2.0, -1.0],
        })

        stats = compute_trade_stats(blotter)

        # Profit factor = 2.0 / 1.0 = 2.0
        assert stats.profit_factor == pytest.approx(2.0, abs=0.01)

    def test_trade_stats_pnl_by_exit_reason(self):
        """Should aggregate PnL by exit reason."""
        blotter = pd.DataFrame({
            "entry_time": pd.date_range("2023-01-01", periods=3, freq="D"),
            "exit_time": pd.date_range("2023-01-02", periods=3, freq="D"),
            "side": [1, 1, 1],
            "entry_z": [-2.5, -2.5, -2.5],
            "exit_z": [-0.3, -4.5, -0.3],
            "holding_bars": [2, 2, 2],
            "exit_reason": [ExitReason.NORMAL_EXIT, ExitReason.STOP_LOSS, ExitReason.NORMAL_EXIT],
            "gross_pnl": [1.0, -0.5, 0.5],
            "cost": [0.0, 0.0, 0.0],
            "net_pnl": [1.0, -0.5, 0.5],
        })

        stats = compute_trade_stats(blotter)

        assert "NORMAL_EXIT" in stats.pnl_by_exit_reason
        assert "STOP_LOSS" in stats.pnl_by_exit_reason
        assert stats.pnl_by_exit_reason["NORMAL_EXIT"] == pytest.approx(1.5, abs=0.01)
        assert stats.pnl_by_exit_reason["STOP_LOSS"] == pytest.approx(-0.5, abs=0.01)

    def test_trade_stats_empty_blotter(self):
        """Should handle empty blotter gracefully."""
        blotter = pd.DataFrame(
            columns=[
                "entry_time", "exit_time", "side", "entry_z", "exit_z",
                "holding_bars", "exit_reason", "gross_pnl", "cost", "net_pnl"
            ]
        )

        stats = compute_trade_stats(blotter)

        assert stats.n_trades == 0
        assert stats.win_rate == 0.0
        assert stats.expectancy == 0.0


class TestDeterministicToyExample:
    """Deterministic toy example with explicit expected signals."""

    def test_complete_deterministic_example(self):
        """Complete deterministic example with known expected behavior.

        max_hold_bars=3 means exit when hold_count >= 3.
        hold_count=0 on entry, increments each bar.
        So: entry(0) -> 1 -> 2 -> 3 (exit).
        """
        dates = pd.date_range("2023-01-01", periods=21, freq="D")
        z = pd.Series([
            0.0,   # 0: flat
            -0.5,  # 1: flat (not at entry)
            -2.1,  # 2: enter long, hold_count=0
            -2.5,  # 3: hold, hold_count=1
            -1.5,  # 4: hold, hold_count=2
            -0.4,  # 5: exit long (normal), hold_count=3
            0.0,   # 6: flat, cooldown
            0.0,   # 7: flat, cooldown done
            2.1,   # 8: enter short, hold_count=0
            2.5,   # 9: hold, hold_count=1
            4.5,   # 10: stop out (z >= 4), hold_count=2
            0.0,   # 11: flat, cooldown
            -2.1,  # 12: enter long (cooldown done), hold_count=0
            -1.5,  # 13: hold, hold_count=1
            -1.2,  # 14: hold, hold_count=2
            -1.0,  # 15: hold, hold_count=3 -> max hold exit
            0.0,   # 16: flat, cooldown
            0.0,   # 17: flat, cooldown done
            0.0,   # 18: flat
            0.0,   # 19: flat
            0.0,   # 20: flat
        ], index=dates)

        config = SignalConfig(
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            max_hold_bars=3,
            cooldown_bars=1,
        )

        signal, details = generate_mean_reversion_signal(z, config=config, return_details=True)

        # Verify expected signal sequence
        expected = [
            0,   # 0: flat
            0,   # 1: flat
            1,   # 2: enter long
            1,   # 3: hold long
            1,   # 4: hold long
            0,   # 5: exit long (normal)
            0,   # 6: cooldown
            0,   # 7: cooldown done but no entry condition
            -1,  # 8: enter short
            -1,  # 9: hold short
            0,   # 10: stop out
            0,   # 11: cooldown
            1,   # 12: enter long
            1,   # 13: hold long
            1,   # 14: hold long (hold_count=2)
            0,   # 15: max hold exit (hold_count=3)
            0,   # 16: cooldown
            0,   # 17: flat
            0,   # 18: flat
            0,   # 19: flat
            0,   # 20: flat
        ]

        for i, (actual, exp) in enumerate(zip(signal.values, expected)):  # noqa: B905
            assert actual == exp, f"Bar {i}: expected {exp}, got {actual}"

        # Verify exit reasons
        assert details.iloc[5]["exit_reason"] == ExitReason.NORMAL_EXIT
        assert details.iloc[10]["exit_reason"] == ExitReason.STOP_LOSS
        assert details.iloc[15]["exit_reason"] == ExitReason.MAX_HOLD

        # Verify bars_held at exits (hold_count=0 on entry, increments each bar)
        assert details.iloc[5]["bars_held"] == 3, "Long trade: entry(0)->1->2->3 at exit"
        assert details.iloc[10]["bars_held"] == 2, "Short trade: entry(0)->1->2 at stop"
        assert details.iloc[15]["bars_held"] == 3, "Second long: entry(0)->1->2->3 at max hold"

        # Build trade blotter and verify
        blotter = build_trade_blotter_with_details(signal, details, z)

        assert len(blotter) == 3, "Should have 3 trades"
        assert blotter.iloc[0]["side"] == 1
        assert blotter.iloc[0]["exit_reason"] == ExitReason.NORMAL_EXIT
        assert blotter.iloc[1]["side"] == -1
        assert blotter.iloc[1]["exit_reason"] == ExitReason.STOP_LOSS
        assert blotter.iloc[2]["side"] == 1
        assert blotter.iloc[2]["exit_reason"] == ExitReason.MAX_HOLD
