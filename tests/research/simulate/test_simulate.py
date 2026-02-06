"""Tests for market data simulation."""

import pandas as pd

from mlstudy.research.simulate.market import simulate_market_data, simulate_with_known_signal


class TestSimulateMarketData:
    """Tests for simulate_market_data."""

    def test_returns_dataframe(self) -> None:
        """Should return a DataFrame."""
        df = simulate_market_data(n_assets=2, n_periods=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self) -> None:
        """Should have expected columns."""
        df = simulate_market_data(n_assets=2, n_periods=100, seed=42)
        expected_cols = ["datetime", "asset", "close", "volume", "flow_imbalance"]
        assert list(df.columns) == expected_cols

    def test_correct_shape(self) -> None:
        """Should have correct number of rows."""
        n_assets = 3
        n_periods = 100
        df = simulate_market_data(n_assets=n_assets, n_periods=n_periods, seed=42)
        assert len(df) == n_assets * n_periods

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical results."""
        df1 = simulate_market_data(n_assets=3, n_periods=100, seed=42)
        df2 = simulate_market_data(n_assets=3, n_periods=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_results(self) -> None:
        """Different seeds should produce different results."""
        df1 = simulate_market_data(n_assets=3, n_periods=100, seed=42)
        df2 = simulate_market_data(n_assets=3, n_periods=100, seed=123)
        assert not df1["close"].equals(df2["close"])

    def test_monotonic_datetime_per_group(self) -> None:
        """Datetime should be monotonically increasing within each asset."""
        df = simulate_market_data(n_assets=3, n_periods=100, seed=42)

        for asset in df["asset"].unique():
            asset_df = df[df["asset"] == asset]
            datetimes = asset_df["datetime"].values
            assert all(datetimes[i] < datetimes[i + 1] for i in range(len(datetimes) - 1))

    def test_prices_positive(self) -> None:
        """Prices should always be positive."""
        df = simulate_market_data(n_assets=5, n_periods=500, seed=42)
        assert (df["close"] > 0).all()

    def test_volume_positive(self) -> None:
        """Volume should always be positive."""
        df = simulate_market_data(n_assets=5, n_periods=500, seed=42)
        assert (df["volume"] > 0).all()

    def test_flow_imbalance_bounded(self) -> None:
        """Flow imbalance should be between -1 and 1."""
        df = simulate_market_data(n_assets=5, n_periods=500, seed=42)
        assert (df["flow_imbalance"] >= -1).all()
        assert (df["flow_imbalance"] <= 1).all()

    def test_unique_assets(self) -> None:
        """Should have correct number of unique assets."""
        n_assets = 5
        df = simulate_market_data(n_assets=n_assets, n_periods=100, seed=42)
        assert df["asset"].nunique() == n_assets

    def test_datetime_is_datetime_type(self) -> None:
        """Datetime column should be datetime64 type."""
        df = simulate_market_data(n_assets=2, n_periods=100, seed=42)
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])


class TestSimulateWithKnownSignal:
    """Tests for simulate_with_known_signal."""

    def test_returns_dataframe(self) -> None:
        """Should return a DataFrame."""
        df = simulate_with_known_signal(n_assets=2, n_periods=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self) -> None:
        """Should have expected columns."""
        df = simulate_with_known_signal(n_assets=2, n_periods=100, seed=42)
        expected_cols = ["datetime", "asset", "close", "volume", "flow_imbalance"]
        assert list(df.columns) == expected_cols

    def test_deterministic(self) -> None:
        """Same seed should produce identical results."""
        df1 = simulate_with_known_signal(n_assets=2, n_periods=100, seed=42)
        df2 = simulate_with_known_signal(n_assets=2, n_periods=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_signal_correlation(self) -> None:
        """With strong signal, lagged flow should correlate with returns."""
        df = simulate_with_known_signal(
            n_assets=1,
            n_periods=1000,
            seed=42,
            signal_lag=1,
            signal_strength=0.5,
        )

        # Compute returns
        df = df.sort_values("datetime")
        df["return"] = df["close"].pct_change()

        # Lag flow imbalance
        df["flow_lag1"] = df["flow_imbalance"].shift(1)

        # Check correlation (should be positive and meaningful)
        corr = df["return"].corr(df["flow_lag1"])
        assert corr > 0.2  # Should have positive correlation
