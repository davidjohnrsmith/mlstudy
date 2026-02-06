"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.core.features import (
    FeatureSpec,
    build_features,
    get_feature,
    list_features,
    quick_features,
)
from mlstudy.core.features.ops.groupby import grouped_lag, grouped_rolling_mean
from mlstudy.core.features.ops.time import lag, rolling_mean


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create synthetic DataFrame for testing.

    Contains:
    - datetime: increasing timestamps
    - symbol: 2 groups (A, B)
    - close: price-like column
    - volume: volume column
    - buy_volume, sell_volume: for flow features
    """
    np.random.seed(42)
    n_per_group = 100
    n = n_per_group * 2

    # Generate dates
    dates = pd.date_range("2024-01-01", periods=n_per_group, freq="D")
    dates = pd.concat([pd.Series(dates), pd.Series(dates)]).reset_index(drop=True)

    # Generate groups
    groups = ["A"] * n_per_group + ["B"] * n_per_group

    # Generate price-like data (random walk)
    np.random.seed(42)
    returns_a = np.random.randn(n_per_group) * 0.02
    returns_b = np.random.randn(n_per_group) * 0.02
    price_a = 100 * np.cumprod(1 + returns_a)
    price_b = 50 * np.cumprod(1 + returns_b)
    prices = np.concatenate([price_a, price_b])

    # Generate volume
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    # Generate flow columns
    buy_volume = volume * np.random.uniform(0.3, 0.7, n)
    sell_volume = volume - buy_volume

    df = pd.DataFrame({
        "datetime": dates,
        "symbol": groups,
        "close": prices,
        "volume": volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
    })

    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


class TestOpsTime:
    """Tests for time operations."""

    def test_lag_shift(self) -> None:
        """Lag should shift values correctly."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = lag(s, periods=1, shift=0)
        expected = pd.Series([np.nan, 1, 2, 3, 4])
        pd.testing.assert_series_equal(result, expected)

    def test_lag_with_leak_safe_shift(self) -> None:
        """Lag with shift=1 should add extra lag."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = lag(s, periods=1, shift=1)
        expected = pd.Series([np.nan, np.nan, 1, 2, 3])
        pd.testing.assert_series_equal(result, expected)

    def test_rolling_mean_shift(self) -> None:
        """Rolling mean should be leak-safe with shift."""
        s = pd.Series([1, 2, 3, 4, 5])
        # With shift=1, rolling mean at t uses values up to t-1
        # Shifted series: [NaN, 1, 2, 3, 4]
        result = rolling_mean(s, window=2, shift=1)
        # At index 0: window [NaN] -> NaN
        # At index 1: window [NaN, 1] -> only 1 valid, NaN
        # At index 2: window [1, 2] -> mean = 1.5
        # At index 3: window [2, 3] -> mean = 2.5
        # At index 4: window [3, 4] -> mean = 3.5
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.5
        assert result.iloc[3] == 2.5
        assert result.iloc[4] == 3.5


class TestOpsGroupby:
    """Tests for group-aware operations."""

    def test_grouped_lag(self, synthetic_df: pd.DataFrame) -> None:
        """Grouped lag should work within groups."""
        result = grouped_lag(synthetic_df, "close", "symbol", periods=1)

        # Check that lag is applied within groups
        for symbol in ["A", "B"]:
            mask = synthetic_df["symbol"] == symbol
            group_result = result[mask]
            group_close = synthetic_df.loc[mask, "close"]

            # First value should be NaN
            assert pd.isna(group_result.iloc[0])
            # Second value should equal first close
            assert group_result.iloc[1] == group_close.iloc[0]

    def test_grouped_rolling_mean(self, synthetic_df: pd.DataFrame) -> None:
        """Grouped rolling mean should compute within groups."""
        result = grouped_rolling_mean(
            synthetic_df, "close", "symbol", window=5, shift=1
        )

        # Should have NaN at beginning of each group
        for symbol in ["A", "B"]:
            mask = synthetic_df["symbol"] == symbol
            group_result = result[mask]
            # First 5 values should be NaN (shift=1 + window=5)
            assert group_result.iloc[:5].isna().all()


class TestNoLookahead:
    """Tests to verify no lookahead bias."""

    def test_returns_no_lookahead(self, synthetic_df: pd.DataFrame) -> None:
        """Returns at time t should not use price at t."""
        specs = [FeatureSpec(name="returns", params={"price_col": "close", "periods": 1})]
        X, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        # Get returns for group A
        mask = synthetic_df["symbol"] == "A"
        returns = X.loc[mask, "close_return_1"]
        close = synthetic_df.loc[mask, "close"]

        # Return at index i should be (close[i-1] - close[i-2]) / close[i-2]
        # Not (close[i] - close[i-1]) / close[i-1]
        for i in range(3, 10):
            expected = (close.iloc[i - 1] - close.iloc[i - 2]) / close.iloc[i - 2]
            assert abs(returns.iloc[i] - expected) < 1e-10

    def test_rolling_vol_no_lookahead(self, synthetic_df: pd.DataFrame) -> None:
        """Rolling volatility should not use current value."""
        specs = [
            FeatureSpec(
                name="rolling_volatility",
                params={"price_col": "close", "window": 5},
            )
        ]
        X, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        # Volatility at t should only use returns up to t-1
        mask = synthetic_df["symbol"] == "A"
        vol = X.loc[mask, "close_vol_5"]

        # Check that early values are NaN (need window + shift)
        assert vol.iloc[:6].isna().all()
        # Later values should be valid
        assert vol.iloc[10:].notna().all()


class TestDeterminism:
    """Tests for deterministic output."""

    def test_build_features_deterministic(self, synthetic_df: pd.DataFrame) -> None:
        """Same input should produce same output."""
        specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
            FeatureSpec(name="momentum", params={"price_col": "close", "window": 10}),
            FeatureSpec(name="volume_shock", params={"volume_col": "volume", "window": 10}),
        ]

        X1, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )
        X2, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        pd.testing.assert_frame_equal(X1, X2)


class TestRegistry:
    """Tests for feature registry."""

    def test_list_features_not_empty(self) -> None:
        """Registry should have registered features."""
        features = list_features()
        assert len(features) > 0
        assert "returns" in features
        assert "momentum" in features

    def test_get_feature_returns_info(self) -> None:
        """get_feature should return FeatureInfo."""
        info = get_feature("returns")
        assert info.name == "returns"
        assert callable(info.func)

    def test_get_unknown_feature_raises(self) -> None:
        """Getting unknown feature should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_feature("unknown_feature_xyz")


class TestPipeline:
    """Tests for build_features pipeline."""

    def test_build_features_creates_columns(self, synthetic_df: pd.DataFrame) -> None:
        """Pipeline should create expected columns."""
        specs = [
            FeatureSpec(name="returns", params={"price_col": "close", "periods": 1}),
            FeatureSpec(name="returns", params={"price_col": "close", "periods": 5}),
        ]

        X, report = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        assert "close_return_1" in X.columns
        assert "close_return_5" in X.columns
        assert report.total_columns == 2

    def test_build_features_report_stats(self, synthetic_df: pd.DataFrame) -> None:
        """Report should contain valid stats."""
        specs = [FeatureSpec(name="returns", params={"price_col": "close"})]

        _, report = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        assert len(report.results) == 1
        result = report.results[0]
        assert "close_return_1" in result.null_rates
        assert result.null_rates["close_return_1"] < 1.0  # Not all NaN
        assert result.stats["close_return_1"]["mean"] is not None

    def test_quick_features(self, synthetic_df: pd.DataFrame) -> None:
        """Quick features should work with defaults."""
        X, report = quick_features(
            synthetic_df,
            price_col="close",
            volume_col="volume",
            datetime_col="datetime",
            group_col="symbol",
            windows=[5, 10],
        )

        # Should have returns, momentum, volatility, volume features
        assert X.shape[1] > 5
        assert report.total_columns == X.shape[1]


class TestTimeSeriesFeatures:
    """Tests for time-series feature families."""

    def test_price_features(self, synthetic_df: pd.DataFrame) -> None:
        """Price features should compute correctly."""
        specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
            FeatureSpec(name="log_returns", params={"price_col": "close"}),
            FeatureSpec(name="momentum", params={"price_col": "close", "window": 10}),
            FeatureSpec(name="rolling_volatility", params={"price_col": "close", "window": 10}),
        ]

        X, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        assert "close_return_1" in X.columns
        assert "close_log_return_1" in X.columns
        assert "close_momentum_10" in X.columns
        assert "close_vol_10" in X.columns

    def test_volume_features(self, synthetic_df: pd.DataFrame) -> None:
        """Volume features should compute correctly."""
        specs = [
            FeatureSpec(name="volume_shock", params={"volume_col": "volume", "window": 10}),
            FeatureSpec(name="rolling_volume", params={"volume_col": "volume", "window": 10}),
        ]

        X, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        assert "volume_shock_10" in X.columns
        assert "volume_ma_10" in X.columns

    def test_flow_features(self, synthetic_df: pd.DataFrame) -> None:
        """Flow features should compute correctly."""
        specs = [
            FeatureSpec(
                name="flow_imbalance",
                params={"buy_col": "buy_volume", "sell_col": "sell_volume", "window": 10},
            ),
            FeatureSpec(
                name="signed_volume",
                params={"volume_col": "volume", "price_col": "close", "window": 10},
            ),
        ]

        X, _ = build_features(
            synthetic_df, specs, datetime_col="datetime", group_col="symbol"
        )

        assert "flow_imbalance_10" in X.columns
        assert "signed_volume_10" in X.columns


class TestCrossSectionalFeatures:
    """Tests for cross-sectional features."""

    def test_cross_sectional_rank(self, synthetic_df: pd.DataFrame) -> None:
        """Cross-sectional rank should rank within datetime."""
        specs = [
            FeatureSpec(
                name="cross_sectional_rank",
                params={"value_col": "close", "datetime_col": "datetime"},
            )
        ]

        X, _ = build_features(synthetic_df, specs, datetime_col="datetime")

        assert "close_cs_rank" in X.columns
        # Ranks should be between 0 and 1 (percentile)
        valid = X["close_cs_rank"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1


class TestCalendarFeatures:
    """Tests for calendar features."""

    def test_day_of_week(self, synthetic_df: pd.DataFrame) -> None:
        """Day of week should be 0-6."""
        specs = [FeatureSpec(name="day_of_week", params={"datetime_col": "datetime"})]

        X, _ = build_features(synthetic_df, specs, datetime_col="datetime")

        assert "day_of_week" in X.columns
        assert X["day_of_week"].min() >= 0
        assert X["day_of_week"].max() <= 6

    def test_time_features_all(self, synthetic_df: pd.DataFrame) -> None:
        """time_features should create multiple columns."""
        specs = [FeatureSpec(name="time_features", params={"datetime_col": "datetime"})]

        X, _ = build_features(synthetic_df, specs, datetime_col="datetime")

        assert "day_of_week" in X.columns
        assert "month" in X.columns
        assert "quarter" in X.columns
