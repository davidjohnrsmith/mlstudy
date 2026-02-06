"""Tests for make_supervised_dataset."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.core.features import FeatureSpec
from mlstudy.ml.pipeline.make_dataset import TargetSpec, make_supervised_dataset


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Create sample market DataFrame."""
    np.random.seed(42)
    n = 100

    # Generate price-like data
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "asset": "TEST",
            "close": prices,
            "volume": np.random.uniform(1000, 5000, n),
            "flow_imbalance": np.random.uniform(-1, 1, n),
        }
    )


@pytest.fixture
def multi_asset_df() -> pd.DataFrame:
    """Create multi-asset DataFrame."""
    np.random.seed(42)
    n = 50

    dfs = []
    for asset in ["A", "B"]:
        returns = np.random.randn(n) * 0.02
        prices = 100 * np.cumprod(1 + returns)
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "asset": asset,
                "close": prices,
                "volume": np.random.uniform(1000, 5000, n),
            }
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


class TestMakeSupervisedDataset:
    """Tests for make_supervised_dataset."""

    def test_returns_correct_types(self, sample_market_df: pd.DataFrame) -> None:
        """Should return DataFrames and DatasetMeta."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
        )

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(meta, pd.DataFrame)

    def test_x_y_alignment(self, sample_market_df: pd.DataFrame) -> None:
        """X and y should have same number of rows."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
            FeatureSpec(name="momentum", params={"price_col": "close", "window": 5}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=True,
        )

        assert len(X) == len(y)
        assert len(X) == len(meta)

    def test_dropna_removes_nulls(self, sample_market_df: pd.DataFrame) -> None:
        """With dropna=True, should have no NaN values."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
            FeatureSpec(name="momentum", params={"price_col": "close", "window": 10}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=True,
        )

        assert not X.isna().any().any()
        assert not y.isna().any().any()

    def test_dropna_false_keeps_nulls(self, sample_market_df: pd.DataFrame) -> None:
        """With dropna=False, should keep NaN values."""
        feature_specs = [
            FeatureSpec(name="momentum", params={"price_col": "close", "window": 10}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=False,
        )

        # Should have same length as input
        assert len(X) == len(sample_market_df)

    def test_meta_has_datetime(self, sample_market_df: pd.DataFrame) -> None:
        """Meta should contain datetime column."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=True,
        )

        assert "datetime" in meta.columns

    def test_meta_has_group_col(self, multi_asset_df: pd.DataFrame) -> None:
        """Meta should contain group column when specified."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            multi_asset_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            group_col="asset",
            dropna=True,
        )

        assert "asset" in meta.columns

    def test_dataset_meta_info(self, sample_market_df: pd.DataFrame) -> None:
        """DatasetMeta should contain correct information."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
            FeatureSpec(name="momentum", params={"price_col": "close", "window": 5}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=True,
        )

        assert info.n_samples == len(X)
        assert info.n_features == len(X.columns)
        assert len(info.feature_cols) == info.n_features
        assert info.null_dropped > 0  # Should have dropped some rows

    def test_no_lookahead_in_features(self, sample_market_df: pd.DataFrame) -> None:
        """Features at time t should not use data after t."""
        # This is a sanity check - features use shift=1 by default
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close", "periods": 1}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=True,
        )

        # The return feature at row i should be based on price change from i-2 to i-1
        # (due to shift=1). It should NOT be the same as the target (which is i to i+1)
        # So feature and target should have low/no correlation at the individual row level

        # Simple check: features and targets should not be identical
        assert not np.allclose(X.iloc[:, 0].values, y.iloc[:, 0].values)

    def test_multi_asset_processing(self, multi_asset_df: pd.DataFrame) -> None:
        """Should process multiple assets correctly."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
        ]
        target_spec = TargetSpec(horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            multi_asset_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            group_col="asset",
            dropna=True,
        )

        # Should have data from both assets
        assert meta["asset"].nunique() == 2

    def test_forward_direction_target(self, sample_market_df: pd.DataFrame) -> None:
        """Should work with forward_direction target type."""
        feature_specs = [
            FeatureSpec(name="returns", params={"price_col": "close"}),
        ]
        target_spec = TargetSpec(target_type="forward_direction", horizon_steps=1)

        X, y, meta, info = make_supervised_dataset(
            sample_market_df,
            feature_specs,
            target_spec,
            datetime_col="datetime",
            dropna=True,
        )

        # Target should be binary
        assert set(y.iloc[:, 0].unique()).issubset({0, 1})
