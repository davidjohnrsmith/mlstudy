"""Tests for MLDataFrameDataset."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.core.data.dataset import MLDataFrameDataset


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT"], n),
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "target": np.random.randn(n),
        }
    )


@pytest.fixture
def dataset(sample_df: pd.DataFrame) -> MLDataFrameDataset:
    """Create MLDataFrameDataset for testing."""
    return MLDataFrameDataset(
        df=sample_df,
        datetime_col="date",
        target_col="target",
        feature_cols=["feature_1", "feature_2"],
        group_col="symbol",
    )


class TestMLDataFrameDataset:
    """Tests for MLDataFrameDataset."""

    def test_init_validates_columns(self, sample_df: pd.DataFrame) -> None:
        """Should raise ValueError for missing columns."""
        with pytest.raises(ValueError, match="Columns not found"):
            MLDataFrameDataset(
                df=sample_df,
                datetime_col="nonexistent",
                target_col="target",
                feature_cols=["feature_1"],
            )

    def test_init_validates_feature_cols(self, sample_df: pd.DataFrame) -> None:
        """Should raise ValueError for missing feature columns."""
        with pytest.raises(ValueError, match="feature_col"):
            MLDataFrameDataset(
                df=sample_df,
                datetime_col="date",
                target_col="target",
                feature_cols=["feature_1", "missing_feature"],
            )

    def test_n_samples(self, dataset: MLDataFrameDataset) -> None:
        """Should return correct number of samples."""
        assert dataset.n_samples == 100

    def test_n_features(self, dataset: MLDataFrameDataset) -> None:
        """Should return correct number of features."""
        assert dataset.n_features == 2

    def test_groups(self, dataset: MLDataFrameDataset) -> None:
        """Should return unique groups."""
        groups = dataset.groups
        assert groups is not None
        assert set(groups) == {"AAPL", "GOOGL", "MSFT"}

    def test_groups_none_when_no_group_col(self, sample_df: pd.DataFrame) -> None:
        """Should return None when group_col not set."""
        ds = MLDataFrameDataset(
            df=sample_df,
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
        )
        assert ds.groups is None

    def test_date_range(self, dataset: MLDataFrameDataset) -> None:
        """Should return correct date range."""
        start, end = dataset.date_range
        assert start == pd.Timestamp("2024-01-01")
        assert end == pd.Timestamp("2024-04-09")

    def test_ensure_sorted_by_time(self, sample_df: pd.DataFrame) -> None:
        """Should sort DataFrame by datetime."""
        # Shuffle the dataframe
        shuffled = sample_df.sample(frac=1, random_state=123)
        ds = MLDataFrameDataset(
            df=shuffled,
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1"],
            group_col="symbol",
        )

        ds.ensure_sorted_by_time()

        # Check sorted by group then date
        for symbol in ds.groups:
            group_data = ds.df[ds.df["symbol"] == symbol]
            dates = group_data["date"].values
            assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_ensure_sorted_returns_self(self, dataset: MLDataFrameDataset) -> None:
        """Should return self for method chaining."""
        result = dataset.ensure_sorted_by_time()
        assert result is dataset

    def test_get_X_y_shapes(self, dataset: MLDataFrameDataset) -> None:
        """Should return correct shapes for X and y."""
        X, y = dataset.get_X_y()
        assert X.shape == (100, 2)
        assert y.shape == (100,)

    def test_get_X_y_dropna(self, sample_df: pd.DataFrame) -> None:
        """Should drop rows with NaN when dropna=True."""
        sample_df.loc[0, "feature_1"] = np.nan
        sample_df.loc[1, "target"] = np.nan

        ds = MLDataFrameDataset(
            df=sample_df,
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1", "feature_2"],
        )

        X, y = ds.get_X_y(dropna=True)
        assert len(y) == 98
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_get_X_y_df(self, dataset: MLDataFrameDataset) -> None:
        """Should return DataFrame with features and target."""
        result = dataset.get_X_y_df()
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 3

    def test_get_X_y_df_include_meta(self, dataset: MLDataFrameDataset) -> None:
        """Should include meta columns when requested."""
        result = dataset.get_X_y_df(include_meta=True)
        assert "date" in result.columns
        assert "symbol" in result.columns

    def test_filter_dates(self, dataset: MLDataFrameDataset) -> None:
        """Should filter to date range."""
        filtered = dataset.filter_dates(start="2024-02-01", end="2024-02-28")
        start, end = filtered.date_range
        assert start >= pd.Timestamp("2024-02-01")
        assert end <= pd.Timestamp("2024-02-28")

    def test_filter_dates_returns_new_dataset(
        self, dataset: MLDataFrameDataset
    ) -> None:
        """Should return new dataset, not modify original."""
        original_len = dataset.n_samples
        filtered = dataset.filter_dates(start="2024-02-01")
        assert dataset.n_samples == original_len
        assert filtered.n_samples < original_len

    def test_filter_groups(self, dataset: MLDataFrameDataset) -> None:
        """Should filter to specific groups."""
        filtered = dataset.filter_groups(["AAPL", "GOOGL"])
        assert set(filtered.groups) == {"AAPL", "GOOGL"}

    def test_filter_groups_raises_without_group_col(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Should raise error when group_col not set."""
        ds = MLDataFrameDataset(
            df=sample_df,
            datetime_col="date",
            target_col="target",
            feature_cols=["feature_1"],
        )
        with pytest.raises(ValueError, match="group_col is not set"):
            ds.filter_groups(["AAPL"])

    def test_summary(self, dataset: MLDataFrameDataset) -> None:
        """Should return summary dict with expected keys."""
        summary = dataset.summary()
        assert "n_samples" in summary
        assert "n_features" in summary
        assert "n_groups" in summary
        assert "date_range" in summary
        assert "target_mean" in summary
        assert "feature_cols" in summary
        assert summary["n_features"] == 2
        assert summary["n_groups"] == 3
