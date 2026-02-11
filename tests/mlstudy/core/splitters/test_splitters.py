"""Tests for time-series splitters."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.core.splitters.time import time_train_val_test_split
from mlstudy.core.splitters.walk_forward import walk_forward_splits


@pytest.fixture
def time_series_df() -> pd.DataFrame:
    """Create sample time series DataFrame."""
    np.random.seed(42)
    n = 365  # One year of daily data
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "symbol": np.random.choice(["A", "B", "C"], n),
            "feature": np.random.randn(n),
            "target": np.random.randn(n),
        }
    )


class TestTimeSplit:
    """Tests for time_train_val_test_split."""

    def test_basic_split(self, time_series_df: pd.DataFrame) -> None:
        """Should split data into non-overlapping sets."""
        train, val, test = time_train_val_test_split(
            df=time_series_df,
            datetime_col="date",
            train_end="2024-06-30",
            val_end="2024-09-30",
            test_end="2024-12-31",
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(time_series_df)

    def test_no_overlap(self, time_series_df: pd.DataFrame) -> None:
        """Train/val/test sets should not overlap."""
        train, val, test = time_train_val_test_split(
            df=time_series_df,
            datetime_col="date",
            train_end="2024-06-30",
            val_end="2024-09-30",
            test_end="2024-12-31",
        )

        train_dates = set(train["date"])
        val_dates = set(val["date"])
        test_dates = set(test["date"])

        assert len(train_dates & val_dates) == 0
        assert len(train_dates & test_dates) == 0
        assert len(val_dates & test_dates) == 0

    def test_time_ordering(self, time_series_df: pd.DataFrame) -> None:
        """Train dates should be before val, val before test."""
        train, val, test = time_train_val_test_split(
            df=time_series_df,
            datetime_col="date",
            train_end="2024-06-30",
            val_end="2024-09-30",
            test_end="2024-12-31",
        )

        assert train["date"].max() <= pd.Timestamp("2024-06-30")
        assert val["date"].min() > pd.Timestamp("2024-06-30")
        assert val["date"].max() <= pd.Timestamp("2024-09-30")
        assert test["date"].min() > pd.Timestamp("2024-09-30")

    def test_invalid_date_order_raises(self, time_series_df: pd.DataFrame) -> None:
        """Should raise error if train_end >= val_end."""
        with pytest.raises(ValueError, match="train_end.*before.*val_end"):
            time_train_val_test_split(
                df=time_series_df,
                datetime_col="date",
                train_end="2024-09-30",
                val_end="2024-06-30",  # Before train_end
                test_end="2024-12-31",
            )

    def test_min_count_filtering(self, time_series_df: pd.DataFrame) -> None:
        """Should filter groups with too few samples."""
        train, val, test = time_train_val_test_split(
            df=time_series_df,
            datetime_col="date",
            train_end="2024-06-30",
            val_end="2024-09-30",
            test_end="2024-12-31",
            group_col="symbol",
            min_count=50,
        )

        # All remaining groups should have >= 50 samples in each split
        for split in [train, val, test]:
            if len(split) > 0:
                counts = split["symbol"].value_counts()
                assert all(counts >= 50)


class TestWalkForward:
    """Tests for walk_forward_splits."""

    def test_generates_folds(self, time_series_df: pd.DataFrame) -> None:
        """Should generate at least one fold."""
        folds = list(
            walk_forward_splits(
                df=time_series_df,
                datetime_col="date",
                train_days=90,
                val_days=30,
                test_days=30,
                step_days=30,
            )
        )

        assert len(folds) > 0

    def test_fold_has_correct_attributes(self, time_series_df: pd.DataFrame) -> None:
        """Each fold should have required attributes."""
        fold = next(
            walk_forward_splits(
                df=time_series_df,
                datetime_col="date",
                train_days=90,
                val_days=30,
                test_days=30,
                step_days=30,
            )
        )

        assert hasattr(fold, "fold_id")
        assert hasattr(fold, "train_df")
        assert hasattr(fold, "val_df")
        assert hasattr(fold, "test_df")
        assert len(fold.train_df) > 0
        assert len(fold.val_df) > 0
        assert len(fold.test_df) > 0

    def test_no_overlap_within_fold(self, time_series_df: pd.DataFrame) -> None:
        """Train/val/test within a fold should not overlap."""
        for fold in walk_forward_splits(
            df=time_series_df,
            datetime_col="date",
            train_days=90,
            val_days=30,
            test_days=30,
            step_days=30,
        ):
            train_dates = set(fold.train_df["date"])
            val_dates = set(fold.val_df["date"])
            test_dates = set(fold.test_df["date"])

            assert len(train_dates & val_dates) == 0
            assert len(train_dates & test_dates) == 0
            assert len(val_dates & test_dates) == 0

    def test_time_ordering_within_fold(self, time_series_df: pd.DataFrame) -> None:
        """Within each fold, train < val < test."""
        for fold in walk_forward_splits(
            df=time_series_df,
            datetime_col="date",
            train_days=90,
            val_days=30,
            test_days=30,
            step_days=30,
        ):
            assert fold.train_df["date"].max() < fold.val_df["date"].min()
            assert fold.val_df["date"].max() < fold.test_df["date"].min()

    def test_fold_ids_sequential(self, time_series_df: pd.DataFrame) -> None:
        """Fold IDs should be sequential starting from 0."""
        folds = list(
            walk_forward_splits(
                df=time_series_df,
                datetime_col="date",
                train_days=90,
                val_days=30,
                test_days=30,
                step_days=30,
            )
        )

        fold_ids = [f.fold_id for f in folds]
        assert fold_ids == list(range(len(folds)))

    def test_expanding_vs_rolling(self, time_series_df: pd.DataFrame) -> None:
        """Expanding window should have growing train size."""
        expanding_folds = list(
            walk_forward_splits(
                df=time_series_df,
                datetime_col="date",
                train_days=60,
                val_days=30,
                test_days=30,
                step_days=30,
                expanding=True,
            )
        )

        rolling_folds = list(
            walk_forward_splits(
                df=time_series_df,
                datetime_col="date",
                train_days=60,
                val_days=30,
                test_days=30,
                step_days=30,
                expanding=False,
            )
        )

        if len(expanding_folds) >= 2:
            # Expanding: train size should grow
            assert len(expanding_folds[1].train_df) > len(expanding_folds[0].train_df)

        if len(rolling_folds) >= 2:
            # Rolling: train size should stay roughly the same
            size_diff = abs(len(rolling_folds[1].train_df) - len(rolling_folds[0].train_df))
            # Allow some variance due to data distribution
            assert size_diff < len(rolling_folds[0].train_df) * 0.2

    def test_no_test_overlap_between_folds(self, time_series_df: pd.DataFrame) -> None:
        """Test sets from different folds may overlap (stepped), but should advance."""
        folds = list(
            walk_forward_splits(
                df=time_series_df,
                datetime_col="date",
                train_days=90,
                val_days=30,
                test_days=30,
                step_days=30,
            )
        )

        if len(folds) >= 2:
            # Test periods should advance
            assert folds[1].test_start > folds[0].test_start
