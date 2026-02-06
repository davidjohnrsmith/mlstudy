"""Dataset abstraction for ML training pipelines.

Provides a structured wrapper around pandas DataFrames for ML workflows,
with time-series aware operations for trading research.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class MLDataFrameDataset:
    """Dataset wrapper for ML training with time-series support.

    Wraps a pandas DataFrame with metadata about columns used for ML training.
    Provides utilities for time-aware data handling common in trading research.

    Attributes:
        df: The underlying pandas DataFrame.
        datetime_col: Column name containing datetime/timestamp.
        target_col: Column name for the prediction target (y).
        feature_cols: List of column names to use as features (X).
        group_col: Optional column name for grouping (e.g., asset symbol).

    Example:
        >>> dataset = MLDataFrameDataset(
        ...     df=prices_df,
        ...     datetime_col="date",
        ...     target_col="forward_return",
        ...     feature_cols=["momentum", "volatility", "volume_ratio"],
        ...     group_col="symbol",
        ... )
        >>> dataset.ensure_sorted_by_time()
        >>> X, y = dataset.get_X_y()
    """

    df: pd.DataFrame
    datetime_col: str
    target_col: str
    feature_cols: list[str]
    group_col: str | None = None
    _sorted: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate column names exist in DataFrame."""
        self._validate_columns()

    def _validate_columns(self) -> None:
        """Check that all specified columns exist in the DataFrame."""
        missing = []

        if self.datetime_col not in self.df.columns:
            missing.append(f"datetime_col: {self.datetime_col}")

        if self.target_col not in self.df.columns:
            missing.append(f"target_col: {self.target_col}")

        for col in self.feature_cols:
            if col not in self.df.columns:
                missing.append(f"feature_col: {col}")

        if self.group_col and self.group_col not in self.df.columns:
            missing.append(f"group_col: {self.group_col}")

        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    def ensure_sorted_by_time(self) -> MLDataFrameDataset:
        """Sort DataFrame by datetime column (and group if present).

        Sorts in-place and marks the dataset as sorted. If a group column
        is specified, sorts by [group_col, datetime_col] to maintain
        per-group time ordering.

        Returns:
            Self for method chaining.
        """
        if self._sorted:
            return self

        sort_cols = [self.datetime_col]
        if self.group_col:
            sort_cols = [self.group_col, self.datetime_col]

        self.df = self.df.sort_values(sort_cols).reset_index(drop=True)
        self._sorted = True
        return self

    def get_X_y(
        self,
        dropna: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Extract feature matrix X and target vector y.

        Args:
            dropna: If True, drop rows with any NaN in features or target.

        Returns:
            Tuple of (X, y) as numpy arrays.
            X has shape (n_samples, n_features).
            y has shape (n_samples,).
        """
        subset = self.df[self.feature_cols + [self.target_col]]

        if dropna:
            subset = subset.dropna()

        X = subset[self.feature_cols].values.astype(np.float64)
        y = subset[self.target_col].values.astype(np.float64)

        return X, y

    def get_X_y_df(
        self,
        dropna: bool = True,
        include_meta: bool = False,
    ) -> pd.DataFrame:
        """Extract features and target as a DataFrame.

        Args:
            dropna: If True, drop rows with any NaN in features or target.
            include_meta: If True, include datetime and group columns.

        Returns:
            DataFrame with feature columns and target column.
        """
        cols = list(self.feature_cols) + [self.target_col]

        if include_meta:
            cols = [self.datetime_col] + cols
            if self.group_col:
                cols.insert(1, self.group_col)

        subset = self.df[cols]

        if dropna:
            subset = subset.dropna()

        return subset

    @property
    def n_samples(self) -> int:
        """Number of samples (rows) in the dataset."""
        return len(self.df)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_cols)

    @property
    def groups(self) -> list | None:
        """Unique groups if group_col is set, else None."""
        if self.group_col is None:
            return None
        return self.df[self.group_col].unique().tolist()

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Min and max datetime in the dataset."""
        dates = pd.to_datetime(self.df[self.datetime_col])
        return dates.min(), dates.max()

    def filter_dates(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> MLDataFrameDataset:
        """Filter dataset to a date range.

        Args:
            start: Start date (inclusive). If None, no lower bound.
            end: End date (inclusive). If None, no upper bound.

        Returns:
            New MLDataFrameDataset with filtered data.
        """
        df = self.df.copy()
        dates = pd.to_datetime(df[self.datetime_col])

        if start is not None:
            df = df[dates >= pd.to_datetime(start)]
            dates = pd.to_datetime(df[self.datetime_col])

        if end is not None:
            df = df[dates <= pd.to_datetime(end)]

        return MLDataFrameDataset(
            df=df,
            datetime_col=self.datetime_col,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            group_col=self.group_col,
        )

    def filter_groups(self, groups: list) -> MLDataFrameDataset:
        """Filter dataset to specific groups.

        Args:
            groups: List of group values to include.

        Returns:
            New MLDataFrameDataset with filtered data.

        Raises:
            ValueError: If group_col is not set.
        """
        if self.group_col is None:
            raise ValueError("Cannot filter by groups: group_col is not set")

        df = self.df[self.df[self.group_col].isin(groups)].copy()

        return MLDataFrameDataset(
            df=df,
            datetime_col=self.datetime_col,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            group_col=self.group_col,
        )

    def summary(self) -> dict:
        """Return summary statistics about the dataset."""
        X, y = self.get_X_y(dropna=True)
        return {
            "n_samples": self.n_samples,
            "n_samples_valid": len(y),
            "n_features": self.n_features,
            "n_groups": len(self.groups) if self.groups else 1,
            "date_range": self.date_range,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "feature_cols": self.feature_cols,
        }
