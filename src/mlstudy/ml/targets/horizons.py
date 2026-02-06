"""Multi-horizon target generation for panel data.

Generate prediction targets at multiple future horizons for
time-series panel data (e.g., bond hedged yield changes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def make_forward_change_target(
    df: pd.DataFrame,
    value_col: str,
    horizon_steps: int,
    datetime_col: str = "datetime",
    group_col: str | None = None,
    target_col: str | None = None,
) -> pd.Series:
    """Create forward change target for a single horizon.

    Computes target = value(t + horizon) - value(t).

    Args:
        df: DataFrame with datetime, value, and optional group columns.
        value_col: Column containing values (e.g., hedged_yield).
        horizon_steps: Number of steps ahead to predict.
        datetime_col: Datetime column name.
        group_col: Group column for panel data (e.g., bond_id).
        target_col: Output column name. Defaults to f"target_{horizon_steps}".

    Returns:
        Series with forward change values. NaN at end of each group.

    Example:
        >>> target = make_forward_change_target(
        ...     df, value_col="hedged_yield", horizon_steps=5, group_col="bond_id"
        ... )
    """
    if target_col is None:
        target_col = f"target_{horizon_steps}"

    df = df.sort_values([group_col, datetime_col] if group_col else [datetime_col])

    if group_col is not None:
        future_val = df.groupby(group_col)[value_col].shift(-horizon_steps)
    else:
        future_val = df[value_col].shift(-horizon_steps)

    target = future_val - df[value_col]
    target.name = target_col

    return target


def make_multi_horizon_targets(
    df: pd.DataFrame,
    value_col: str,
    horizons: list[int],
    datetime_col: str = "datetime",
    group_col: str | None = None,
    method: str = "change",
    prefix: str = "target",
) -> pd.DataFrame:
    """Generate targets at multiple future horizons.

    For each horizon h, creates:
    - method="change": target_h = value(t+h) - value(t)
    - method="return": target_h = value(t+h) / value(t) - 1
    - method="log_return": target_h = log(value(t+h) / value(t))

    Args:
        df: DataFrame with datetime, value, and optional group columns.
        value_col: Column containing values to predict (e.g., hedged_yield).
        horizons: List of horizons in periods (e.g., [1, 5, 10, 20] for days).
        datetime_col: Datetime column name.
        group_col: Column for panel groups (e.g., bond_id).
        method: "change", "return", or "log_return".
        prefix: Prefix for target column names.

    Returns:
        DataFrame with original columns plus target columns: {prefix}_{h}.

    Example:
        >>> df_targets = make_multi_horizon_targets(
        ...     df, value_col="hedged_yield", horizons=[1, 5, 10, 20],
        ...     group_col="bond_id", method="change"
        ... )
        >>> # Returns df with columns: target_1, target_5, target_10, target_20
    """
    result = df.copy()

    # Sort for proper shifting
    sort_cols = [group_col, datetime_col] if group_col else [datetime_col]
    result = result.sort_values(sort_cols)

    for h in horizons:
        col_name = f"{prefix}_{h}"

        if group_col is not None:
            future_val = result.groupby(group_col)[value_col].shift(-h)
        else:
            future_val = result[value_col].shift(-h)

        current_val = result[value_col]

        if method == "change":
            result[col_name] = future_val - current_val
        elif method == "return":
            result[col_name] = (future_val / current_val) - 1
        elif method == "log_return":
            result[col_name] = np.log(future_val / current_val)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'change', 'return', or 'log_return'")

    return result


def get_horizon_columns(df: pd.DataFrame, prefix: str = "target") -> list[str]:
    """Get target column names for all horizons.

    Args:
        df: DataFrame with multi-horizon targets.
        prefix: Target column prefix.

    Returns:
        List of target column names sorted by horizon.
    """
    cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def extract_horizon(col_name: str) -> int:
    """Extract horizon from target column name.

    Args:
        col_name: Column name like "target_5".

    Returns:
        Horizon as integer.
    """
    return int(col_name.split("_")[-1])


class MultiHorizonTargetGenerator:
    """Generate and manage multi-horizon targets for training.

    Provides utilities for creating train/test splits that respect
    the multi-horizon structure (no leakage from future horizons).

    Example:
        >>> generator = MultiHorizonTargetGenerator(
        ...     horizons=[1, 5, 10, 20],
        ...     value_col="hedged_yield",
        ...     group_col="bond_id",
        ... )
        >>> df_with_targets = generator.fit_transform(df)
        >>> X, y_dict = generator.get_X_y(df_with_targets, feature_cols)
    """

    def __init__(
        self,
        horizons: list[int],
        value_col: str,
        datetime_col: str = "datetime",
        group_col: str | None = None,
        method: str = "change",
        prefix: str = "target",
    ):
        """Initialize multi-horizon target generator.

        Args:
            horizons: List of prediction horizons.
            value_col: Column with values to predict.
            datetime_col: Datetime column name.
            group_col: Panel group column.
            method: Target calculation method.
            prefix: Target column prefix.
        """
        self.horizons = sorted(horizons)
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.group_col = group_col
        self.method = method
        self.prefix = prefix
        self.target_cols = [f"{prefix}_{h}" for h in self.horizons]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate targets for all horizons.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with target columns added.
        """
        return make_multi_horizon_targets(
            df,
            value_col=self.value_col,
            horizons=self.horizons,
            datetime_col=self.datetime_col,
            group_col=self.group_col,
            method=self.method,
            prefix=self.prefix,
        )

    def get_X_y(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        dropna: bool = True,
    ) -> tuple[NDArray, dict[int, NDArray]]:
        """Extract features and targets for all horizons.

        Args:
            df: DataFrame with features and targets.
            feature_cols: Feature column names.
            dropna: Drop rows with any NaN in features or targets.

        Returns:
            Tuple of (X array, dict mapping horizon to y array).
        """
        if dropna:
            mask = df[feature_cols + self.target_cols].notna().all(axis=1)
            df = df[mask]

        X = df[feature_cols].values
        y_dict = {h: df[f"{self.prefix}_{h}"].values for h in self.horizons}

        return X, y_dict

    def get_X_y_stacked(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        dropna: bool = True,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Get stacked X, y with horizon indicator.

        Useful for training a single model with horizon as a feature.

        Args:
            df: DataFrame with features and targets.
            feature_cols: Feature column names.
            dropna: Drop rows with NaN.

        Returns:
            Tuple of (X_stacked, y_stacked, horizons_stacked).
            X_stacked has shape (n_samples * n_horizons, n_features).
        """
        X, y_dict = self.get_X_y(df, feature_cols, dropna=dropna)

        n_samples = X.shape[0]
        n_horizons = len(self.horizons)

        # Stack: repeat X for each horizon
        X_stacked = np.tile(X, (n_horizons, 1))

        # Stack targets
        y_stacked = np.concatenate([y_dict[h] for h in self.horizons])

        # Horizon indicator
        horizons_stacked = np.repeat(self.horizons, n_samples)

        return X_stacked, y_stacked, horizons_stacked

    @property
    def max_horizon(self) -> int:
        """Maximum prediction horizon."""
        return max(self.horizons)

    @property
    def n_horizons(self) -> int:
        """Number of horizons."""
        return len(self.horizons)
