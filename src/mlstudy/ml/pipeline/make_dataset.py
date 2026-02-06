"""Build supervised datasets from raw data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlstudy.core.features import FeatureSpec, build_features
from mlstudy.ml.targets.returns import make_forward_return_target


@dataclass
class TargetSpec:
    """Specification for target variable generation.

    Attributes:
        target_type: Type of target ("forward_return", "forward_direction").
        price_col: Price column name.
        horizon_steps: Forward-looking horizon.
        log_return: Whether to use log returns (for forward_return).
    """

    target_type: str = "forward_return"
    price_col: str = "close"
    horizon_steps: int = 1
    log_return: bool = True


@dataclass
class DatasetMeta:
    """Metadata about the created dataset.

    Attributes:
        n_samples: Number of samples after dropna.
        n_features: Number of feature columns.
        feature_cols: List of feature column names.
        target_col: Target column name.
        datetime_col: Datetime column name.
        group_col: Group column name (if any).
        null_dropped: Number of rows dropped due to NaN.
    """

    n_samples: int
    n_features: int
    feature_cols: list[str]
    target_col: str
    datetime_col: str
    group_col: str | None
    null_dropped: int


def make_supervised_dataset(
    df: pd.DataFrame,
    feature_specs: list[FeatureSpec],
    target_spec: TargetSpec,
    datetime_col: str = "datetime",
    group_col: str | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DatasetMeta]:
    """Build a supervised learning dataset from raw data.

    Constructs features and target, aligns them, and optionally drops
    rows with missing values.

    IMPORTANT: Features are computed with leak-safe shifting (shift=1 by default),
    so features at time t only use data up to t-1. The target at time t is the
    forward return from t to t+horizon. This ensures no lookahead bias.

    Args:
        df: Input DataFrame with raw data.
        feature_specs: List of FeatureSpec defining features to compute.
        target_spec: TargetSpec defining target variable.
        datetime_col: Name of datetime column.
        group_col: Name of group column (e.g., asset symbol).
        dropna: Whether to drop rows with any NaN in features or target.

    Returns:
        Tuple of (X, y, meta_df, dataset_meta):
        - X: DataFrame with feature columns only.
        - y: DataFrame with target column only.
        - meta_df: DataFrame with datetime and group columns for each row.
        - dataset_meta: DatasetMeta with summary information.

    Example:
        >>> feature_specs = [
        ...     FeatureSpec(name="returns", params={"price_col": "close"}),
        ...     FeatureSpec(name="momentum", params={"price_col": "close", "window": 10}),
        ... ]
        >>> target_spec = TargetSpec(horizon_steps=1)
        >>> X, y, meta, info = make_supervised_dataset(df, feature_specs, target_spec)
    """
    # Sort data
    sort_cols = [datetime_col]
    if group_col:
        sort_cols = [group_col, datetime_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Build features
    X, feature_report = build_features(
        df,
        feature_specs,
        datetime_col=datetime_col,
        group_col=group_col,
    )
    feature_cols = list(X.columns)

    # Build target
    if target_spec.target_type == "forward_return":
        target_df = make_forward_return_target(
            df,
            price_col=target_spec.price_col,
            datetime_col=datetime_col,
            group_col=group_col,
            horizon_steps=target_spec.horizon_steps,
            log_return=target_spec.log_return,
        )
    elif target_spec.target_type == "forward_direction":
        from mlstudy.ml.targets.returns import make_forward_direction_target

        target_df = make_forward_direction_target(
            df,
            price_col=target_spec.price_col,
            datetime_col=datetime_col,
            group_col=group_col,
            horizon_steps=target_spec.horizon_steps,
        )
    else:
        raise ValueError(f"Unknown target_type: {target_spec.target_type}")

    target_col = target_df.columns[0]
    y = target_df

    # Build meta DataFrame
    meta_cols = [datetime_col]
    if group_col:
        meta_cols.append(group_col)
    meta_df = df[meta_cols].copy()

    # Combine for dropna
    combined = pd.concat([meta_df, X, y], axis=1)

    if dropna:
        combined_clean = combined.dropna()
        null_dropped = len(combined) - len(combined_clean)
        combined = combined_clean
    else:
        null_dropped = 0

    # Split back
    X = combined[feature_cols].reset_index(drop=True)
    y = combined[[target_col]].reset_index(drop=True)
    meta_df = combined[meta_cols].reset_index(drop=True)

    dataset_meta = DatasetMeta(
        n_samples=len(X),
        n_features=len(feature_cols),
        feature_cols=feature_cols,
        target_col=target_col,
        datetime_col=datetime_col,
        group_col=group_col,
        null_dropped=null_dropped,
    )

    return X, y, meta_df, dataset_meta


def get_default_feature_specs(
    price_col: str = "close",
    volume_col: str | None = "volume",
    flow_col: str | None = "flow_imbalance",
    windows: list[int] | None = None,
) -> list[FeatureSpec]:
    """Get a default set of feature specifications for market data.

    Args:
        price_col: Price column name.
        volume_col: Volume column name (if available).
        flow_col: Flow imbalance column name (if available).
        windows: List of rolling windows.

    Returns:
        List of FeatureSpec objects.
    """
    if windows is None:
        windows = [5, 10, 20]

    specs = []

    # Price features
    specs.append(FeatureSpec(name="returns", params={"price_col": price_col, "periods": 1}))

    for w in windows:
        specs.append(
            FeatureSpec(name="momentum", params={"price_col": price_col, "window": w})
        )
        specs.append(
            FeatureSpec(
                name="rolling_volatility", params={"price_col": price_col, "window": w}
            )
        )

    # Volume features
    if volume_col:
        for w in windows:
            specs.append(
                FeatureSpec(
                    name="volume_shock", params={"volume_col": volume_col, "window": w}
                )
            )

    # Flow features - use EWMA on flow_imbalance
    # Note: We need to add these features to the registry
    # For now, we'll use lagged values of flow_imbalance as a simple feature

    return specs
