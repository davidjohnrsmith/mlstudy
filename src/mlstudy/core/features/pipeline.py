"""Feature pipeline for building features from specifications."""

from __future__ import annotations

import pandas as pd

from mlstudy.core.features.base import FeatureReport, FeatureResult, FeatureSpec
from mlstudy.core.features.registry import get_feature


def build_features(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    datetime_col: str,
    group_col: str | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Build features from a list of specifications.

    Args:
        df: Input DataFrame.
        specs: List of FeatureSpec defining features to compute.
        datetime_col: Datetime column name for sorting.
        group_col: Optional group column name.
        validate: Whether to validate required columns exist.

    Returns:
        Tuple of (feature_df, report):
        - feature_df: DataFrame containing only the computed feature columns.
        - report: FeatureReport with metadata about each feature.

    Raises:
        KeyError: If a feature name is not registered.
        ValueError: If required columns are missing (when validate=True).
    """
    # Sort by datetime (and group if provided)
    sort_cols = [datetime_col]
    if group_col:
        sort_cols = [group_col, datetime_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    results: list[FeatureResult] = []
    feature_dfs: list[pd.DataFrame] = []

    for spec in specs:
        # Get feature info
        feature_info = get_feature(spec.name)

        # Resolve required columns with params
        if validate:
            _validate_required_cols(df, feature_info.required_cols, spec.params)

        # Build kwargs for feature function
        kwargs = spec.params.copy()
        if group_col and "group_col" not in kwargs:
            kwargs["group_col"] = group_col

        # Call feature function
        feature_df = feature_info.func(df, **kwargs)

        # Get output column names
        if spec.output_cols and len(spec.output_cols) == len(feature_df.columns):
            feature_df.columns = spec.output_cols
        output_cols = list(feature_df.columns)

        # Compute stats for report
        null_rates = {}
        stats = {}
        for col in output_cols:
            null_rates[col] = float(feature_df[col].isna().mean())
            valid = feature_df[col].dropna()
            if len(valid) > 0:
                stats[col] = {
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                }
            else:
                stats[col] = {"mean": None, "std": None, "min": None, "max": None}

        results.append(
            FeatureResult(
                spec=spec,
                columns=output_cols,
                null_rates=null_rates,
                stats=stats,
            )
        )

        feature_dfs.append(feature_df)

    # Combine all feature DataFrames
    X = pd.concat(feature_dfs, axis=1) if feature_dfs else pd.DataFrame(index=df.index)

    # Build report
    report = FeatureReport(
        results=results,
        total_columns=len(X.columns),
        total_rows=len(X),
        datetime_col=datetime_col,
        group_col=group_col,
    )

    return X, report


def _validate_required_cols(
    df: pd.DataFrame,
    required_cols: list[str],
    params: dict,
) -> None:
    """Validate that required columns exist in DataFrame.

    Args:
        df: Input DataFrame.
        required_cols: List of required column patterns (may contain {param}).
        params: Parameters to resolve column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = []
    for col_pattern in required_cols:
        # Resolve pattern with params
        if "{" in col_pattern:
            # Extract param name
            param_name = col_pattern.strip("{}")
            col_name = params.get(param_name)
            if col_name is None:
                # Use default if not specified
                continue
        else:
            col_name = col_pattern

        if col_name not in df.columns:
            missing.append(col_name)

    if missing:
        raise ValueError(f"Required columns not found: {missing}")


def quick_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str | None = None,
    datetime_col: str = "datetime",
    group_col: str | None = None,
    windows: list[int] | None = None,
) -> tuple[pd.DataFrame, FeatureReport]:
    """Quick feature generation with sensible defaults.

    Generates a standard set of price and volume features.

    Args:
        df: Input DataFrame.
        price_col: Price column name.
        volume_col: Optional volume column name.
        datetime_col: Datetime column name.
        group_col: Optional group column name.
        windows: List of rolling windows to use. Defaults to [5, 10, 20].

    Returns:
        Tuple of (feature_df, report).
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
            FeatureSpec(name="rolling_volatility", params={"price_col": price_col, "window": w})
        )

    # Volume features (if available)
    if volume_col and volume_col in df.columns:
        for w in windows:
            specs.append(
                FeatureSpec(name="volume_shock", params={"volume_col": volume_col, "window": w})
            )
            specs.append(
                FeatureSpec(name="rolling_volume", params={"volume_col": volume_col, "window": w})
            )

    return build_features(df, specs, datetime_col=datetime_col, group_col=group_col)
