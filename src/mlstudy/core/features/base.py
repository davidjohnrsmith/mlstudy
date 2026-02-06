"""Base classes and dataclasses for feature engineering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class FeatureSpec:
    """Specification for a feature to be computed.

    Attributes:
        name: Name of the registered feature function.
        params: Parameters to pass to the feature function.
        output_cols: Optional list of output column names (auto-generated if None).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    output_cols: list[str] | None = None


@dataclass
class FeatureResult:
    """Result of computing a single feature.

    Attributes:
        spec: The FeatureSpec that was computed.
        columns: List of column names created.
        null_rates: Dict mapping column name to null rate (0-1).
        stats: Dict mapping column name to basic stats (mean, std, min, max).
    """

    spec: FeatureSpec
    columns: list[str]
    null_rates: dict[str, float]
    stats: dict[str, dict[str, float]]


@dataclass
class FeatureReport:
    """Report from build_features pipeline.

    Attributes:
        results: List of FeatureResult for each computed feature.
        total_columns: Total number of feature columns created.
        total_rows: Number of rows in the output.
        datetime_col: Name of datetime column used.
        group_col: Name of group column used (if any).
    """

    results: list[FeatureResult]
    total_columns: int
    total_rows: int
    datetime_col: str
    group_col: str | None = None

    def summary(self) -> pd.DataFrame:
        """Return summary DataFrame of all features."""
        rows = []
        for result in self.results:
            for col in result.columns:
                rows.append(
                    {
                        "feature_name": result.spec.name,
                        "column": col,
                        "null_rate": result.null_rates.get(col, 0),
                        "mean": result.stats.get(col, {}).get("mean"),
                        "std": result.stats.get(col, {}).get("std"),
                        "min": result.stats.get(col, {}).get("min"),
                        "max": result.stats.get(col, {}).get("max"),
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class FeatureInfo:
    """Metadata about a registered feature.

    Attributes:
        name: Feature name.
        func: The feature function.
        required_cols: List of required input column names.
        output_cols_fn: Function to generate output column names from params.
        description: Description of the feature.
    """

    name: str
    func: Callable
    required_cols: list[str]
    output_cols_fn: Callable[[dict[str, Any]], list[str]] | None = None
    description: str = ""
