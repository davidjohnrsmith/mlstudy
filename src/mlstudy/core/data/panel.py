"""Panel data utilities for bond time series.

Provides validation and reshaping utilities for panel data
with datetime, bond_id, and various value columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PanelValidationResult:
    """Results from panel data validation."""

    is_valid: bool
    n_dates: int
    n_bonds: int
    n_observations: int

    # Issues found
    duplicate_keys: int
    missing_rate: float
    non_monotonic_bonds: list[str]
    bonds_with_gaps: list[str]

    # Warnings
    warnings: list[str]

    def __str__(self) -> str:
        """Format validation result as string."""
        lines = [
            f"Panel Validation: {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Dates: {self.n_dates}",
            f"  Bonds: {self.n_bonds}",
            f"  Observations: {self.n_observations}",
            f"  Missing rate: {self.missing_rate:.2%}",
        ]

        if self.duplicate_keys > 0:
            lines.append(f"  ERROR: {self.duplicate_keys} duplicate (date, bond) keys")

        if self.non_monotonic_bonds:
            lines.append(f"  ERROR: {len(self.non_monotonic_bonds)} bonds with non-monotonic datetime")

        if self.bonds_with_gaps:
            lines.append(f"  WARNING: {len(self.bonds_with_gaps)} bonds with date gaps")

        for w in self.warnings:
            lines.append(f"  WARNING: {w}")

        return "\n".join(lines)


def validate_panel(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    value_cols: list[str] | None = None,
    require_balanced: bool = False,
    max_gap_days: int | None = None,
) -> PanelValidationResult:
    """Validate panel data for bond time series.

    Checks:
    - No duplicate (datetime, bond_id) pairs
    - Datetime is monotonically increasing per bond
    - Missing value rates
    - Optional: balanced panel (all bonds on all dates)
    - Optional: no gaps larger than max_gap_days

    Args:
        df: Panel DataFrame.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        value_cols: Value columns to check for missing data. If None, checks all numeric columns.
        require_balanced: If True, warns when panel is unbalanced.
        max_gap_days: If set, warns about gaps larger than this.

    Returns:
        PanelValidationResult with validation details.

    Example:
        >>> result = validate_panel(df, datetime_col="datetime", bond_id_col="bond_id")
        >>> if not result.is_valid:
        ...     print(result)
    """
    warnings = []
    is_valid = True

    # Basic counts
    n_dates = df[datetime_col].nunique()
    n_bonds = df[bond_id_col].nunique()
    n_observations = len(df)

    # Check for duplicate keys
    key_counts = df.groupby([datetime_col, bond_id_col]).size()
    duplicate_keys = int((key_counts > 1).sum())
    if duplicate_keys > 0:
        is_valid = False

    # Check monotonic datetime per bond
    non_monotonic_bonds = []
    for bond_id, group in df.groupby(bond_id_col):
        dates = group[datetime_col].values
        if len(dates) > 1 and not np.all(dates[:-1] <= dates[1:]):
            non_monotonic_bonds.append(str(bond_id))
            is_valid = False

    # Check for gaps
    bonds_with_gaps = []
    if max_gap_days is not None:
        for bond_id, group in df.groupby(bond_id_col):
            dates = pd.to_datetime(group[datetime_col]).sort_values()
            if len(dates) > 1:
                gaps = dates.diff().dt.days
                max_gap = gaps.max()
                if max_gap > max_gap_days:
                    bonds_with_gaps.append(str(bond_id))

    # Check missing values
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude datetime if numeric
        value_cols = [c for c in value_cols if c != datetime_col]

    if value_cols:
        missing_count = df[value_cols].isna().sum().sum()
        total_values = len(df) * len(value_cols)
        missing_rate = missing_count / total_values if total_values > 0 else 0.0
    else:
        missing_rate = 0.0

    # Check balanced panel
    if require_balanced:
        expected_obs = n_dates * n_bonds
        if n_observations < expected_obs:
            missing_obs = expected_obs - n_observations
            warnings.append(f"Unbalanced panel: {missing_obs} missing observations")

    return PanelValidationResult(
        is_valid=is_valid,
        n_dates=n_dates,
        n_bonds=n_bonds,
        n_observations=n_observations,
        duplicate_keys=duplicate_keys,
        missing_rate=missing_rate,
        non_monotonic_bonds=non_monotonic_bonds,
        bonds_with_gaps=bonds_with_gaps,
        warnings=warnings,
    )


def pivot_by_bond(
    df: pd.DataFrame,
    value_col: str,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
) -> pd.DataFrame:
    """Pivot panel data to wide format with bonds as columns.

    Args:
        df: Panel DataFrame with datetime, bond_id, and value columns.
        value_col: Value column to pivot.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.

    Returns:
        DataFrame with datetime as index, bond_ids as columns.

    Example:
        >>> yields = pivot_by_bond(df, value_col="yield")
        >>> # yields.columns = ['bond_01', 'bond_02', ...]
        >>> # yields.index = DatetimeIndex
    """
    pivoted = df.pivot(index=datetime_col, columns=bond_id_col, values=value_col)
    pivoted.index = pd.to_datetime(pivoted.index)
    return pivoted


def unpivot_to_panel(
    wide_df: pd.DataFrame,
    value_name: str = "value",
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
) -> pd.DataFrame:
    """Convert wide format back to panel (long) format.

    Args:
        wide_df: Wide DataFrame with datetime index and bond_id columns.
        value_name: Name for the value column.
        datetime_col: Name for datetime column in output.
        bond_id_col: Name for bond_id column in output.

    Returns:
        Panel DataFrame with datetime, bond_id, and value columns.
    """
    result = wide_df.reset_index().melt(
        id_vars=[wide_df.index.name or "index"],
        var_name=bond_id_col,
        value_name=value_name,
    )
    result = result.rename(columns={wide_df.index.name or "index": datetime_col})
    return result


def fill_panel_gaps(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    freq: str = "D",
    method: str = "ffill",
    limit: int | None = None,
) -> pd.DataFrame:
    """Fill gaps in panel data by forward/backward filling.

    Args:
        df: Panel DataFrame.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        freq: Date frequency for reindexing (e.g., "D" for daily).
        method: Fill method ("ffill" or "bfill").
        limit: Maximum number of consecutive fills.

    Returns:
        DataFrame with gaps filled.
    """
    # Get full date range
    all_dates = pd.date_range(
        df[datetime_col].min(),
        df[datetime_col].max(),
        freq=freq,
    )

    all_bonds = df[bond_id_col].unique()

    # Create full index
    full_index = pd.MultiIndex.from_product(
        [all_dates, all_bonds],
        names=[datetime_col, bond_id_col],
    )

    # Reindex and fill
    df_indexed = df.set_index([datetime_col, bond_id_col])
    df_reindexed = df_indexed.reindex(full_index)

    # Fill within each bond
    if method == "ffill":
        df_filled = df_reindexed.groupby(level=bond_id_col).ffill(limit=limit)
    else:
        df_filled = df_reindexed.groupby(level=bond_id_col).bfill(limit=limit)

    return df_filled.reset_index()


def get_panel_summary(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Get summary statistics per bond.

    Args:
        df: Panel DataFrame.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        value_cols: Columns to summarize. If None, uses all numeric columns.

    Returns:
        DataFrame with summary stats per bond.
    """
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    summary = df.groupby(bond_id_col).agg(
        n_obs=(datetime_col, "count"),
        first_date=(datetime_col, "min"),
        last_date=(datetime_col, "max"),
        **{
            f"{col}_mean": (col, "mean") for col in value_cols
        },
        **{
            f"{col}_std": (col, "std") for col in value_cols
        },
    )

    return summary


def align_panels(
    *dfs: pd.DataFrame,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    how: str = "inner",
) -> list[pd.DataFrame]:
    """Align multiple panel DataFrames to common (datetime, bond_id) pairs.

    Args:
        *dfs: Panel DataFrames to align.
        datetime_col: Datetime column name.
        bond_id_col: Bond identifier column name.
        how: Join type ("inner" or "outer").

    Returns:
        List of aligned DataFrames.
    """
    if len(dfs) < 2:
        return list(dfs)

    # Get common keys
    keys = [set(zip(df[datetime_col], df[bond_id_col], strict=True)) for df in dfs]

    if how == "inner":
        common_keys = keys[0]
        for k in keys[1:]:
            common_keys = common_keys.intersection(k)
    else:
        common_keys = keys[0]
        for k in keys[1:]:
            common_keys = common_keys.union(k)

    # Filter each DataFrame
    results = []
    for df in dfs:
        mask = df.apply(
            lambda row: (row[datetime_col], row[bond_id_col]) in common_keys,
            axis=1,
        )
        results.append(df[mask].copy())

    return results
