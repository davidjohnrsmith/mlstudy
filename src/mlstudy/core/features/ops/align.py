"""Alignment utilities for merging time series data."""

from __future__ import annotations

import pandas as pd


def merge_asof_on_datetime(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: str | list[str] | None = None,
    direction: str = "backward",
    tolerance: pd.Timedelta | None = None,
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    """Merge two DataFrames using asof join on datetime.

    Useful for aligning data from different sources (e.g., prices and signals)
    that may not have exact timestamp matches.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        on: Datetime column name (must be sorted in both DataFrames).
        by: Column(s) to match exactly before doing asof merge.
        direction: "backward" (default), "forward", or "nearest".
        tolerance: Maximum time difference for a match.
        suffixes: Suffixes for overlapping column names.

    Returns:
        Merged DataFrame.
    """
    left = left.sort_values(on)
    right = right.sort_values(on)

    return pd.merge_asof(
        left,
        right,
        on=on,
        by=by,
        direction=direction,
        tolerance=tolerance,
        suffixes=suffixes,
    )


def align_to_reference(
    df: pd.DataFrame,
    reference_dates: pd.Series | pd.DatetimeIndex,
    datetime_col: str,
    group_col: str | None = None,
    method: str = "ffill",
) -> pd.DataFrame:
    """Align DataFrame to a reference set of dates.

    Useful for aligning irregular data to a regular calendar or
    to another dataset's timestamps.

    Args:
        df: Input DataFrame.
        reference_dates: Target dates to align to.
        datetime_col: Name of datetime column.
        group_col: If provided, align within each group.
        method: Fill method for missing dates ("ffill", "bfill", None).

    Returns:
        DataFrame aligned to reference dates.
    """
    if isinstance(reference_dates, pd.Series):
        reference_dates = pd.DatetimeIndex(reference_dates.unique())

    if group_col is None:
        # Simple case: no grouping
        df = df.set_index(datetime_col)
        aligned = df.reindex(reference_dates, method=method)
        aligned = aligned.reset_index()
        aligned = aligned.rename(columns={"index": datetime_col})
        return aligned

    # With grouping: align each group separately
    results = []
    for group_val, group_df in df.groupby(group_col):
        group_df = group_df.set_index(datetime_col)
        aligned = group_df.reindex(reference_dates, method=method)
        aligned[group_col] = group_val
        aligned = aligned.reset_index()
        aligned = aligned.rename(columns={"index": datetime_col})
        results.append(aligned)

    return pd.concat(results, ignore_index=True)
