"""Time-based train/validation/test split."""

from __future__ import annotations

import pandas as pd


def time_train_val_test_split(
    df: pd.DataFrame,
    datetime_col: str,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    group_col: str | None = None,
    min_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by time into train/validation/test sets.

    Creates non-overlapping splits based on datetime boundaries:
    - Train: datetime <= train_end
    - Validation: train_end < datetime <= val_end
    - Test: val_end < datetime <= test_end (or all remaining if test_end is None)

    Args:
        df: Input DataFrame.
        datetime_col: Column name containing datetime values.
        train_end: End date for training set (inclusive).
        val_end: End date for validation set (inclusive).
        test_end: End date for test set (inclusive). If None, includes all
            data after val_end.
        group_col: Optional grouping column. If provided, groups with fewer
            than min_count samples in any split are dropped entirely.
        min_count: Minimum samples per group in each split. Only used if
            group_col is provided.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If train_end >= val_end or val_end >= test_end.
    """
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    if train_end >= val_end:
        raise ValueError(f"train_end ({train_end}) must be before val_end ({val_end})")

    if test_end is not None:
        test_end = pd.to_datetime(test_end)
        if val_end >= test_end:
            raise ValueError(f"val_end ({val_end}) must be before test_end ({test_end})")

    dates = pd.to_datetime(df[datetime_col])

    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)

    test_mask = (
        (dates > val_end) & (dates <= test_end) if test_end is not None else dates > val_end
    )

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Filter by min_count per group if specified
    if group_col is not None and min_count > 0:
        # Find groups that meet min_count in all splits
        train_counts = train_df[group_col].value_counts()
        val_counts = val_df[group_col].value_counts()
        test_counts = test_df[group_col].value_counts()

        valid_groups = set(train_counts[train_counts >= min_count].index)
        valid_groups &= set(val_counts[val_counts >= min_count].index)
        valid_groups &= set(test_counts[test_counts >= min_count].index)

        train_df = train_df[train_df[group_col].isin(valid_groups)]
        val_df = val_df[val_df[group_col].isin(valid_groups)]
        test_df = test_df[test_df[group_col].isin(valid_groups)]

    return train_df, val_df, test_df
