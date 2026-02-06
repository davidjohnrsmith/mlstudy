"""Walk-forward (rolling/expanding window) cross-validation splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    """A single fold from walk-forward cross-validation.

    Attributes:
        fold_id: Zero-based fold index.
        train_df: Training data for this fold.
        val_df: Validation data for this fold.
        test_df: Test data for this fold.
        train_start: Start date of training period.
        train_end: End date of training period.
        val_start: Start date of validation period.
        val_end: End date of validation period.
        test_start: Start date of test period.
        test_end: End date of test period.
    """

    fold_id: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    df: pd.DataFrame,
    datetime_col: str,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    expanding: bool = True,
    group_col: str | None = None,
    min_count: int = 0,
) -> Iterator[WalkForwardFold]:
    """Generate walk-forward cross-validation folds.

    Creates rolling or expanding window splits that respect time ordering.
    Each fold advances by step_days.

    For expanding windows (expanding=True):
        - Train window starts from the beginning and grows
        - Val and test windows slide forward

    For rolling windows (expanding=False):
        - Train window has fixed size and slides forward
        - Val and test windows slide forward

    Args:
        df: Input DataFrame.
        datetime_col: Column name containing datetime values.
        train_days: Number of days for training period (initial or fixed size).
        val_days: Number of days for validation period.
        test_days: Number of days for test period.
        step_days: Number of days to step forward between folds.
        expanding: If True, use expanding window for training. If False,
            use rolling window with fixed train_days size.
        group_col: Optional grouping column for filtering.
        min_count: Minimum samples per group in each split.

    Yields:
        WalkForwardFold objects for each valid fold.
    """
    df = df.copy()
    df["_dt"] = pd.to_datetime(df[datetime_col])
    df = df.sort_values("_dt")

    min_date = df["_dt"].min()
    max_date = df["_dt"].max()

    # Initial window boundaries
    train_start = min_date

    train_end = min_date + pd.Timedelta(days=train_days - 1)
    val_start = train_end + pd.Timedelta(days=1)
    val_end = val_start + pd.Timedelta(days=val_days - 1)
    test_start = val_end + pd.Timedelta(days=1)
    test_end = test_start + pd.Timedelta(days=test_days - 1)

    fold_id = 0

    while test_end <= max_date:
        # Extract splits
        train_mask = (df["_dt"] >= train_start) & (df["_dt"] <= train_end)
        val_mask = (df["_dt"] >= val_start) & (df["_dt"] <= val_end)
        test_mask = (df["_dt"] >= test_start) & (df["_dt"] <= test_end)

        train_df = df[train_mask].drop(columns=["_dt"]).copy()
        val_df = df[val_mask].drop(columns=["_dt"]).copy()
        test_df = df[test_mask].drop(columns=["_dt"]).copy()

        # Filter by min_count if group_col specified
        if group_col is not None and min_count > 0:
            train_counts = train_df[group_col].value_counts()
            val_counts = val_df[group_col].value_counts()
            test_counts = test_df[group_col].value_counts()

            valid_groups = set(train_counts[train_counts >= min_count].index)
            valid_groups &= set(val_counts[val_counts >= min_count].index)
            valid_groups &= set(test_counts[test_counts >= min_count].index)

            train_df = train_df[train_df[group_col].isin(valid_groups)]
            val_df = val_df[val_df[group_col].isin(valid_groups)]
            test_df = test_df[test_df[group_col].isin(valid_groups)]

        # Only yield if we have data in all splits
        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            yield WalkForwardFold(
                fold_id=fold_id,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
            fold_id += 1

        # Advance windows
        if expanding:
            # Train start stays fixed, end grows
            pass
        else:
            # Rolling: train start advances
            train_start = train_start + pd.Timedelta(days=step_days)

        train_end = train_end + pd.Timedelta(days=step_days)
        val_start = val_start + pd.Timedelta(days=step_days)
        val_end = val_end + pd.Timedelta(days=step_days)
        test_start = test_start + pd.Timedelta(days=step_days)
        test_end = test_end + pd.Timedelta(days=step_days)
