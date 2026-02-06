"""Cross-sectional ranking and normalization features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlstudy.core.features.registry import register_feature


@register_feature(
    name="cross_sectional_rank",
    required_cols=["{value_col}"],
    output_cols_fn=lambda p: [f"{p.get('value_col')}_cs_rank"],
    description="Rank values cross-sectionally at each datetime.",
)
def cross_sectional_rank(
    df: pd.DataFrame,
    value_col: str,
    datetime_col: str,
    shift: int = 1,
    pct: bool = True,
) -> pd.DataFrame:
    """Compute cross-sectional rank at each datetime.

    Ranks all values within each datetime period. Useful for
    comparing relative position across assets at each time point.

    Args:
        df: Input DataFrame.
        value_col: Column to rank.
        datetime_col: Datetime column for grouping.
        shift: Shift for leak safety (default 1).
        pct: If True, return percentile ranks (0-1). If False, integer ranks.

    Returns:
        DataFrame with cross-sectional rank column.
    """
    out_col = f"{value_col}_cs_rank"

    # Shift the value first to avoid lookahead
    shifted_val = df[value_col].shift(shift)

    # Rank within each datetime
    if pct:
        ranked = df.assign(_val=shifted_val).groupby(datetime_col)["_val"].rank(pct=True)
    else:
        ranked = df.assign(_val=shifted_val).groupby(datetime_col)["_val"].rank()

    return pd.DataFrame({out_col: ranked})


@register_feature(
    name="cross_sectional_zscore",
    required_cols=["{value_col}"],
    output_cols_fn=lambda p: [f"{p.get('value_col')}_cs_zscore"],
    description="Z-score normalize values cross-sectionally at each datetime.",
)
def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    datetime_col: str,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute cross-sectional z-score at each datetime.

    Normalizes values to have mean 0 and std 1 within each datetime period.

    Args:
        df: Input DataFrame.
        value_col: Column to normalize.
        datetime_col: Datetime column for grouping.
        shift: Shift for leak safety (default 1).

    Returns:
        DataFrame with cross-sectional z-score column.
    """
    out_col = f"{value_col}_cs_zscore"

    # Shift the value first to avoid lookahead
    shifted_val = df[value_col].shift(shift)
    df_temp = df.assign(_val=shifted_val)

    # Compute mean and std within each datetime
    grouped = df_temp.groupby(datetime_col)["_val"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")

    zscore = (shifted_val - mean) / std.replace(0, np.nan)

    return pd.DataFrame({out_col: zscore})


@register_feature(
    name="cross_sectional_demean",
    required_cols=["{value_col}"],
    output_cols_fn=lambda p: [f"{p.get('value_col')}_cs_demean"],
    description="Demean values cross-sectionally at each datetime.",
)
def cross_sectional_demean(
    df: pd.DataFrame,
    value_col: str,
    datetime_col: str,
    shift: int = 1,
) -> pd.DataFrame:
    """Compute cross-sectional demeaned values at each datetime.

    Subtracts the cross-sectional mean at each datetime.

    Args:
        df: Input DataFrame.
        value_col: Column to demean.
        datetime_col: Datetime column for grouping.
        shift: Shift for leak safety (default 1).

    Returns:
        DataFrame with cross-sectional demeaned column.
    """
    out_col = f"{value_col}_cs_demean"

    shifted_val = df[value_col].shift(shift)
    df_temp = df.assign(_val=shifted_val)

    mean = df_temp.groupby(datetime_col)["_val"].transform("mean")
    demeaned = shifted_val - mean

    return pd.DataFrame({out_col: demeaned})
