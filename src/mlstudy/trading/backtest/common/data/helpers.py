"""Shared data-loading helpers for backtest data loaders.

Provides reusable functions for detecting book levels, pivoting long-format
DataFrames to wide format, reshaping book data into numpy arrays, and
warning about NaN values.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd


def detect_book_levels(columns: pd.Index) -> int:
    """Auto-detect the number of book levels from column names.

    Looks for ``bid_px_l0, bid_px_l1, ...`` pattern.
    """
    levels = set()
    for col in columns:
        m = re.match(r"bid_px_l(\d+)$", col)
        if m:
            levels.add(int(m.group(1)))
    if not levels:
        raise ValueError(
            "Cannot detect book levels: no columns matching 'bid_px_l<N>' found"
        )
    n = max(levels) + 1
    if levels != set(range(n)):
        raise ValueError(f"Book levels are not contiguous: found {sorted(levels)}")
    return n


def pivot_simple(
    df: pd.DataFrame,
    dt_col: str,
    inst_col: str,
    instruments: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Pivot a long-format DataFrame to wide: index=datetime, columns=instruments."""
    pivoted = df.pivot(index=dt_col, columns=inst_col, values=value_col)
    # Reorder columns to match instrument_ids; missing instruments → NaN
    pivoted = pivoted.reindex(columns=instruments)
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted


def pivot_book(
    df: pd.DataFrame,
    dt_col: str,
    inst_col: str,
    instruments: list[str],
    n_levels: int,
) -> pd.DataFrame:
    """Pivot book data: each (field, level, instrument) becomes a column.

    Returns a wide DataFrame with columns named like
    ``bid_px_l0__UST_2Y, bid_size_l0__UST_2Y, ...``.
    """
    value_cols = []
    for lvl in range(n_levels):
        value_cols.extend([
            f"bid_px_l{lvl}", f"bid_sz_l{lvl}",
            f"ask_px_l{lvl}", f"ask_sz_l{lvl}",
        ])

    parts = []
    for vc in value_cols:
        p = df.pivot(index=dt_col, columns=inst_col, values=vc)
        p = p.reindex(columns=instruments)
        p.columns = [f"{vc}__{inst}" for inst in instruments]
        parts.append(p)

    result = pd.concat(parts, axis=1)
    result.index = pd.DatetimeIndex(result.index)
    result = result.sort_index()
    return result


def reshape_book(
    book_wide: pd.DataFrame,
    instruments: list[str],
    n_levels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape pivoted book DataFrame into (T, N, L) arrays.

    Returns (bid_px, bid_sz, ask_px, ask_sz).
    """
    T = len(book_wide)
    N = len(instruments)
    L = n_levels

    bid_px = np.empty((T, N, L), dtype=np.float64)
    bid_sz = np.empty((T, N, L), dtype=np.float64)
    ask_px = np.empty((T, N, L), dtype=np.float64)
    ask_sz = np.empty((T, N, L), dtype=np.float64)

    for j, inst in enumerate(instruments):
        for lvl in range(L):
            bid_px[:, j, lvl] = book_wide[f"bid_px_l{lvl}__{inst}"].values
            bid_sz[:, j, lvl] = book_wide[f"bid_sz_l{lvl}__{inst}"].values
            ask_px[:, j, lvl] = book_wide[f"ask_px_l{lvl}__{inst}"].values
            ask_sz[:, j, lvl] = book_wide[f"ask_sz_l{lvl}__{inst}"].values

    return bid_px, bid_sz, ask_px, ask_sz


def warn_nans(**arrays: np.ndarray) -> None:
    """Warn if any arrays contain NaN values."""
    for name, arr in arrays.items():
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            warnings.warn(
                f"{name} contains {n_nan} NaN values ({n_nan / arr.size:.1%})",
                stacklevel=2,
            )


def align_and_fill(
    sources: dict[str, pd.DataFrame],
    fill_method: str,
    essential_keys: tuple[str, ...] | None = None,
    fillna_defaults: dict[str, float] | None = None,
    datetime_source_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Align multiple DataFrames onto a common datetime index and fill NaNs.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Pivoted wide-format DataFrames, each indexed by datetime.
    fill_method : str
        ``"ffill"`` — forward-fill all, drop leading rows where
        *essential_keys* still have NaN, then apply *fillna_defaults*.
        ``"drop"`` — keep only rows where all sources have data.
    essential_keys : tuple[str, ...] or None
        Keys whose NaN values determine where to trim leading rows
        (only used with ``"ffill"``).
    fillna_defaults : dict[str, float] or None
        After ffill + trim, fill remaining NaN in these sources with
        the given default values (e.g. ``{"dv01": 1.0, "mid": 0.0}``).
    datetime_source_keys : tuple[str, ...] or None
        When provided, only collect datetimes from these source keys
        instead of from all sources.  Other sources are still reindexed
        onto the resulting datetime index.

    Returns
    -------
    (aligned_sources, datetime_index)
    """
    # Build unified datetime index
    all_dts: set = set()
    if datetime_source_keys is not None:
        for key in datetime_source_keys:
            all_dts.update(sources[key].index)
    else:
        for df in sources.values():
            all_dts.update(df.index)
    all_dts_idx = pd.DatetimeIndex(sorted(all_dts))

    # Reindex all sources
    aligned = {k: df.reindex(all_dts_idx) for k, df in sources.items()}

    if fill_method == "ffill":
        for key in aligned:
            aligned[key] = aligned[key].ffill()

        # Drop leading rows where essential sources still have NaN
        if essential_keys:
            essential_valid = pd.concat(
                [aligned[k].notna().all(axis=1) for k in essential_keys],
                axis=1,
            ).all(axis=1)
            first_valid = (
                essential_valid.idxmax() if essential_valid.any() else None
            )
            if first_valid is None:
                raise ValueError("No rows with complete data after ffill")
            mask = all_dts_idx >= first_valid
            for key in aligned:
                aligned[key] = aligned[key].loc[mask]
            all_dts_idx = all_dts_idx[mask]

        # Fill remaining NaN with defaults
        if fillna_defaults:
            for key, default in fillna_defaults.items():
                if key in aligned:
                    aligned[key] = aligned[key].fillna(default)
    elif fill_method == "drop":
        all_valid = pd.concat(
            [s.notna().all(axis=1) for s in aligned.values()], axis=1,
        ).all(axis=1)
        mask = all_valid.values
        for key in aligned:
            aligned[key] = aligned[key].loc[mask]
        all_dts_idx = all_dts_idx[mask]
        if len(all_dts_idx) == 0:
            raise ValueError("No rows with complete data in drop mode")
    else:
        raise ValueError(f"Unknown fill_method={fill_method!r}")

    return aligned, all_dts_idx


def extract_hedge_ids(hedge_ratio_df: pd.DataFrame) -> list[str]:
    """Return sorted union of all hedge instruments from the hedge_ratios file."""
    all_hedges: set[str] = set()
    for hi_list in hedge_ratio_df["hedge_instruments"]:
        all_hedges.update(hi_list)
    return sorted(all_hedges)
