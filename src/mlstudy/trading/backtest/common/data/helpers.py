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
    no_ffill_keys: tuple[str, ...] | None = None,
    close_time: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Align multiple DataFrames onto a common datetime index and fill NaNs.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Pivoted wide-format DataFrames, each indexed by datetime.
    fill_method : str
        ``"ffill"`` — forward-fill all, drop rows where *essential_keys*
        have all-NaN across columns, then apply *fillna_defaults*.
        ``"drop"`` — keep only rows where all sources have data.
    essential_keys : tuple[str, ...] or None
        Keys whose data validity determines which rows to keep.  A row is
        invalid for a given key when **all** of its columns are NaN.  If a
        row is invalid in **any** essential key it is removed from every
        source — unless it is a close-time row (see *close_time*).
    fillna_defaults : dict[str, float] or None
        After ffill + trim, fill remaining NaN in these sources with
        the given default values (e.g. ``{"dv01": 1.0, "mid": 0.0}``).
    datetime_source_keys : tuple[str, ...] or None
        When provided, only collect datetimes from these source keys
        instead of from all sources.  Other sources are still reindexed
        onto the resulting datetime index.
    no_ffill_keys : tuple[str, ...] or None
        Source keys to exclude from forward-filling.  NaN values in
        these sources are filled with 0 instead.  Use for book data
        where stale quotes would create phantom liquidity.
    close_time : str or None
        Time string (e.g. ``"17:00"``).  When set, rows whose time
        matches *close_time* are never dropped by the essential-key
        check.  Instead their essential-key data is forward-filled from
        the previous valid row.  If a close-time row still has all-NaN
        essential data after forward-filling, a ``ValueError`` is raised.

    Returns
    -------
    (aligned_sources, datetime_index)
    """
    _no_ffill = set(no_ffill_keys) if no_ffill_keys else set()

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
            if key in _no_ffill:
                aligned[key] = aligned[key].fillna(0.0)
            else:
                aligned[key] = aligned[key].ffill()

        # Drop rows where any essential source has all-NaN across columns.
        # Close-time rows are exempt: they are forward-filled instead.
        if essential_keys:
            # A row is invalid for a key when every column is NaN.
            # valid_per_key[k] is True where at least one column is not NaN.
            valid_per_key = {
                k: aligned[k].notna().any(axis=1) for k in essential_keys
            }
            # Row is valid only if valid in ALL essential keys.
            all_valid = pd.concat(valid_per_key.values(), axis=1).all(axis=1)

            if close_time is not None:
                target_time = pd.Timestamp(close_time).time()
                is_close = pd.Series(
                    all_dts_idx.time == target_time, index=all_dts_idx,
                )
                # Close-time rows that are invalid need forward-fill
                invalid_close = is_close & ~all_valid
                if invalid_close.any():
                    # Forward-fill essential sources at invalid close rows
                    for k in essential_keys:
                        if k in _no_ffill:
                            continue
                        aligned[k] = aligned[k].ffill()
                    # Re-check: if close-time rows still all-NaN, raise
                    for k in essential_keys:
                        still_invalid = (
                            aligned[k].loc[invalid_close].isna().all(axis=1)
                        )
                        if still_invalid.any():
                            bad_dts = still_invalid[still_invalid].index.tolist()
                            raise ValueError(
                                f"Close-time rows still have all-NaN data in "
                                f"essential key {k!r} after forward-fill at: "
                                f"{bad_dts[:5]}"
                            )
                # Keep valid rows + close-time rows
                keep = all_valid | is_close
            else:
                keep = all_valid

            if not keep.any():
                raise ValueError("No rows with complete data after ffill")
            for key in aligned:
                aligned[key] = aligned[key].loc[keep]
            all_dts_idx = all_dts_idx[keep.values]

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
