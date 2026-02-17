"""Load market data from parquet files for the mean-reversion backtester.

Reads long-format parquet files (datetime × instrument_id), aligns them
onto a common datetime index with forward-fill (no forward-looking), pivots
to the numpy arrays expected by ``run_backtest`` / ``run_sweep``.

Usage::

    loader = BacktestDataLoader(
        data_path="data/20240101",
        book_filename="book.parquet",
        mid_filename="mid.parquet",
        dv01_filename="dv01.parquet",
        signal_filename="signal.parquet",
        hedge_ratio_filename="hedge_ratios.parquet",
        instrument_ids=["UST_2Y", "UST_5Y", "UST_10Y"],
        ref_instrument_id="UST_5Y",
    )
    md = loader.load()
    results = run_sweep(scenarios, **md.to_dict())
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketData:
    """Container for aligned market data arrays.

    Shapes
    ------
    bid_px, bid_sz, ask_px, ask_sz : (T, N, L)
    mid_px, dv01, hedge_ratios : (T, N)
    zscore, expected_yield_pnl_bps, package_yield_bps : (T,)
    datetimes : (T,) datetime64
    instrument_ids : list[str] of length N
    """

    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    mid_px: np.ndarray
    dv01: np.ndarray
    zscore: np.ndarray
    expected_yield_pnl_bps: np.ndarray
    package_yield_bps: np.ndarray
    hedge_ratios: np.ndarray
    datetimes: np.ndarray
    instrument_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return dict suitable for ``run_sweep(**md.to_dict())``."""
        return {
            "bid_px": self.bid_px,
            "bid_sz": self.bid_sz,
            "ask_px": self.ask_px,
            "ask_sz": self.ask_sz,
            "mid_px": self.mid_px,
            "dv01": self.dv01,
            "zscore": self.zscore,
            "expected_yield_pnl_bps": self.expected_yield_pnl_bps,
            "package_yield_bps": self.package_yield_bps,
            "hedge_ratios": self.hedge_ratios,
        }


@dataclass
class BacktestDataLoader:
    """Loads and aligns parquet market data for the MR backtester.

    Parameters
    ----------
    book_filename, mid_filename, dv01_filename, signal_filename, hedge_ratio_filename : str
        Filenames within *data_path*.
    instrument_ids : list[str]
        Ordered list of instrument IDs — defines N and column order.
    ref_instrument_id : str
        Instrument ID to filter signal by (produces ``(T,)`` arrays).
    data_path : str, Path, or None
        Directory containing the parquet files.  Can be omitted here
        and passed to :meth:`load` instead — useful for keeping the
        YAML config platform-independent while supplying the path at
        launch time.
    datetime_col : str
        Name of the datetime column in parquets (default ``"datetime"``).
    instrument_col : str
        Name of the instrument ID column (default ``"instrument_id"``).
    fill_method : str
        ``"ffill"`` (forward-fill then drop leading NaN rows) or
        ``"drop"`` (keep only rows where all sources have data).
    """

    book_filename: str
    mid_filename: str
    dv01_filename: str
    signal_filename: str
    hedge_ratio_filename: str
    instrument_ids: list[str]
    ref_instrument_id: str
    data_path: str | Path | None = None
    datetime_col: str = "datetime"
    instrument_col: str = "instrument_id"
    fill_method: str = "ffill"

    def load(self, data_path: str | Path | None = None) -> MarketData:
        """Read parquets, align, pivot, and return ``MarketData``.

        Parameters
        ----------
        data_path : str, Path, or None
            Directory containing the parquet files.  Overrides
            ``self.data_path`` when supplied — use this to keep a
            platform-independent YAML config and pass the path at
            runtime.
        """
        resolved = data_path or self.data_path
        if resolved is None:
            raise ValueError(
                "data_path must be provided either at construction time "
                "or as an argument to load()"
            )
        data_path_dir = Path(resolved)
        dt_col = self.datetime_col
        inst_col = self.instrument_col
        instruments = self.instrument_ids
        n_inst = len(instruments)

        # --- 1. Read parquets ---------------------------------------------------
        book_df = pd.read_parquet(data_path_dir / self.book_filename)
        mid_df = pd.read_parquet(data_path_dir / self.mid_filename)
        dv01_df = pd.read_parquet(data_path_dir / self.dv01_filename)
        signal_df = pd.read_parquet(data_path_dir / self.signal_filename)
        hedge_df = pd.read_parquet(data_path_dir / self.hedge_ratio_filename)

        # --- 2. Auto-detect book levels -----------------------------------------
        n_levels = _detect_book_levels(book_df.columns)
        logger.info("Detected %d book levels", n_levels)

        # --- 3. Filter signal to ref instrument ---------------------------------
        signal_df = signal_df[
            signal_df[inst_col] == self.ref_instrument_id
        ].copy()
        if signal_df.empty:
            raise ValueError(
                f"No signal data for ref_instrument_id={self.ref_instrument_id!r}"
            )

        # --- 4. Pivot long → wide -----------------------------------------------
        # Book: pivot each level column per instrument
        book_pivoted = _pivot_book(book_df, dt_col, inst_col, instruments, n_levels)
        mid_pivoted = _pivot_simple(mid_df, dt_col, inst_col, instruments, "mid_px")
        dv01_pivoted = _pivot_simple(dv01_df, dt_col, inst_col, instruments, "dv01")

        # Signal: already filtered to single instrument, just set index
        signal_indexed = signal_df.set_index(dt_col)[
            ["zscore", "expected_yield_pnl_bps", "package_yield_bps"]
        ].sort_index()

        # Hedge ratios: filter to ref instrument first (no list expansion),
        # then pivot the list columns into a (T_hedge, N) DataFrame.
        hedge_pivoted = _pivot_hedge_ratios(
            hedge_df, inst_col, dt_col,
            self.ref_instrument_id, instruments,
        )

        # --- 5. Build unified datetime index & align ----------------------------
        all_dts = sorted(
            set(book_pivoted.index)
            | set(mid_pivoted.index)
            | set(dv01_pivoted.index)
            | set(signal_indexed.index)
            | set(hedge_pivoted.index)
        )
        all_dts_idx = pd.DatetimeIndex(all_dts)

        sources = {
            "book": book_pivoted.reindex(all_dts_idx),
            "mid": mid_pivoted.reindex(all_dts_idx),
            "dv01": dv01_pivoted.reindex(all_dts_idx),
            "signal": signal_indexed.reindex(all_dts_idx),
            "hedge": hedge_pivoted.reindex(all_dts_idx),
        }

        if self.fill_method == "ffill":
            for key in sources:
                sources[key] = sources[key].ffill()
            # Drop leading rows where any source still has NaN
            all_valid = pd.concat(
                [s.notna().all(axis=1) for s in sources.values()], axis=1
            ).all(axis=1)
            first_valid = all_valid.idxmax() if all_valid.any() else None
            if first_valid is None:
                raise ValueError("No rows with complete data after ffill")
            mask = all_dts_idx >= first_valid
            for key in sources:
                sources[key] = sources[key].loc[mask]
            all_dts_idx = all_dts_idx[mask]
        elif self.fill_method == "drop":
            all_valid = pd.concat(
                [s.notna().all(axis=1) for s in sources.values()], axis=1
            ).all(axis=1)
            mask = all_valid.values
            for key in sources:
                sources[key] = sources[key].loc[mask]
            all_dts_idx = all_dts_idx[mask]
            if len(all_dts_idx) == 0:
                raise ValueError("No rows with complete data in drop mode")
        else:
            raise ValueError(f"Unknown fill_method={self.fill_method!r}")

        T = len(all_dts_idx)

        # --- 6. Extract numpy arrays --------------------------------------------
        # Book: (T, N*L*4 columns) → (T, N, L) per field
        bid_px, bid_sz, ask_px, ask_sz = _reshape_book(
            sources["book"], instruments, n_levels
        )

        mid_px = sources["mid"].values.astype(np.float64)   # (T, N)
        dv01_arr = sources["dv01"].values.astype(np.float64)  # (T, N)

        zscore = sources["signal"]["zscore"].values.astype(np.float64)
        expected_yield_pnl_bps = sources["signal"]["expected_yield_pnl_bps"].values.astype(np.float64)
        package_yield_bps = sources["signal"]["package_yield_bps"].values.astype(np.float64)

        hedge_ratios = sources["hedge"].values.astype(np.float64)  # (T, N)

        datetimes = all_dts_idx.values  # datetime64

        # --- 8. Validate ----------------------------------------------------------
        _validate_shapes(
            bid_px, bid_sz, ask_px, ask_sz, mid_px, dv01_arr,
            zscore, expected_yield_pnl_bps, package_yield_bps,
            hedge_ratios, T, n_inst, n_levels,
        )
        _warn_nans(
            bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
            mid_px=mid_px, dv01=dv01_arr,
        )

        return MarketData(
            bid_px=bid_px,
            bid_sz=bid_sz,
            ask_px=ask_px,
            ask_sz=ask_sz,
            mid_px=mid_px,
            dv01=dv01_arr,
            zscore=zscore,
            expected_yield_pnl_bps=expected_yield_pnl_bps,
            package_yield_bps=package_yield_bps,
            hedge_ratios=hedge_ratios,
            datetimes=datetimes,
            instrument_ids=list(self.instrument_ids),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_book_levels(columns: pd.Index) -> int:
    """Auto-detect the number of book levels from column names.

    Looks for ``bid_price_l0, bid_price_l1, ...`` pattern.
    """
    levels = set()
    for col in columns:
        m = re.match(r"bid_price_l(\d+)$", col)
        if m:
            levels.add(int(m.group(1)))
    if not levels:
        raise ValueError(
            "Cannot detect book levels: no columns matching 'bid_price_l<N>' found"
        )
    n = max(levels) + 1
    if levels != set(range(n)):
        raise ValueError(f"Book levels are not contiguous: found {sorted(levels)}")
    return n


def _pivot_simple(
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


def _pivot_book(
    df: pd.DataFrame,
    dt_col: str,
    inst_col: str,
    instruments: list[str],
    n_levels: int,
) -> pd.DataFrame:
    """Pivot book data: each (field, level, instrument) becomes a column.

    Returns a wide DataFrame with columns named like
    ``bid_price_l0__UST_2Y, bid_size_l0__UST_2Y, ...``.
    """
    value_cols = []
    for l in range(n_levels):
        value_cols.extend([
            f"bid_price_l{l}", f"bid_size_l{l}",
            f"ask_price_l{l}", f"ask_size_l{l}",
        ])

    # Pivot each value column separately, then concat
    parts = []
    for vc in value_cols:
        p = df.pivot(index=dt_col, columns=inst_col, values=vc)
        p = p.reindex(columns=instruments)
        # Rename columns to include the value field
        p.columns = [f"{vc}__{inst}" for inst in instruments]
        parts.append(p)

    result = pd.concat(parts, axis=1)
    result.index = pd.DatetimeIndex(result.index)
    result = result.sort_index()
    return result


def _reshape_book(
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
        for l in range(L):
            bid_px[:, j, l] = book_wide[f"bid_price_l{l}__{inst}"].values
            bid_sz[:, j, l] = book_wide[f"bid_size_l{l}__{inst}"].values
            ask_px[:, j, l] = book_wide[f"ask_price_l{l}__{inst}"].values
            ask_sz[:, j, l] = book_wide[f"ask_size_l{l}__{inst}"].values

    return bid_px, bid_sz, ask_px, ask_sz


def _pivot_hedge_ratios(
    hedge_df: pd.DataFrame,
    inst_col: str,
    dt_col: str,
    ref_instrument_id: str,
    instruments: list[str],
) -> pd.DataFrame:
    """Pivot list-column hedge ratios into a wide ``(T, N)`` DataFrame.

    The parquet has columns ``[datetime, instrument_id, hedge_instruments,
    hedge_ratios]`` where ``hedge_instruments`` and ``hedge_ratios`` are
    lists.  The ``instrument_id`` column identifies the target/reference
    instrument; the target instrument itself is **not** in
    ``hedge_instruments`` — it is added here with ratio 1.0.

    Steps:

    1. Filter to ``ref_instrument_id`` — no list expansion beforehand.
    2. For each row, map the list entries onto the ordered ``instruments``
       positions, setting the target instrument to 1.0.
    3. Return a DataFrame indexed by datetime with one column per instrument.
    """
    ref_rows = hedge_df[hedge_df[inst_col] == ref_instrument_id].copy()
    if ref_rows.empty:
        raise ValueError(
            f"No hedge ratio data for ref_instrument_id={ref_instrument_id!r}"
        )

    ref_rows = ref_rows.sort_values(dt_col)

    N = len(instruments)
    inst_to_idx = {inst: j for j, inst in enumerate(instruments)}
    ref_idx = inst_to_idx.get(ref_instrument_id)

    datetimes = []
    ratio_rows = []

    for _, row in ref_rows.iterrows():
        hi_list = list(row["hedge_instruments"])
        hr_list = list(row["hedge_ratios"])

        if len(hi_list) != len(hr_list):
            raise ValueError(
                f"hedge_instruments length ({len(hi_list)}) != "
                f"hedge_ratios length ({len(hr_list)}) "
                f"at {row[dt_col]}"
            )

        ratios = np.full(N, np.nan, dtype=np.float64)

        # Target instrument gets 1.0
        if ref_idx is not None:
            ratios[ref_idx] = 1.0

        for hi, hr in zip(hi_list, hr_list):
            j = inst_to_idx.get(hi)
            if j is not None:
                ratios[j] = hr

        # Check that all backtest instruments are covered
        missing = [
            instruments[j] for j in range(N)
            if np.isnan(ratios[j])
        ]
        if missing:
            raise ValueError(
                f"Instruments {missing} not found in hedge_instruments "
                f"for ref {ref_instrument_id!r} at {row[dt_col]}. "
                f"Available: {hi_list}"
            )

        datetimes.append(row[dt_col])
        ratio_rows.append(ratios)

    result = pd.DataFrame(
        np.array(ratio_rows),
        index=pd.DatetimeIndex(datetimes),
        columns=instruments,
    )
    result.index.name = dt_col

    # Validate sum ≈ 0 on last row
    hr_sum = float(result.iloc[-1].sum())
    if abs(hr_sum) > 1e-6:
        warnings.warn(
            f"Hedge ratios do not sum to zero: sum={hr_sum:.6g}",
            stacklevel=3,
        )

    return result


def _validate_shapes(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    dv01: np.ndarray,
    zscore: np.ndarray,
    expected_yield_pnl_bps: np.ndarray,
    package_yield_bps: np.ndarray,
    hedge_ratios: np.ndarray,
    T: int,
    N: int,
    L: int,
) -> None:
    """Validate that all arrays have consistent shapes."""
    expected = {
        "bid_px": (T, N, L),
        "bid_sz": (T, N, L),
        "ask_px": (T, N, L),
        "ask_sz": (T, N, L),
        "mid_px": (T, N),
        "dv01": (T, N),
        "zscore": (T,),
        "expected_yield_pnl_bps": (T,),
        "package_yield_bps": (T,),
        "hedge_ratios": (T, N),
    }
    arrays = {
        "bid_px": bid_px,
        "bid_sz": bid_sz,
        "ask_px": ask_px,
        "ask_sz": ask_sz,
        "mid_px": mid_px,
        "dv01": dv01,
        "zscore": zscore,
        "expected_yield_pnl_bps": expected_yield_pnl_bps,
        "package_yield_bps": package_yield_bps,
        "hedge_ratios": hedge_ratios,
    }
    for name, arr in arrays.items():
        if arr.shape != expected[name]:
            raise ValueError(
                f"{name} shape mismatch: expected {expected[name]}, got {arr.shape}"
            )


def _warn_nans(**arrays: np.ndarray) -> None:
    """Warn if any arrays contain NaN values."""
    for name, arr in arrays.items():
        n_nan = np.isnan(arr).sum()
        if n_nan > 0:
            warnings.warn(
                f"{name} contains {n_nan} NaN values ({n_nan / arr.size:.1%})",
                stacklevel=2,
            )
