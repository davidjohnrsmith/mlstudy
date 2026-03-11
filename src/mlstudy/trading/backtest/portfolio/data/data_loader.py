"""Load market data from parquet files for the portfolio backtester.

Reads long-format parquet files, aligns them onto a common datetime index,
pivots to the numpy arrays expected by ``run_backtest``.

All instruments (trading + hedge) share the same book/mid/dv01 files.
The loader auto-detects trading instruments from the meta file and hedge
instruments from the hedge_ratios file.

Usage::

    loader = PortfolioDataLoader(
        book_filename="book.parquet",
        mid_filename="mid.parquet",
        dv01_filename="dv01.parquet",
        signal_filename="signal.parquet",
        meta_filename="meta.parquet",
        hedge_ratio_filename="hedge_ratios.parquet",
    )
    md = loader.load(data_path="data/20240101")
    results = run_backtest(**md.to_dict())
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mlstudy.trading.backtest.common.data.helpers import (
    align_and_fill,
    detect_book_levels,
    pivot_book,
    pivot_simple,
    reshape_book,
    warn_nans,
)

logger = logging.getLogger(__name__)

# Required columns in the instrument meta file
META_REQUIRED_COLUMNS = (
    "tradable",
    "pos_limit_long",
    "pos_limit_short",
    "max_trade_notional_inc",
    "max_trade_notional_dec",
    "qty_step",
    "min_qty_trade",
    "maturity_date",
)

# Required columns in the hedge meta file
HEDGE_META_REQUIRED_COLUMNS = (
    "qty_step",
    "min_qty_trade",
)


@dataclass(frozen=True)
class PortfolioMarketData:
    """Container for aligned portfolio market data arrays.

    Shapes
    ------
    bid_px, bid_sz, ask_px, ask_sz : (T, B, L)
    mid_px, dv01 : (T, B)
    fair_price, zscore, adf_p_value : (T, B)
    tradable, pos_limits_long, pos_limits_short : (B,)
    max_trade_notional_inc, max_trade_notional_dec : (B,)
    maturity : (T, B) or (B,)
    issuer_bucket : (B,)
    maturity_bucket : (T, B) or (B,)
    issuer_dv01_caps : (n_issuers,)
    mat_bucket_dv01_caps : (n_buckets,)
    hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz : (T, H, L_h)
    hedge_mid_px, hedge_dv01 : (T, H)
    hedge_ratios : (T, B, H)
    datetimes : (T,) datetime64
    instrument_ids : list[str] of length B
    hedge_ids : list[str] of length H
    """

    # Instrument L2
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    mid_px: np.ndarray
    dv01: np.ndarray
    # Per-instrument signals
    fair_price: np.ndarray
    zscore: np.ndarray
    adf_p_value: np.ndarray
    # Static meta
    tradable: np.ndarray
    pos_limits_long: np.ndarray
    pos_limits_short: np.ndarray
    max_trade_notional_inc: np.ndarray
    max_trade_notional_dec: np.ndarray
    qty_step: np.ndarray              # (B,) per-instrument notional rounding step
    min_qty_trade: np.ndarray         # (B,) per-instrument min trade size
    # Meta
    maturity: np.ndarray              # (T, B) or (B,)
    issuer_bucket: np.ndarray         # (B,)
    maturity_bucket: np.ndarray       # (T, B) or (B,)
    # Bucket caps
    issuer_dv01_caps: np.ndarray      # (n_issuers,)
    mat_bucket_dv01_caps: np.ndarray  # (n_buckets,)
    # Hedge L2
    hedge_bid_px: np.ndarray
    hedge_bid_sz: np.ndarray
    hedge_ask_px: np.ndarray
    hedge_ask_sz: np.ndarray
    hedge_mid_px: np.ndarray
    hedge_dv01: np.ndarray
    # Hedge ratios
    hedge_ratios: np.ndarray
    # Hedge meta
    hedge_qty_step: np.ndarray        # (H,) per-hedge notional rounding step
    hedge_min_qty_trade: np.ndarray   # (H,) per-hedge min trade size
    # Context
    datetimes: np.ndarray
    instrument_ids: list[str]
    hedge_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return dict suitable for ``run_backtest(**md.to_dict())``."""
        return {
            "bid_px": self.bid_px,
            "bid_sz": self.bid_sz,
            "ask_px": self.ask_px,
            "ask_sz": self.ask_sz,
            "mid_px": self.mid_px,
            "dv01": self.dv01,
            "fair_price": self.fair_price,
            "zscore": self.zscore,
            "adf_p_value": self.adf_p_value,
            "tradable": self.tradable,
            "pos_limits_long": self.pos_limits_long,
            "pos_limits_short": self.pos_limits_short,
            "max_trade_notional_inc": self.max_trade_notional_inc,
            "max_trade_notional_dec": self.max_trade_notional_dec,
            "qty_step": self.qty_step,
            "min_qty_trade": self.min_qty_trade,
            "maturity": self.maturity,
            "issuer_bucket": self.issuer_bucket,
            "maturity_bucket": self.maturity_bucket,
            "issuer_dv01_caps": self.issuer_dv01_caps,
            "mat_bucket_dv01_caps": self.mat_bucket_dv01_caps,
            "hedge_bid_px": self.hedge_bid_px,
            "hedge_bid_sz": self.hedge_bid_sz,
            "hedge_ask_px": self.hedge_ask_px,
            "hedge_ask_sz": self.hedge_ask_sz,
            "hedge_mid_px": self.hedge_mid_px,
            "hedge_dv01": self.hedge_dv01,
            "hedge_ratios": self.hedge_ratios,
            "hedge_qty_step": self.hedge_qty_step,
            "hedge_min_qty_trade": self.hedge_min_qty_trade,
            "datetimes": self.datetimes,
            "instrument_ids": self.instrument_ids,
            "hedge_ids": self.hedge_ids,
        }


@dataclass
class PortfolioDataLoader:
    """Loads and aligns parquet market data for the portfolio backtester.

    All instruments (trading + hedge) are stored in the same book/mid/dv01
    files.  The loader pivots them separately for instrument_ids and
    hedge_ids.

    Parameters
    ----------
    book_filename .. hedge_ratio_filename : str
        Filenames within *data_path*.
    data_path : str, Path, or None
        Directory containing the parquet files.  Can be omitted here
        and passed to :meth:`load` instead.
    datetime_col : str
        Name of the datetime column in parquets (default ``"datetime"``).
    instrument_col : str
        Name of the instrument ID column (default ``"instrument_id"``).
    fill_method : str
        ``"ffill"`` (forward-fill then drop leading NaN rows) or
        ``"drop"`` (keep only rows where all sources have data).
    """

    # data files — single set for all instruments (trading + hedge)
    book_filename: str
    mid_filename: str
    dv01_filename: str
    # Signal file
    signal_filename: str
    # Static meta file
    meta_filename: str
    # Hedge ratios file
    hedge_ratio_filename: str
    # Hedge meta file (static per-hedge metadata)
    hedge_meta_filename: str
    # Config
    data_path: str | Path | None = None
    datetime_col: str = "datetime"
    instrument_col: str = "instrument_id"
    fill_method: str = "ffill"
    close_time: str | None = None

    def _resolve_file(
        self,
        data_path_dir: Path,
        base_filename: str,
    ) -> Path:
        """Resolve a filename — if it ends with ``.parquet``, use as-is.

        Otherwise treat as a base name and look for a single file (backward
        compat: ``book.parquet`` still works).
        """
        if base_filename.endswith(".parquet"):
            return data_path_dir / base_filename
        # Try exact match first
        exact = data_path_dir / f"{base_filename}.parquet"
        if exact.exists():
            return exact
        # Must be date-suffixed — caller should use _discover_data_files
        raise FileNotFoundError(
            f"No single file found for base name {base_filename!r} in {data_path_dir}. "
            f"Use load_chunked() for date-suffixed files."
        )

    def load(
        self,
        instrument_ids: list[str] | None = None,
        hedge_ids: list[str] | None = None,
        data_path: str | Path | None = None,
        maturity_bucket_bins: tuple[float, ...] = (),
        issuer_dv01_caps_map: dict[str, float] | None = None,
        mat_bucket_dv01_caps: tuple[float, ...] | None = None,
    ) -> PortfolioMarketData:
        """Read parquets, align, pivot, and return ``PortfolioMarketData``.

        Parameters
        ----------
        instrument_ids : list[str] or None
            Ordered list of instrument IDs — defines B and column order.
            When *None*, auto-detected from the meta file.
        hedge_ids : list[str] or None
            Ordered list of hedge instrument IDs — defines H.
            When *None*, auto-detected from the hedge_ratios file.
        data_path : str, Path, or None
            Overrides ``self.data_path`` when supplied.
        maturity_bucket_bins : tuple[float, ...]
            Bin edges for ``np.digitize`` to compute time-varying
            maturity_bucket.  Empty to disable.
        issuer_dv01_caps_map : dict[str, float] or None
            Mapping from issuer name → max absolute DV01 cap.  The keys
            define the canonical issuer ordering and are used to convert
            string ``issuer_bucket`` values in meta to integer indices.
        mat_bucket_dv01_caps : tuple[float, ...] or None
            Per-bucket DV01 cap indexed by ``np.digitize`` bucket index.
            Length must equal ``len(maturity_bucket_bins) + 1``.
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
        logger.info("Loading single-file market data from %s", data_path_dir)

        # --- 1. Read parquets ---------------------------------------------------
        book_df = pd.read_parquet(self._resolve_file(data_path_dir, self.book_filename))
        mid_df = pd.read_parquet(self._resolve_file(data_path_dir, self.mid_filename))
        dv01_df = pd.read_parquet(self._resolve_file(data_path_dir, self.dv01_filename))
        signal_df = pd.read_parquet(self._resolve_file(data_path_dir, self.signal_filename))
        meta_df = pd.read_parquet(data_path_dir / self.meta_filename)
        hedge_ratio_df = pd.read_parquet(
            self._resolve_file(data_path_dir, self.hedge_ratio_filename)
        )
        hedge_meta_df = pd.read_parquet(data_path_dir / self.hedge_meta_filename)

        # --- 2. Extract meta arrays --------------------------------------------
        _, static_arrays = _extract_meta_arrays(
            meta_df, inst_col, instrument_ids,
            issuer_dv01_caps_map,
        )
        instrument_ids = static_arrays["instrument_ids"]
        if hedge_ids is None:
            hedge_ids = sorted(hedge_meta_df[inst_col].unique().tolist())

        return _build_market_data(
            book_df=book_df,
            mid_df=mid_df,
            dv01_df=dv01_df,
            signal_df=signal_df,
            hedge_ratio_df=hedge_ratio_df,
            hedge_meta_df=hedge_meta_df,
            static_arrays=static_arrays,
            instrument_ids=instrument_ids,
            hedge_ids=hedge_ids,
            dt_col=dt_col,
            inst_col=inst_col,
            fill_method=self.fill_method,
            maturity_bucket_bins=maturity_bucket_bins,
            mat_bucket_dv01_caps=mat_bucket_dv01_caps,
            close_time=self.close_time,
        )

    def load_chunked(
        self,
        chunk_freq: str,
        start_date: str | None = None,
        end_date: str | None = None,
        instrument_ids: list[str] | None = None,
        hedge_ids: list[str] | None = None,
        data_path: str | Path | None = None,
        maturity_bucket_bins: tuple[float, ...] = (),
        issuer_dv01_caps_map: dict[str, float] | None = None,
        mat_bucket_dv01_caps: tuple[float, ...] | None = None,
    ) -> Iterator[PortfolioMarketData]:
        """Yield ``PortfolioMarketData`` one chunk at a time.

        Parameters
        ----------
        chunk_freq : str
            Pandas frequency string for chunk boundaries (e.g. ``"D"``,
            ``"W"``, ``"MS"``).
        start_date, end_date : str or None
            Date strings like ``"20240101"`` bounding the overall range.
        instrument_ids, hedge_ids : list[str] or None
            As in :meth:`load`.
        data_path : str, Path, or None
            Overrides ``self.data_path`` when supplied.
        maturity_bucket_bins : tuple[float, ...]
            Bin edges for maturity bucket computation.
        issuer_dv01_caps_map : dict[str, float] or None
            Issuer name → DV01 cap mapping.
        mat_bucket_dv01_caps : tuple[float, ...] or None
            Per-bucket DV01 caps.
        """
        resolved = data_path or self.data_path
        if resolved is None:
            raise ValueError(
                "data_path must be provided either at construction time "
                "or as an argument to load_chunked()"
            )
        data_path_dir = Path(resolved)
        dt_col = self.datetime_col
        inst_col = self.instrument_col

        # --- 1. Read meta (static, small) once ---------------------------------
        meta_df = pd.read_parquet(data_path_dir / self.meta_filename)
        hedge_meta_df = pd.read_parquet(data_path_dir / self.hedge_meta_filename)

        # Extract hedge_ids from hedge meta file
        if hedge_ids is None:
            hedge_ids = sorted(hedge_meta_df[inst_col].unique().tolist())

        _, static_arrays = _extract_meta_arrays(
            meta_df, inst_col, instrument_ids,
            issuer_dv01_caps_map,
        )
        instrument_ids = static_arrays["instrument_ids"]

        # --- 2. Discover date-suffixed files for each data type ----------------
        logger.info(
            "Discovering data files in %s (start=%s, end=%s)",
            data_path_dir, start_date, end_date,
        )
        book_files = _discover_data_files(
            data_path_dir, self.book_filename, start_date, end_date,
        )
        mid_files = _discover_data_files(
            data_path_dir, self.mid_filename, start_date, end_date,
        )
        dv01_files = _discover_data_files(
            data_path_dir, self.dv01_filename, start_date, end_date,
        )
        signal_files = _discover_data_files(
            data_path_dir, self.signal_filename, start_date, end_date,
        )
        hedge_ratio_files = _discover_data_files(
            data_path_dir, self.hedge_ratio_filename, start_date, end_date,
        )
        logger.info(
            "Discovered files: book=%d, mid=%d, dv01=%d, signal=%d, hedge=%d",
            len(book_files), len(mid_files), len(dv01_files),
            len(signal_files), len(hedge_ratio_files),
        )

        # --- 2b. Validate cross-data date consistency -------------------------
        all_file_sets = {
            "book": _extract_date_set(book_files, self.book_filename),
            "mid": _extract_date_set(mid_files, self.mid_filename),
            "dv01": _extract_date_set(dv01_files, self.dv01_filename),
            "signal": _extract_date_set(signal_files, self.signal_filename),
            "hedge_ratio": _extract_date_set(hedge_ratio_files, self.hedge_ratio_filename),
        }
        # Only validate sets that have files (non-empty)
        non_empty = {k: v for k, v in all_file_sets.items() if v}
        if non_empty:
            union_dates = set.union(*non_empty.values())
            for name, ds in non_empty.items():
                missing = union_dates - ds
                if missing:
                    raise ValueError(
                        f"Data type {name!r} is missing files for dates: {sorted(missing)}"
                    )

        # --- 2c. Validate column consistency within each data type -------------
        _validate_column_consistency(book_files, "book")
        _validate_column_consistency(mid_files, "mid")
        _validate_column_consistency(dv01_files, "dv01")
        _validate_column_consistency(signal_files, "signal")
        _validate_column_consistency(hedge_ratio_files, "hedge_ratio")

        # --- 3. Generate chunk boundaries --------------------------------------
        sd = pd.Timestamp(start_date) if start_date else None
        ed = pd.Timestamp(end_date) if end_date else None

        if sd is None or ed is None:
            # Infer from book file dates
            all_dates = []
            for p in book_files:
                fs, fe = _parse_file_dates(p.stem, self.book_filename)
                if fs:
                    all_dates.extend([fs, fe])
            if not all_dates:
                raise ValueError("Cannot infer date range from files")
            if sd is None:
                sd = min(all_dates)
            if ed is None:
                ed = max(all_dates)

        chunk_starts = pd.date_range(sd, ed, freq=chunk_freq)
        if len(chunk_starts) == 0:
            chunk_starts = pd.DatetimeIndex([sd])

        # Build chunk boundaries as (start, end) pairs
        chunk_boundaries = []
        for i, cs in enumerate(chunk_starts):
            if i + 1 < len(chunk_starts):
                ce = chunk_starts[i + 1] - pd.Timedelta(nanoseconds=1)
            else:
                ce = ed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            chunk_boundaries.append((cs, ce))

        logger.info(
            "Chunked loading: %d chunk boundaries from %s to %s",
            len(chunk_boundaries), sd, ed,
        )

        # --- 4. Yield one PortfolioMarketData per chunk ------------------------
        n_chunks = len(chunk_boundaries)
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries, 1):
            cs_str = chunk_start.strftime("%Y%m%d")
            ce_str = chunk_end.strftime("%Y%m%d")
            logger.info(
                "Processing chunk %d/%d (%s–%s)",
                chunk_idx, n_chunks, cs_str, ce_str,
            )

            book_df = _read_parquet_date_range(
                _filter_files_for_range(book_files, self.book_filename, cs_str, ce_str),
                dt_col, chunk_start, chunk_end,
            )
            if book_df.empty:
                logger.info("Chunk %s–%s: no book data, skipping", cs_str, ce_str)
                continue

            mid_df = _read_parquet_date_range(
                _filter_files_for_range(mid_files, self.mid_filename, cs_str, ce_str),
                dt_col, chunk_start, chunk_end,
            )
            dv01_df = _read_parquet_date_range(
                _filter_files_for_range(dv01_files, self.dv01_filename, cs_str, ce_str),
                dt_col, chunk_start, chunk_end,
            )
            signal_df = _read_parquet_date_range(
                _filter_files_for_range(signal_files, self.signal_filename, cs_str, ce_str),
                dt_col, chunk_start, chunk_end,
            )
            hedge_ratio_df = _read_parquet_date_range(
                _filter_files_for_range(hedge_ratio_files, self.hedge_ratio_filename, cs_str, ce_str),
                dt_col, chunk_start, chunk_end,
            )

            logger.info(
                "Chunk %s–%s: book=%d rows, mid=%d, dv01=%d, signal=%d, hedge=%d",
                cs_str, ce_str, len(book_df), len(mid_df), len(dv01_df),
                len(signal_df), len(hedge_ratio_df),
            )

            yield _build_market_data(
                book_df=book_df,
                mid_df=mid_df,
                dv01_df=dv01_df,
                signal_df=signal_df,
                hedge_ratio_df=hedge_ratio_df,
                hedge_meta_df=hedge_meta_df,
                static_arrays=static_arrays,
                instrument_ids=instrument_ids,
                hedge_ids=hedge_ids,
                dt_col=dt_col,
                inst_col=inst_col,
                fill_method=self.fill_method,
                maturity_bucket_bins=maturity_bucket_bins,
                mat_bucket_dv01_caps=mat_bucket_dv01_caps,
                close_time=self.close_time,
            )

# ---------------------------------------------------------------------------
# Internal helpers - main
# ---------------------------------------------------------------------------

def _build_market_data(
    *,
    book_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    dv01_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    hedge_ratio_df: pd.DataFrame,
    hedge_meta_df: pd.DataFrame,
    static_arrays: dict[str, Any],
    instrument_ids: list[str],
    hedge_ids: list[str],
    dt_col: str,
    inst_col: str,
    fill_method: str,
    maturity_bucket_bins: tuple[float, ...],
    mat_bucket_dv01_caps: tuple[float, ...] | None,
    close_time: str | None = None,
) -> PortfolioMarketData:
    """Shared pivot + align + reshape logic used by both load() and load_chunked()."""
    B = len(instrument_ids)
    H = len(hedge_ids)

    # Auto-detect book levels
    n_levels = detect_book_levels(book_df.columns)

    # Pivot book/mid/dv01 separately for instruments and hedges
    inst_book_pivoted = pivot_book(
        book_df, dt_col, inst_col, instrument_ids, n_levels,
    )
    inst_mid_pivoted = pivot_simple(
        mid_df, dt_col, inst_col, instrument_ids, "mid_px",
    )
    inst_dv01_pivoted = pivot_simple(
        dv01_df, dt_col, inst_col, instrument_ids, "dv01",
    )

    hedge_book_pivoted = pivot_book(
        book_df, dt_col, inst_col, hedge_ids, n_levels,
    )
    hedge_mid_pivoted = pivot_simple(
        mid_df, dt_col, inst_col, hedge_ids, "mid_px",
    )
    hedge_dv01_pivoted = pivot_simple(
        dv01_df, dt_col, inst_col, hedge_ids, "dv01",
    )

    # Pivot signals
    fair_pivoted = pivot_simple(
        signal_df, dt_col, inst_col, instrument_ids, "fair_price",
    )
    zscore_pivoted = pivot_simple(
        signal_df, dt_col, inst_col, instrument_ids, "zscore",
    )
    adf_pivoted = pivot_simple(
        signal_df, dt_col, inst_col, instrument_ids, "adf_p_value",
    )

    # Pivot hedge ratios
    hedge_ratios_pivoted = _pivot_hedge_ratios_portfolio(
        hedge_ratio_df, inst_col, dt_col, instrument_ids, hedge_ids,
    )

    # Align all sources on common datetime index
    sources = {
        "inst_book": inst_book_pivoted,
        "inst_mid": inst_mid_pivoted,
        "inst_dv01": inst_dv01_pivoted,
        "fair_price": fair_pivoted,
        "zscore": zscore_pivoted,
        "adf": adf_pivoted,
        "hedge_book": hedge_book_pivoted,
        "hedge_mid": hedge_mid_pivoted,
        "hedge_dv01": hedge_dv01_pivoted,
        "hedge_ratios": hedge_ratios_pivoted,
    }

    sources, all_dts_idx = align_and_fill(
        sources,
        fill_method=fill_method,
        essential_keys=("inst_mid", "hedge_mid", "inst_dv01"),
        datetime_source_keys=("inst_book", ),
        no_ffill_keys=("inst_book", "hedge_book"),
        close_time=close_time,
    )

    T = len(all_dts_idx)

    # Extract numpy arrays
    bid_px, bid_sz, ask_px, ask_sz = reshape_book(
        sources["inst_book"], instrument_ids, n_levels,
    )
    hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz = reshape_book(
        sources["hedge_book"], hedge_ids, n_levels,
    )

    mid_px = sources["inst_mid"].values.astype(np.float64)
    dv01 = sources["inst_dv01"].values.astype(np.float64)
    fair_price = sources["fair_price"].values.astype(np.float64)
    zscore = sources["zscore"].values.astype(np.float64)
    adf_p_value = sources["adf"].values.astype(np.float64)

    hedge_mid_px = sources["hedge_mid"].values.astype(np.float64)
    hedge_dv01_arr = sources["hedge_dv01"].values.astype(np.float64)

    hedge_ratios_arr = sources["hedge_ratios"].values.astype(np.float64)
    hedge_ratios_arr = hedge_ratios_arr.reshape(T, B, H)

    datetimes = all_dts_idx.values

    # Unpack static arrays (copy tradable — may be mutated per-chunk)
    tradable = static_arrays["tradable"].copy()
    pos_limits_long = static_arrays["pos_limits_long"]
    pos_limits_short = static_arrays["pos_limits_short"]
    max_trade_notional_inc = static_arrays["max_trade_notional_inc"]
    max_trade_notional_dec = static_arrays["max_trade_notional_dec"]
    qty_step_arr = static_arrays["qty_step"]
    min_qty_trade_arr = static_arrays["min_qty_trade"]
    issuer_bucket = static_arrays["issuer_bucket"]
    _issuer_dv01_caps = static_arrays["issuer_dv01_caps"]
    maturity_dates_raw = static_arrays["maturity_dates_raw"]

    # Compute time-varying maturity from maturity_date
    dt_ns = datetimes.astype("datetime64[ns]")
    delta = maturity_dates_raw[np.newaxis, :] - dt_ns[:, np.newaxis]
    maturity = delta.astype("timedelta64[D]").astype(np.float64) / 365.25
    if len(maturity_bucket_bins) > 0:
        bins = np.asarray(maturity_bucket_bins, dtype=np.float64)
        maturity_bucket = np.digitize(maturity, bins).astype(np.int64)
    else:
        maturity_bucket = np.zeros(B, dtype=np.int64)

    # Disable tradable for instruments with all-NaN mid_px or signals
    _disable_allnan_instruments(
        tradable, mid_px, fair_price, zscore, adf_p_value, instrument_ids,
    )

    # Validate shapes, warn NaNs
    _validate_shapes(
        bid_px, bid_sz, ask_px, ask_sz, mid_px, dv01,
        fair_price, zscore, adf_p_value,
        tradable, pos_limits_long, pos_limits_short,
        hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz,
        hedge_mid_px, hedge_dv01_arr, hedge_ratios_arr,
        T, B, n_levels, H, n_levels,
    )
    warn_nans(
        bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz,
        mid_px=mid_px, dv01=dv01, fair_price=fair_price,
        zscore=zscore, adf_p_value=adf_p_value,
        hedge_bid_px=hedge_bid_px, hedge_bid_sz=hedge_bid_sz,
        hedge_ask_px=hedge_ask_px, hedge_ask_sz=hedge_ask_sz,
        hedge_mid_px=hedge_mid_px, hedge_dv01=hedge_dv01_arr,
        hedge_ratios=hedge_ratios_arr,
    )

    _mat_bucket_dv01_caps = (
        np.asarray(mat_bucket_dv01_caps, dtype=np.float64)
        if mat_bucket_dv01_caps
        else np.empty(0, dtype=np.float64)
    )

    # Extract hedge_qty_step and hedge_min_qty_trade from hedge_meta
    hedge_qty_step, hedge_min_qty_trade = _extract_hedge_meta_arrays(
        hedge_meta_df, inst_col, hedge_ids,
    )

    return PortfolioMarketData(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        mid_px=mid_px,
        dv01=dv01,
        fair_price=fair_price,
        zscore=zscore,
        adf_p_value=adf_p_value,
        tradable=tradable,
        pos_limits_long=pos_limits_long,
        pos_limits_short=pos_limits_short,
        max_trade_notional_inc=max_trade_notional_inc,
        max_trade_notional_dec=max_trade_notional_dec,
        qty_step=qty_step_arr,
        min_qty_trade=min_qty_trade_arr,
        maturity=maturity,
        issuer_bucket=issuer_bucket,
        maturity_bucket=maturity_bucket,
        issuer_dv01_caps=_issuer_dv01_caps,
        mat_bucket_dv01_caps=_mat_bucket_dv01_caps,
        hedge_bid_px=hedge_bid_px,
        hedge_bid_sz=hedge_bid_sz,
        hedge_ask_px=hedge_ask_px,
        hedge_ask_sz=hedge_ask_sz,
        hedge_mid_px=hedge_mid_px,
        hedge_dv01=hedge_dv01_arr,
        hedge_ratios=hedge_ratios_arr,
        hedge_qty_step=hedge_qty_step,
        hedge_min_qty_trade=hedge_min_qty_trade,
        datetimes=datetimes,
        instrument_ids=instrument_ids,
        hedge_ids=hedge_ids,
    )



def _validate_shapes(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    dv01: np.ndarray,
    fair_price: np.ndarray,
    zscore: np.ndarray,
    adf_p_value: np.ndarray,
    tradable: np.ndarray,
    pos_limits_long: np.ndarray,
    pos_limits_short: np.ndarray,
    hedge_bid_px: np.ndarray,
    hedge_bid_sz: np.ndarray,
    hedge_ask_px: np.ndarray,
    hedge_ask_sz: np.ndarray,
    hedge_mid_px: np.ndarray,
    hedge_dv01: np.ndarray,
    hedge_ratios: np.ndarray,
    T: int,
    B: int,
    L: int,
    H: int,
    L_h: int,
) -> None:
    """Validate that all arrays have consistent shapes."""
    expected = {
        "bid_px": (T, B, L),
        "bid_sz": (T, B, L),
        "ask_px": (T, B, L),
        "ask_sz": (T, B, L),
        "mid_px": (T, B),
        "dv01": (T, B),
        "fair_price": (T, B),
        "zscore": (T, B),
        "adf_p_value": (T, B),
        "tradable": (B,),
        "pos_limits_long": (B,),
        "pos_limits_short": (B,),
        "hedge_bid_px": (T, H, L_h),
        "hedge_bid_sz": (T, H, L_h),
        "hedge_ask_px": (T, H, L_h),
        "hedge_ask_sz": (T, H, L_h),
        "hedge_mid_px": (T, H),
        "hedge_dv01": (T, H),
        "hedge_ratios": (T, B, H),
    }
    arrays = {
        "bid_px": bid_px,
        "bid_sz": bid_sz,
        "ask_px": ask_px,
        "ask_sz": ask_sz,
        "mid_px": mid_px,
        "dv01": dv01,
        "fair_price": fair_price,
        "zscore": zscore,
        "adf_p_value": adf_p_value,
        "tradable": tradable,
        "pos_limits_long": pos_limits_long,
        "pos_limits_short": pos_limits_short,
        "hedge_bid_px": hedge_bid_px,
        "hedge_bid_sz": hedge_bid_sz,
        "hedge_ask_px": hedge_ask_px,
        "hedge_ask_sz": hedge_ask_sz,
        "hedge_mid_px": hedge_mid_px,
        "hedge_dv01": hedge_dv01,
        "hedge_ratios": hedge_ratios,
    }
    for name, arr in arrays.items():
        if arr.shape != expected[name]:
            raise ValueError(
                f"{name} shape mismatch: expected {expected[name]}, got {arr.shape}"
            )


# ---------------------------------------------------------------------------
# Internal helpers - files
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"^(.+?)_(\d{8})_(\d{8})$")


def _extract_date_set(files: list[Path], base_name: str) -> set[tuple[str, str]]:
    """Extract set of (start_date, end_date) tuples from discovered files."""
    dates = set()
    for p in files:
        fs, fe = _parse_file_dates(p.stem, base_name)
        if fs is not None:
            dates.add((fs.strftime("%Y%m%d"), fe.strftime("%Y%m%d")))
    return dates


def _validate_column_consistency(files: list[Path], data_name: str) -> None:
    """Validate all files for a data type have the same columns."""
    if len(files) <= 1:
        return
    ref_cols = set(pq.read_schema(files[0]).names)
    for p in files[1:]:
        cols = set(pq.read_schema(p).names)
        if cols != ref_cols:
            extra = cols - ref_cols
            missing = ref_cols - cols
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"extra {sorted(extra)}")
            raise ValueError(
                f"{data_name}: columns in {p.name} differ from {files[0].name}: "
                f"{', '.join(parts)}"
            )


def _parse_file_dates(stem: str, base_name: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Parse start/end dates from a file stem like ``book_20240101_20240201``."""
    # Strip .parquet suffix from base_name if present
    base = base_name.replace(".parquet", "")
    m = _DATE_PATTERN.match(stem)
    if m and m.group(1) == base:
        return pd.Timestamp(m.group(2)), pd.Timestamp(m.group(3))
    return None, None


def _discover_data_files(
    data_path: Path,
    base_name: str,
    start_date: str | None,
    end_date: str | None,
) -> list[Path]:
    """Find all files matching ``{base_name}_{YYYYMMDD}_{YYYYMMDD}.parquet``
    that overlap with ``[start_date, end_date]``.  Sorted by file start date.

    If ``base_name`` ends with ``.parquet``, returns ``[data_path / base_name]``
    if it exists (single-file mode).
    """
    if base_name.endswith(".parquet"):
        p = data_path / base_name
        return [p] if p.exists() else []

    base = base_name.replace(".parquet", "")
    pattern = f"{base}_*.parquet"
    candidates = sorted(data_path.glob(pattern))

    sd = pd.Timestamp(start_date) if start_date else None
    ed = pd.Timestamp(end_date) if end_date else None

    result = []
    for p in candidates:
        fs, fe = _parse_file_dates(p.stem, base_name)
        if fs is None:
            continue
        # Check overlap: file range [fs, fe] overlaps [sd, ed]
        if sd is not None and fe < sd:
            continue
        if ed is not None and fs > ed:
            continue
        result.append(p)

    result.sort(key=lambda p: _parse_file_dates(p.stem, base_name)[0])
    return result


def _filter_files_for_range(
    files: list[Path],
    base_name: str,
    start_date: str,
    end_date: str,
) -> list[Path]:
    """Filter already-discovered files to those overlapping [start_date, end_date]."""
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date)
    result = []
    for p in files:
        fs, fe = _parse_file_dates(p.stem, base_name)
        if fs is None:
            # Single file (no date suffix) — always include
            result.append(p)
            continue
        if fe < sd or fs > ed:
            continue
        result.append(p)
    return result


def _read_parquet_date_range(
    paths: list[Path],
    dt_col: str,
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> pd.DataFrame:
    """Read and concatenate parquets, filter to ``[start_dt, end_dt]``."""
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    if dt_col in df.columns:
        if start_dt is not None:
            df = df[df[dt_col] >= start_dt]
        if end_dt is not None:
            df = df[df[dt_col] <= end_dt]
    return df


# ---------------------------------------------------------------------------
# Internal helpers - meta
# ---------------------------------------------------------------------------
def _extract_meta_arrays(
    meta_df: pd.DataFrame,
    inst_col: str,
    instrument_ids: list[str] | None,
    issuer_dv01_caps_map: dict[str, float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract static arrays from meta DataFrame.

    Returns (meta_indexed, static_arrays) where static_arrays contains
    tradable, pos_limits_long, pos_limits_short, issuer_bucket,
    issuer_dv01_caps, maturity_dates_raw, maturity, maturity_bucket,
    and instrument_ids.
    """
    if instrument_ids is None:
        instrument_ids = sorted(meta_df[inst_col].unique().tolist())
        logger.info("Auto-detected instrument_ids from meta: %s", instrument_ids)

    B = len(instrument_ids)
    meta_indexed = meta_df.set_index(inst_col).reindex(instrument_ids)

    # Validate all required columns
    missing = [c for c in META_REQUIRED_COLUMNS if c not in meta_indexed.columns]
    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found in meta file"
        )

    tradable = meta_indexed["tradable"].values.astype(np.float64)
    pos_limits_long = meta_indexed["pos_limit_long"].values.astype(np.float64)
    pos_limits_short = meta_indexed["pos_limit_short"].values.astype(np.float64)
    max_trade_notional_inc = meta_indexed["max_trade_notional_inc"].values.astype(np.float64)
    max_trade_notional_dec = meta_indexed["max_trade_notional_dec"].values.astype(np.float64)
    qty_step = meta_indexed["qty_step"].values.astype(np.float64)
    min_qty_trade = meta_indexed["min_qty_trade"].values.astype(np.float64)

    # Issuer bucket
    if "issuer_bucket" in meta_indexed.columns and issuer_dv01_caps_map:
        issuer_names = list(issuer_dv01_caps_map.keys())
        issuer_name_to_idx = {name: idx for idx, name in enumerate(issuer_names)}
        raw_issuers = meta_indexed["issuer_bucket"].values
        issuer_bucket = np.array(
            [issuer_name_to_idx.get(str(v), 0) for v in raw_issuers],
            dtype=np.int64,
        )
        issuer_dv01_caps = np.array(
            [issuer_dv01_caps_map[name] for name in issuer_names],
            dtype=np.float64,
        )
    elif "issuer_bucket" in meta_indexed.columns:
        issuer_bucket = meta_indexed["issuer_bucket"].values.astype(np.int64)
        issuer_dv01_caps = np.empty(0, dtype=np.float64)
    else:
        issuer_bucket = np.zeros(B, dtype=np.int64)
        issuer_dv01_caps = np.empty(0, dtype=np.float64)

    # Maturity (validated above via META_REQUIRED_COLUMNS)
    maturity_dates_raw = pd.to_datetime(
        meta_indexed["maturity_date"]
    ).values.astype("datetime64[ns]")

    static_arrays = {
        "instrument_ids": instrument_ids,
        "tradable": tradable,
        "pos_limits_long": pos_limits_long,
        "pos_limits_short": pos_limits_short,
        "max_trade_notional_inc": max_trade_notional_inc,
        "max_trade_notional_dec": max_trade_notional_dec,
        "qty_step": qty_step,
        "min_qty_trade": min_qty_trade,
        "issuer_bucket": issuer_bucket,
        "issuer_dv01_caps": issuer_dv01_caps,
        "maturity_dates_raw": maturity_dates_raw,
    }

    return meta_indexed, static_arrays
def _disable_allnan_instruments(
    tradable: np.ndarray,
    mid_px: np.ndarray,
    fair_price: np.ndarray,
    zscore: np.ndarray,
    adf_p_value: np.ndarray,
    instrument_ids: list[str],
) -> None:
    """Set tradable=0 for instruments whose mid_px or signals are all NaN.

    Modifies *tradable* in place.  Logs a warning for each disabled instrument.
    """
    checks = {
        "mid_px": mid_px,
        "fair_price": fair_price,
        "zscore": zscore,
        "adf_p_value": adf_p_value,
    }
    B = len(instrument_ids)
    for b in range(B):
        if tradable[b] == 0.0:
            continue
        for name, arr in checks.items():
            if np.all(np.isnan(arr[:, b])):
                logger.warning(
                    "Instrument %s has all-NaN %s in this chunk — "
                    "setting tradable=False",
                    instrument_ids[b],
                    name,
                )
                tradable[b] = 0.0
                break


# ---------------------------------------------------------------------------
# Internal helpers - hedge meta
# ---------------------------------------------------------------------------
def _extract_hedge_meta_arrays(
    hedge_meta_df: pd.DataFrame,
    inst_col: str,
    hedge_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-hedge static arrays from hedge meta DataFrame.

    Returns (hedge_qty_step, hedge_min_qty_trade) arrays, each (H,).
    """
    hedge_meta_indexed = hedge_meta_df.set_index(inst_col).reindex(hedge_ids)
    missing = [c for c in HEDGE_META_REQUIRED_COLUMNS if c not in hedge_meta_indexed.columns]
    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found in hedge_meta file"
        )
    hedge_qty_step = hedge_meta_indexed["qty_step"].values.astype(np.float64)
    hedge_min_qty_trade = hedge_meta_indexed["min_qty_trade"].values.astype(np.float64)
    return hedge_qty_step, hedge_min_qty_trade




# ---------------------------------------------------------------------------
# Internal helpers - hedge ratio
# ---------------------------------------------------------------------------
def _pivot_hedge_ratios_portfolio(
    hedge_df: pd.DataFrame,
    inst_col: str,
    dt_col: str,
    instrument_ids: list[str],
    hedge_ids: list[str],
) -> pd.DataFrame:
    """Pivot list-column hedge ratios into a wide ``(T, B*H)`` DataFrame.

    The parquet has columns ``[datetime, instrument_id, hedge_instruments,
    hedge_ratios]`` where ``hedge_instruments`` and ``hedge_ratios`` are
    lists.  Each instrument has its own hedge ratios per timestamp.

    Returns a DataFrame indexed by datetime with B*H columns (one per
    instrument-hedge pair), ordered as inst_0_hedge_0, inst_0_hedge_1, ...,
    inst_1_hedge_0, ...
    """
    B = len(instrument_ids)
    H = len(hedge_ids)
    inst_to_idx = {b: i for i, b in enumerate(instrument_ids)}
    hedge_to_idx = {h: j for j, h in enumerate(hedge_ids)}

    # Group by datetime
    grouped = hedge_df.sort_values(dt_col).groupby(dt_col)

    datetimes = []
    ratio_rows = []

    for dt_val, group in grouped:
        ratios = np.zeros(B * H, dtype=np.float64)
        for _, row in group.iterrows():
            inst_id = row[inst_col]
            inst_idx = inst_to_idx.get(inst_id)
            if inst_idx is None:
                continue

            hi_list = list(row["hedge_instruments"])
            hr_list = list(row["hedge_ratios"])

            if len(hi_list) != len(hr_list):
                raise ValueError(
                    f"hedge_instruments length ({len(hi_list)}) != "
                    f"hedge_ratios length ({len(hr_list)}) "
                    f"at {dt_val}, instrument={inst_id}"
                )

            for hi, hr in zip(hi_list, hr_list):
                h_idx = hedge_to_idx.get(hi)
                if h_idx is not None:
                    ratios[inst_idx * H + h_idx] = hr

        datetimes.append(dt_val)
        ratio_rows.append(ratios)

    columns = [
        f"{instrument_ids[i]}__{hedge_ids[j]}"
        for i in range(B)
        for j in range(H)
    ]

    result = pd.DataFrame(
        np.array(ratio_rows),
        index=pd.DatetimeIndex(datetimes),
        columns=columns,
    )
    result.index.name = dt_col
    return result
