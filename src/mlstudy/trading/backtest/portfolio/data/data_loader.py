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
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.common.data.helpers import (
    align_and_fill,
    detect_book_levels,
    extract_hedge_ids,
    pivot_book,
    pivot_simple,
    reshape_book,
    warn_nans,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioMarketData:
    """Container for aligned portfolio market data arrays.

    Shapes
    ------
    bid_px, bid_sz, ask_px, ask_sz : (T, B, L)
    mid_px, dv01 : (T, B)
    fair_price, zscore, adf_p_value : (T, B)
    tradable, pos_limits_long, pos_limits_short : (B,)
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
            "datetimes": self.datetimes,
            "instrument_ids": self.instrument_ids,
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
    # Config
    data_path: str | Path | None = None
    datetime_col: str = "datetime"
    instrument_col: str = "instrument_id"
    fill_method: str = "ffill"

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

        # --- 2. Extract meta arrays --------------------------------------------
        meta_indexed, static_arrays = _extract_meta_arrays(
            meta_df, inst_col, instrument_ids, hedge_ratio_df,
            issuer_dv01_caps_map, maturity_bucket_bins,
        )
        instrument_ids = static_arrays["instrument_ids"]
        hedge_ids = static_arrays.get("hedge_ids") if hedge_ids is None else hedge_ids
        if hedge_ids is None:
            hedge_ids = extract_hedge_ids(hedge_ratio_df)

        return _build_market_data(
            book_df=book_df,
            mid_df=mid_df,
            dv01_df=dv01_df,
            signal_df=signal_df,
            hedge_ratio_df=hedge_ratio_df,
            meta_indexed=meta_indexed,
            static_arrays=static_arrays,
            instrument_ids=instrument_ids,
            hedge_ids=hedge_ids,
            dt_col=dt_col,
            inst_col=inst_col,
            fill_method=self.fill_method,
            maturity_bucket_bins=maturity_bucket_bins,
            mat_bucket_dv01_caps=mat_bucket_dv01_caps,
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

        # We need hedge_ids for meta extraction — read one hedge file to detect
        if hedge_ids is None:
            hedge_files = _discover_data_files(
                data_path_dir, self.hedge_ratio_filename, start_date, end_date,
            )
            if hedge_files:
                sample_hr = pd.read_parquet(hedge_files[0])
                hedge_ids = extract_hedge_ids(sample_hr)
            else:
                hedge_ids = []

        meta_indexed, static_arrays = _extract_meta_arrays(
            meta_df, inst_col, instrument_ids, None,
            issuer_dv01_caps_map, maturity_bucket_bins,
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
        chunks_yielded = 0
        for chunk_start, chunk_end in chunk_boundaries:
            cs_str = chunk_start.strftime("%Y%m%d")
            ce_str = chunk_end.strftime("%Y%m%d")

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

            if book_df.empty:
                continue

            logger.info(
                "Chunk %s–%s: book=%d rows, mid=%d, dv01=%d, signal=%d, hedge=%d",
                cs_str, ce_str, len(book_df), len(mid_df), len(dv01_df),
                len(signal_df), len(hedge_ratio_df),
            )
            chunks_yielded += 1

            yield _build_market_data(
                book_df=book_df,
                mid_df=mid_df,
                dv01_df=dv01_df,
                signal_df=signal_df,
                hedge_ratio_df=hedge_ratio_df,
                meta_indexed=meta_indexed,
                static_arrays=static_arrays,
                instrument_ids=instrument_ids,
                hedge_ids=hedge_ids,
                dt_col=dt_col,
                inst_col=inst_col,
                fill_method=self.fill_method,
                maturity_bucket_bins=maturity_bucket_bins,
                mat_bucket_dv01_caps=mat_bucket_dv01_caps,
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(r"^(.+?)_(\d{8})_(\d{8})$")


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


def _extract_meta_arrays(
    meta_df: pd.DataFrame,
    inst_col: str,
    instrument_ids: list[str] | None,
    hedge_ratio_df: pd.DataFrame | None,
    issuer_dv01_caps_map: dict[str, float] | None,
    maturity_bucket_bins: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract static arrays from meta DataFrame.

    Returns (meta_indexed, static_arrays) where static_arrays contains
    tradable, pos_limits_long, pos_limits_short, issuer_bucket,
    issuer_dv01_caps, maturity_dates_raw, maturity, maturity_bucket,
    instrument_ids, and optionally hedge_ids.
    """
    if instrument_ids is None:
        instrument_ids = sorted(meta_df[inst_col].unique().tolist())
        logger.info("Auto-detected instrument_ids from meta: %s", instrument_ids)

    B = len(instrument_ids)
    meta_indexed = meta_df.set_index(inst_col).reindex(instrument_ids)

    tradable = meta_indexed["tradable"].values.astype(np.float64)
    pos_limits_long = meta_indexed["pos_limit_long"].values.astype(np.float64)
    pos_limits_short = meta_indexed["pos_limit_short"].values.astype(np.float64)

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

    # Maturity
    maturity_dates_raw = None
    if "maturity_date" in meta_indexed.columns:
        maturity_dates_raw = pd.to_datetime(
            meta_indexed["maturity_date"]
        ).values.astype("datetime64[ns]")
        maturity = np.zeros(B, dtype=np.float64)
        maturity_bucket = np.zeros(B, dtype=np.int64)
    else:
        # maturity = (
        #     meta_indexed["maturity"].values.astype(np.float64)
        #     if "maturity" in meta_indexed.columns
        #     else np.zeros(B, dtype=np.float64)
        # )
        # maturity_bucket = (
        #     meta_indexed["maturity_bucket"].values.astype(np.int64)
        #     if "maturity_bucket" in meta_indexed.columns
        #     else np.zeros(B, dtype=np.int64)
        # )
        raise RuntimeError("maturity_date not in meta data")

    static_arrays = {
        "instrument_ids": instrument_ids,
        "tradable": tradable,
        "pos_limits_long": pos_limits_long,
        "pos_limits_short": pos_limits_short,
        "issuer_bucket": issuer_bucket,
        "issuer_dv01_caps": issuer_dv01_caps,
        "maturity_dates_raw": maturity_dates_raw,
        "maturity": maturity,
        "maturity_bucket": maturity_bucket,
    }

    if hedge_ratio_df is not None:
        static_arrays["hedge_ids"] = extract_hedge_ids(hedge_ratio_df)

    return meta_indexed, static_arrays


def _build_market_data(
    *,
    book_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    dv01_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    hedge_ratio_df: pd.DataFrame,
    meta_indexed: pd.DataFrame,
    static_arrays: dict[str, Any],
    instrument_ids: list[str],
    hedge_ids: list[str],
    dt_col: str,
    inst_col: str,
    fill_method: str,
    maturity_bucket_bins: tuple[float, ...],
    mat_bucket_dv01_caps: tuple[float, ...] | None,
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
        "fair": fair_pivoted,
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
        essential_keys=("inst_book", ),
        datetime_source_keys=("inst_book", ),
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
    fair_price = sources["fair"].values.astype(np.float64)
    zscore = sources["zscore"].values.astype(np.float64)
    adf_p_value = sources["adf"].values.astype(np.float64)

    hedge_mid_px = sources["hedge_mid"].values.astype(np.float64)
    hedge_dv01_arr = sources["hedge_dv01"].values.astype(np.float64)

    hedge_ratios_arr = sources["hedge_ratios"].values.astype(np.float64)
    hedge_ratios_arr = hedge_ratios_arr.reshape(T, B, H)

    datetimes = all_dts_idx.values

    # Unpack static arrays
    tradable = static_arrays["tradable"]
    pos_limits_long = static_arrays["pos_limits_long"]
    pos_limits_short = static_arrays["pos_limits_short"]
    issuer_bucket = static_arrays["issuer_bucket"]
    _issuer_dv01_caps = static_arrays["issuer_dv01_caps"]
    _maturity_dates_raw = static_arrays["maturity_dates_raw"]
    maturity = static_arrays["maturity"]
    maturity_bucket = static_arrays["maturity_bucket"]

    # Compute time-varying maturity if maturity_date available
    if _maturity_dates_raw is not None:
        dt_ns = datetimes.astype("datetime64[ns]")
        delta = _maturity_dates_raw[np.newaxis, :] - dt_ns[:, np.newaxis]
        maturity = delta.astype("timedelta64[D]").astype(np.float64) / 365.25
        if len(maturity_bucket_bins) > 0:
            bins = np.asarray(maturity_bucket_bins, dtype=np.float64)
            maturity_bucket = np.digitize(maturity, bins).astype(np.int64)
        else:
            maturity_bucket = (
                meta_indexed["maturity_bucket"].values.astype(np.int64)
                if "maturity_bucket" in meta_indexed.columns
                else np.zeros(B, dtype=np.int64)
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
        datetimes=datetimes,
        instrument_ids=instrument_ids,
        hedge_ids=hedge_ids,
    )


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
