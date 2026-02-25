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
import warnings
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

        # --- 1. Read parquets ---------------------------------------------------
        book_df = pd.read_parquet(data_path_dir / self.book_filename)
        mid_df = pd.read_parquet(data_path_dir / self.mid_filename)
        dv01_df = pd.read_parquet(data_path_dir / self.dv01_filename)
        signal_df = pd.read_parquet(data_path_dir / self.signal_filename)
        meta_df = pd.read_parquet(data_path_dir / self.meta_filename)
        hedge_ratio_df = pd.read_parquet(data_path_dir / self.hedge_ratio_filename)

        # --- 2. Auto-detect instrument_ids from meta file if not provided -------
        if instrument_ids is None:
            instrument_ids = sorted(meta_df[inst_col].unique().tolist())
            logger.info("Auto-detected instrument_ids from meta: %s", instrument_ids)

        # --- 3. Auto-detect hedge_ids from hedge_ratios file --------------------
        if hedge_ids is None:
            hedge_ids = extract_hedge_ids(hedge_ratio_df)
            logger.info("Auto-detected hedge_ids from hedge_ratios: %s", hedge_ids)

        B = len(instrument_ids)
        H = len(hedge_ids)

        # --- 4. Auto-detect book levels (one detection — same file) -------------
        n_levels = detect_book_levels(book_df.columns)
        logger.info("Detected %d book levels", n_levels)

        # --- 5. Pivot book/mid/dv01 separately for instruments and hedges -------
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

        # --- 6. Pivot signals to wide: each becomes (T, B) ---------------------
        fair_pivoted = pivot_simple(
            signal_df, dt_col, inst_col, instrument_ids, "fair_price",
        )
        zscore_pivoted = pivot_simple(
            signal_df, dt_col, inst_col, instrument_ids, "zscore",
        )
        adf_pivoted = pivot_simple(
            signal_df, dt_col, inst_col, instrument_ids, "adf_p_value",
        )

        # --- 7. Load static meta → per-instrument arrays -----------------------
        meta_indexed = meta_df.set_index(inst_col).reindex(instrument_ids)
        tradable = meta_indexed["tradable"].values.astype(np.float64)
        pos_limits_long = meta_indexed["pos_limit_long"].values.astype(np.float64)
        pos_limits_short = meta_indexed["pos_limit_short"].values.astype(np.float64)

        # --- Issuer bucket: map string names → integer indices via caps dict ---
        if "issuer_bucket" in meta_indexed.columns and issuer_dv01_caps_map:
            issuer_names = list(issuer_dv01_caps_map.keys())
            issuer_name_to_idx = {name: idx for idx, name in enumerate(issuer_names)}
            raw_issuers = meta_indexed["issuer_bucket"].values
            issuer_bucket = np.array(
                [issuer_name_to_idx.get(str(v), 0) for v in raw_issuers],
                dtype=np.int64,
            )
            _issuer_dv01_caps = np.array(
                [issuer_dv01_caps_map[name] for name in issuer_names],
                dtype=np.float64,
            )
        elif "issuer_bucket" in meta_indexed.columns:
            issuer_bucket = meta_indexed["issuer_bucket"].values.astype(np.int64)
            _issuer_dv01_caps = np.empty(0, dtype=np.float64)
        else:
            issuer_bucket = np.zeros(B, dtype=np.int64)
            _issuer_dv01_caps = np.empty(0, dtype=np.float64)

        # Maturity: prefer time-varying (T, B) from maturity_date, fall back
        # to static (B,) from maturity column.
        if "maturity_date" in meta_indexed.columns:
            maturity_dates = pd.to_datetime(
                meta_indexed["maturity_date"]
            ).values.astype("datetime64[ns]")
            # datetimes not yet available; computed after alignment (step 9).
            # Store raw dates and defer computation to after alignment.
            _maturity_dates_raw = maturity_dates
            maturity = np.zeros(B, dtype=np.float64)  # placeholder, replaced in 10b
            maturity_bucket = np.zeros(B, dtype=np.int64)  # placeholder, replaced in 10b
        else:
            _maturity_dates_raw = None
            maturity = (
                meta_indexed["maturity"].values.astype(np.float64)
                if "maturity" in meta_indexed.columns
                else np.zeros(B, dtype=np.float64)
            )
            maturity_bucket = (
                meta_indexed["maturity_bucket"].values.astype(np.int64)
                if "maturity_bucket" in meta_indexed.columns
                else np.zeros(B, dtype=np.int64)
            )

        # --- 8. Pivot hedge ratios: (T, B, H) from list-column format -----------
        hedge_ratios_pivoted = _pivot_hedge_ratios_portfolio(
            hedge_ratio_df, inst_col, dt_col, instrument_ids, hedge_ids,
        )

        # --- 9. Align all sources on common datetime index ---------------------
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
            fill_method=self.fill_method,
            essential_keys=("inst_book", ),
            datetime_source_keys=("inst_book", ),
        )

        T = len(all_dts_idx)

        # --- 10. Extract numpy arrays ------------------------------------------
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

        # Hedge ratios: (T, B*H columns) → (T, B, H)
        hedge_ratios_arr = sources["hedge_ratios"].values.astype(np.float64)
        hedge_ratios_arr = hedge_ratios_arr.reshape(T, B, H)

        datetimes = all_dts_idx.values

        # --- 10b. Compute time-varying maturity (T, B) if maturity_date available
        if _maturity_dates_raw is not None:
            dt_ns = datetimes.astype("datetime64[ns]")
            # (T, 1) - (1, B) → (T, B) timedelta
            delta = _maturity_dates_raw[np.newaxis, :] - dt_ns[:, np.newaxis]
            maturity = delta.astype("timedelta64[D]").astype(np.float64) / 365.25
            if len(maturity_bucket_bins) > 0:
                bins = np.asarray(maturity_bucket_bins, dtype=np.float64)
                maturity_bucket = np.digitize(maturity, bins).astype(np.int64)
            else:
                # No bins provided; fall back to static maturity_bucket
                maturity_bucket = (
                    meta_indexed["maturity_bucket"].values.astype(np.int64)
                    if "maturity_bucket" in meta_indexed.columns
                    else np.zeros(B, dtype=np.int64)
                )

        # --- 11. Validate shapes, warn NaNs ------------------------------------
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

        # _issuer_dv01_caps already computed in step 7 (issuer bucket section)
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


# ---------------------------------------------------------------------------
# Internal helpers
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
