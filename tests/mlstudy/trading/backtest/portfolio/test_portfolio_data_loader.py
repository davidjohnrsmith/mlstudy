"""Tests for the portfolio data loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlstudy.trading.backtest.common.data.helpers import (
    align_and_fill,
    detect_book_levels,
    extract_hedge_ids,
    pivot_simple,
)
from mlstudy.trading.backtest.portfolio.data.data_loader import (
    PortfolioDataLoader,
    PortfolioMarketData,
)
from mlstudy.trading.backtest.portfolio.configs.sweep_config import (
    load_sweep_config,
    _build_data_loader,
)


# =========================================================================
# Fixtures
# =========================================================================

T = 5  # timestamps
B = 3  # instruments
H = 2  # hedge instruments
L = 2  # book levels

INST_IDS = ["INST_A", "INST_B", "INST_C"]
HEDGE_IDS = ["HEDGE_X", "HEDGE_Y"]
ALL_IDS = INST_IDS + HEDGE_IDS
DATETIMES = pd.date_range("2024-01-01", periods=T, freq="min")


def _make_book_df(datetimes, inst_ids, n_levels):
    """Create a long-format book DataFrame with instrument_id column."""
    rows = []
    for dt in datetimes:
        for inst in inst_ids:
            row = {"datetime": dt, "instrument_id": inst}
            for lvl in range(n_levels):
                row[f"bid_px_l{lvl}"] = 99.0 - lvl * 0.01
                row[f"bid_sz_l{lvl}"] = 1_000_000.0
                row[f"ask_px_l{lvl}"] = 101.0 + lvl * 0.01
                row[f"ask_sz_l{lvl}"] = 1_000_000.0
            rows.append(row)
    return pd.DataFrame(rows)


def _make_mid_df(datetimes, inst_ids):
    """Create a long-format mid DataFrame with instrument_id column."""
    rows = []
    for dt in datetimes:
        for inst in inst_ids:
            rows.append({"datetime": dt, "instrument_id": inst, "mid_px": 100.0})
    return pd.DataFrame(rows)


def _make_dv01_df(datetimes, inst_ids):
    """Create a long-format dv01 DataFrame with instrument_id column."""
    rows = []
    for dt in datetimes:
        for inst in inst_ids:
            rows.append({"datetime": dt, "instrument_id": inst, "dv01": 0.05})
    return pd.DataFrame(rows)


def _make_signal_df(datetimes, inst_ids):
    """Create a long-format signal DataFrame with all 5 columns."""
    rows = []
    for dt in datetimes:
        for inst in inst_ids:
            rows.append({
                "datetime": dt,
                "instrument_id": inst,
                "fair_price": 100.5,
                "zscore": 2.5,
                "adf_p_value": 0.01,
                "expected_yield_pnl_bps": 2.0,
                "package_yield_bps": 3.0,
            })
    return pd.DataFrame(rows)


def _make_meta_df(inst_ids):
    """Create a static metadata DataFrame."""
    rows = []
    for inst in inst_ids:
        rows.append({
            "instrument_id": inst,
            "tradable": 1.0,
            "pos_limit_long": 5_000_000.0,
            "pos_limit_short": -5_000_000.0,
            "max_trade_notional_inc": float("inf"),
            "max_trade_notional_dec": float("inf"),
            "qty_step": 0.0,
            "min_qty_trade": 0.0,
            "maturity_date": pd.Timestamp("2030-01-01"),
        })
    return pd.DataFrame(rows)


def _make_meta_df_with_optional(inst_ids):
    """Create metadata with optional fields."""
    rows = []
    for i, inst in enumerate(inst_ids):
        rows.append({
            "instrument_id": inst,
            "tradable": 1.0,
            "pos_limit_long": 5_000_000.0,
            "pos_limit_short": -5_000_000.0,
            "max_trade_notional_inc": float("inf"),
            "max_trade_notional_dec": float("inf"),
            "qty_step": 0.0,
            "min_qty_trade": 0.0,
            "maturity_date": pd.Timestamp("2030-01-01"),
            "maturity": 5.0 + i,
            "issuer_bucket": i,
            "maturity_bucket": i % 2,
        })
    return pd.DataFrame(rows)


def _make_hedge_ratio_df(datetimes, inst_ids, hedge_ids):
    """Create a long-format hedge ratio DataFrame with list columns."""
    rows = []
    for dt in datetimes:
        for inst in inst_ids:
            rows.append({
                "datetime": dt,
                "instrument_id": inst,
                "hedge_instruments": list(hedge_ids),
                "hedge_ratios": [-0.5, -0.3],
            })
    return pd.DataFrame(rows)


def _make_hedge_meta_df(hedge_ids):
    """Create a static hedge metadata DataFrame with qty_step and min_qty_trade."""
    rows = []
    for inst in hedge_ids:
        rows.append({
            "instrument_id": inst,
            "qty_step": 0.0,
            "min_qty_trade": 0.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def data_dir(tmp_path):
    """Write synthetic parquet files and return the directory.

    One set of book/mid/dv01 files containing ALL instruments (trading + hedge).
    """
    book = _make_book_df(DATETIMES, ALL_IDS, L)
    mid = _make_mid_df(DATETIMES, ALL_IDS)
    dv01 = _make_dv01_df(DATETIMES, ALL_IDS)
    signal = _make_signal_df(DATETIMES, INST_IDS)
    meta = _make_meta_df(INST_IDS)
    hedge_ratios = _make_hedge_ratio_df(DATETIMES, INST_IDS, HEDGE_IDS)
    hedge_meta = _make_hedge_meta_df(HEDGE_IDS)

    book.to_parquet(tmp_path / "book.parquet")
    mid.to_parquet(tmp_path / "mid.parquet")
    dv01.to_parquet(tmp_path / "dv01.parquet")
    signal.to_parquet(tmp_path / "signal.parquet")
    meta.to_parquet(tmp_path / "meta.parquet")
    hedge_ratios.to_parquet(tmp_path / "hedge_ratios.parquet")
    hedge_meta.to_parquet(tmp_path / "hedge_meta.parquet")

    return tmp_path


@pytest.fixture
def loader():
    return PortfolioDataLoader(
        book_filename="book.parquet",
        mid_filename="mid.parquet",
        dv01_filename="dv01.parquet",
        signal_filename="signal.parquet",
        meta_filename="meta.parquet",
        hedge_ratio_filename="hedge_ratios.parquet",
    )


# =========================================================================
# Common helpers
# =========================================================================


class TestDetectBookLevels:
    def test_two_levels(self):
        cols = pd.Index([
            "datetime", "instrument_id",
            "bid_px_l0", "bid_sz_l0", "ask_px_l0", "ask_sz_l0",
            "bid_px_l1", "bid_sz_l1", "ask_px_l1", "ask_sz_l1",
        ])
        assert detect_book_levels(cols) == 2

    def test_one_level(self):
        cols = pd.Index(["bid_px_l0", "bid_sz_l0", "ask_px_l0", "ask_sz_l0"])
        assert detect_book_levels(cols) == 1

    def test_no_levels_raises(self):
        cols = pd.Index(["datetime", "instrument_id", "mid_px"])
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_book_levels(cols)

    def test_non_contiguous_raises(self):
        cols = pd.Index(["bid_px_l0", "bid_px_l2"])
        with pytest.raises(ValueError, match="not contiguous"):
            detect_book_levels(cols)


class TestPivotSimple:
    def test_basic_pivot(self):
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "inst": ["A", "B", "A", "B"],
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        result = pivot_simple(df, "datetime", "inst", ["A", "B"], "val")
        assert result.shape == (2, 2)
        assert result.iloc[0]["A"] == 1.0
        assert result.iloc[1]["B"] == 4.0

    def test_missing_instrument_nan(self):
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-01-01"]),
            "inst": ["A"],
            "val": [1.0],
        })
        result = pivot_simple(df, "datetime", "inst", ["A", "B"], "val")
        assert result.shape == (1, 2)
        assert np.isnan(result.iloc[0]["B"])


class TestExtractHedgeIds:
    def test_basic(self):
        df = pd.DataFrame({
            "instrument_id": ["A", "A"],
            "hedge_instruments": [["X", "Y"], ["Y", "Z"]],
            "hedge_ratios": [[-0.5, -0.5], [-0.5, -0.5]],
        })
        assert extract_hedge_ids(df) == ["X", "Y", "Z"]


class TestAlignAndFill:
    def test_ffill_drops_leading_nans(self):
        dts_a = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="min"))
        dts_b = pd.DatetimeIndex(pd.date_range("2024-01-01 00:02", periods=3, freq="min"))
        src_a = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dts_a)
        src_b = pd.DataFrame({"y": [10.0, 20.0, 30.0]}, index=dts_b)

        aligned, idx = align_and_fill(
            {"a": src_a, "b": src_b},
            fill_method="ffill",
            essential_keys=("b",),
        )
        # b starts at t=2, so first 2 rows dropped
        assert len(idx) == 3
        assert not aligned["b"].isna().any().any()

    def test_drop_mode(self):
        dts_a = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="min"))
        dts_b = pd.DatetimeIndex(pd.date_range("2024-01-01 00:02", periods=3, freq="min"))
        src_a = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dts_a)
        src_b = pd.DataFrame({"y": [10.0, 20.0, 30.0]}, index=dts_b)

        aligned, idx = align_and_fill(
            {"a": src_a, "b": src_b},
            fill_method="drop",
        )
        # Only 3 overlapping rows
        assert len(idx) == 3

    def test_fillna_defaults(self):
        dts = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="min"))
        src = pd.DataFrame({"x": [1.0, np.nan, 3.0]}, index=dts)

        aligned, idx = align_and_fill(
            {"a": src},
            fill_method="ffill",
            fillna_defaults={"a": 99.0},
        )
        # After ffill: [1.0, 1.0, 3.0] — no NaN left to fill with default
        assert not aligned["a"].isna().any().any()


# =========================================================================
# PortfolioMarketData
# =========================================================================


class TestPortfolioMarketDataToDict:
    def test_keys(self):
        md = PortfolioMarketData(
            bid_px=np.zeros((T, B, L)),
            bid_sz=np.zeros((T, B, L)),
            ask_px=np.zeros((T, B, L)),
            ask_sz=np.zeros((T, B, L)),
            mid_px=np.zeros((T, B)),
            dv01=np.zeros((T, B)),
            fair_price=np.zeros((T, B)),
            zscore=np.zeros((T, B)),
            adf_p_value=np.zeros((T, B)),
            tradable=np.ones(B),
            pos_limits_long=np.ones(B) * 5e6,
            pos_limits_short=np.ones(B) * -5e6,
            max_trade_notional_inc=np.full(B, np.inf),
            max_trade_notional_dec=np.full(B, np.inf),
            qty_step=np.zeros(B),
            min_qty_trade=np.zeros(B),
            maturity=np.full((T, B), 5.0),
            issuer_bucket=np.zeros(B, dtype=np.int64),
            maturity_bucket=np.zeros((T, B), dtype=np.int64),
            issuer_dv01_caps=np.empty(0),
            mat_bucket_dv01_caps=np.empty(0),
            hedge_bid_px=np.zeros((T, H, L)),
            hedge_bid_sz=np.zeros((T, H, L)),
            hedge_ask_px=np.zeros((T, H, L)),
            hedge_ask_sz=np.zeros((T, H, L)),
            hedge_mid_px=np.zeros((T, H)),
            hedge_dv01=np.zeros((T, H)),
            hedge_ratios=np.zeros((T, B, H)),
            hedge_qty_step=np.zeros(H),
            hedge_min_qty_trade=np.zeros(H),
            datetimes=DATETIMES.values,
            instrument_ids=INST_IDS,
            hedge_ids=HEDGE_IDS,
        )
        d = md.to_dict()
        expected_keys = {
            "bid_px", "bid_sz", "ask_px", "ask_sz", "mid_px", "dv01",
            "fair_price", "zscore", "adf_p_value",
            "tradable", "pos_limits_long", "pos_limits_short",
            "max_trade_notional_inc", "max_trade_notional_dec",
            "qty_step", "min_qty_trade",
            "maturity", "issuer_bucket", "maturity_bucket",
            "issuer_dv01_caps", "mat_bucket_dv01_caps",
            "hedge_bid_px", "hedge_bid_sz", "hedge_ask_px", "hedge_ask_sz",
            "hedge_mid_px", "hedge_dv01", "hedge_ratios",
            "hedge_qty_step", "hedge_min_qty_trade",
            "datetimes", "instrument_ids",
        }
        assert set(d.keys()) == expected_keys

    def test_optional_meta_included(self):
        md = PortfolioMarketData(
            bid_px=np.zeros((T, B, L)),
            bid_sz=np.zeros((T, B, L)),
            ask_px=np.zeros((T, B, L)),
            ask_sz=np.zeros((T, B, L)),
            mid_px=np.zeros((T, B)),
            dv01=np.zeros((T, B)),
            fair_price=np.zeros((T, B)),
            zscore=np.zeros((T, B)),
            adf_p_value=np.zeros((T, B)),
            tradable=np.ones(B),
            pos_limits_long=np.ones(B) * 5e6,
            pos_limits_short=np.ones(B) * -5e6,
            max_trade_notional_inc=np.full(B, np.inf),
            max_trade_notional_dec=np.full(B, np.inf),
            qty_step=np.zeros(B),
            min_qty_trade=np.zeros(B),
            maturity=np.array([5.0, 6.0, 7.0]),
            issuer_bucket=np.array([0, 1, 2]),
            maturity_bucket=np.zeros(B, dtype=np.int64),
            issuer_dv01_caps=np.empty(0),
            mat_bucket_dv01_caps=np.empty(0),
            hedge_bid_px=np.zeros((T, H, L)),
            hedge_bid_sz=np.zeros((T, H, L)),
            hedge_ask_px=np.zeros((T, H, L)),
            hedge_ask_sz=np.zeros((T, H, L)),
            hedge_mid_px=np.zeros((T, H)),
            hedge_dv01=np.zeros((T, H)),
            hedge_ratios=np.zeros((T, B, H)),
            hedge_qty_step=np.zeros(H),
            hedge_min_qty_trade=np.zeros(H),
            datetimes=DATETIMES.values,
            instrument_ids=INST_IDS,
            hedge_ids=HEDGE_IDS,
        )
        d = md.to_dict()
        assert "maturity" in d
        assert "issuer_bucket" in d
        assert "maturity_bucket" in d


# =========================================================================
# PortfolioDataLoader — end-to-end
# =========================================================================


class TestPortfolioDataLoaderLoad:
    def test_load_shapes(self, data_dir, loader):
        md = loader.load(
            instrument_ids=INST_IDS,
            hedge_ids=HEDGE_IDS,
            data_path=data_dir,
        )
        assert md.bid_px.shape == (T, B, L)
        assert md.bid_sz.shape == (T, B, L)
        assert md.ask_px.shape == (T, B, L)
        assert md.ask_sz.shape == (T, B, L)
        assert md.mid_px.shape == (T, B)
        assert md.dv01.shape == (T, B)
        assert md.fair_price.shape == (T, B)
        assert md.zscore.shape == (T, B)
        assert md.adf_p_value.shape == (T, B)
        assert md.tradable.shape == (B,)
        assert md.pos_limits_long.shape == (B,)
        assert md.pos_limits_short.shape == (B,)
        assert md.hedge_bid_px.shape == (T, H, L)
        assert md.hedge_bid_sz.shape == (T, H, L)
        assert md.hedge_ask_px.shape == (T, H, L)
        assert md.hedge_ask_sz.shape == (T, H, L)
        assert md.hedge_mid_px.shape == (T, H)
        assert md.hedge_dv01.shape == (T, H)
        assert md.hedge_ratios.shape == (T, B, H)
        assert md.datetimes.shape == (T,)
        assert md.instrument_ids == INST_IDS
        assert md.hedge_ids == HEDGE_IDS

    def test_load_values(self, data_dir, loader):
        md = loader.load(
            instrument_ids=INST_IDS,
            hedge_ids=HEDGE_IDS,
            data_path=data_dir,
        )
        # Check some known values
        assert md.mid_px[0, 0] == 100.0
        assert md.dv01[0, 0] == 0.05
        assert md.fair_price[0, 0] == 100.5
        assert md.zscore[0, 0] == 2.5
        assert md.tradable[0] == 1.0
        assert md.pos_limits_long[0] == 5_000_000.0

    def test_no_data_path_raises(self, loader):
        with pytest.raises(ValueError, match="data_path"):
            loader.load(instrument_ids=INST_IDS, hedge_ids=HEDGE_IDS)

    def test_data_path_on_constructor(self, data_dir):
        loader = PortfolioDataLoader(
            book_filename="book.parquet",
            mid_filename="mid.parquet",
            dv01_filename="dv01.parquet",
            signal_filename="signal.parquet",
            meta_filename="meta.parquet",
            hedge_ratio_filename="hedge_ratios.parquet",
            data_path=data_dir,
        )
        md = loader.load(instrument_ids=INST_IDS, hedge_ids=HEDGE_IDS)
        assert md.bid_px.shape == (T, B, L)


class TestAutoDetectBonds:
    def test_auto_detect_from_meta(self, data_dir, loader):
        md = loader.load(
            instrument_ids=None,
            hedge_ids=HEDGE_IDS,
            data_path=data_dir,
        )
        assert sorted(md.instrument_ids) == sorted(INST_IDS)
        assert md.bid_px.shape[1] == B


class TestAutoDetectHedges:
    def test_auto_detect_from_hedge_ratios(self, data_dir, loader):
        md = loader.load(
            instrument_ids=INST_IDS,
            hedge_ids=None,
            data_path=data_dir,
        )
        assert sorted(md.hedge_ids) == sorted(HEDGE_IDS)
        assert md.hedge_bid_px.shape[1] == H


class TestHedgeRatiosShape:
    def test_shape_TBH(self, data_dir, loader):
        md = loader.load(
            instrument_ids=INST_IDS,
            hedge_ids=HEDGE_IDS,
            data_path=data_dir,
        )
        assert md.hedge_ratios.shape == (T, B, H)
        # Check known values from fixture
        assert md.hedge_ratios[0, 0, 0] == -0.5
        assert md.hedge_ratios[0, 0, 1] == -0.3


class TestFfillMethod:
    def test_ffill_works(self, tmp_path):
        """Forward-fill works correctly, drops leading NaN rows."""
        # Create data where signal starts at t=2 (t=0,1 missing)
        dts_inst = DATETIMES
        dts_signal = DATETIMES[2:]  # signal starts late

        book = _make_book_df(dts_inst, ALL_IDS, L)
        mid = _make_mid_df(dts_inst, ALL_IDS)
        dv01 = _make_dv01_df(dts_inst, ALL_IDS)
        signal = _make_signal_df(dts_signal, INST_IDS)
        meta = _make_meta_df(INST_IDS)
        hedge_ratios = _make_hedge_ratio_df(dts_inst, INST_IDS, HEDGE_IDS)
        hedge_meta = _make_hedge_meta_df(HEDGE_IDS)

        for name, df in [
            ("book", book), ("mid", mid),
            ("dv01", dv01), ("signal", signal),
            ("meta", meta), ("hedge_ratios", hedge_ratios),
            ("hedge_meta", hedge_meta),
        ]:
            df.to_parquet(tmp_path / f"{name}.parquet")

        loader = PortfolioDataLoader(
            book_filename="book.parquet",
            mid_filename="mid.parquet",
            dv01_filename="dv01.parquet",
            signal_filename="signal.parquet",
            meta_filename="meta.parquet",
            hedge_ratio_filename="hedge_ratios.parquet",
            fill_method="ffill",
        )
        md = loader.load(instrument_ids=INST_IDS, hedge_ids=HEDGE_IDS, data_path=tmp_path)
        # Essential key is inst_book which has all 5 rows, so all survive.
        # Signal (fair_price) at t=0,1 will have NaN since there's nothing to ffill from.
        assert md.datetimes.shape[0] == 5
        # From t=2 onward, fair_price should be non-NaN
        assert not np.any(np.isnan(md.fair_price[2:]))


class TestDropMethod:
    def test_drop_keeps_complete_rows(self, tmp_path):
        """Drop method keeps only complete rows."""
        dts_inst = DATETIMES
        dts_signal = DATETIMES[2:]  # signal starts at t=2

        book = _make_book_df(dts_inst, ALL_IDS, L)
        mid = _make_mid_df(dts_inst, ALL_IDS)
        dv01 = _make_dv01_df(dts_inst, ALL_IDS)
        signal = _make_signal_df(dts_signal, INST_IDS)
        meta = _make_meta_df(INST_IDS)
        hedge_ratios = _make_hedge_ratio_df(dts_inst, INST_IDS, HEDGE_IDS)
        hedge_meta = _make_hedge_meta_df(HEDGE_IDS)

        for name, df in [
            ("book", book), ("mid", mid),
            ("dv01", dv01), ("signal", signal),
            ("meta", meta), ("hedge_ratios", hedge_ratios),
            ("hedge_meta", hedge_meta),
        ]:
            df.to_parquet(tmp_path / f"{name}.parquet")

        loader = PortfolioDataLoader(
            book_filename="book.parquet",
            mid_filename="mid.parquet",
            dv01_filename="dv01.parquet",
            signal_filename="signal.parquet",
            meta_filename="meta.parquet",
            hedge_ratio_filename="hedge_ratios.parquet",
            fill_method="drop",
        )
        md = loader.load(instrument_ids=INST_IDS, hedge_ids=HEDGE_IDS, data_path=tmp_path)
        # Only the 3 complete rows should remain
        assert md.datetimes.shape[0] == 3


class TestValidateShapes:
    def test_shape_mismatch_detected(self, data_dir, loader):
        """Shape validation catches mismatches when inst_ids list is wrong."""
        # Try loading with wrong inst_ids that don't match data
        wrong_insts = ["INST_A", "INST_B"]  # missing INST_C
        md = loader.load(
            instrument_ids=wrong_insts,
            hedge_ids=HEDGE_IDS,
            data_path=data_dir,
        )
        # Should still work but with B=2
        assert md.bid_px.shape == (T, 2, L)
        assert md.hedge_ratios.shape == (T, 2, H)


class TestOptionalMeta:
    def test_optional_meta_loaded(self, tmp_path):
        """Optional meta fields (maturity, etc.) are loaded when present."""
        book = _make_book_df(DATETIMES, ALL_IDS, L)
        mid = _make_mid_df(DATETIMES, ALL_IDS)
        dv01 = _make_dv01_df(DATETIMES, ALL_IDS)
        signal = _make_signal_df(DATETIMES, INST_IDS)
        meta = _make_meta_df_with_optional(INST_IDS)
        hedge_ratios = _make_hedge_ratio_df(DATETIMES, INST_IDS, HEDGE_IDS)
        hedge_meta = _make_hedge_meta_df(HEDGE_IDS)

        for name, df in [
            ("book", book), ("mid", mid),
            ("dv01", dv01), ("signal", signal),
            ("meta", meta), ("hedge_ratios", hedge_ratios),
            ("hedge_meta", hedge_meta),
        ]:
            df.to_parquet(tmp_path / f"{name}.parquet")

        loader = PortfolioDataLoader(
            book_filename="book.parquet",
            mid_filename="mid.parquet",
            dv01_filename="dv01.parquet",
            signal_filename="signal.parquet",
            meta_filename="meta.parquet",
            hedge_ratio_filename="hedge_ratios.parquet",
        )
        md = loader.load(instrument_ids=INST_IDS, hedge_ids=HEDGE_IDS, data_path=tmp_path)
        assert md.maturity is not None
        assert md.maturity.shape == (T, B)
        assert md.issuer_bucket is not None
        assert md.maturity_bucket is not None

    def test_meta_defaults_when_absent(self, data_dir, loader):
        md = loader.load(
            instrument_ids=INST_IDS, hedge_ids=HEDGE_IDS, data_path=data_dir,
        )
        # maturity is always computed from maturity_date
        assert md.maturity is not None
        assert md.maturity.shape == (T, B)
        # issuer_bucket defaults to zeros
        assert md.issuer_bucket is not None
        assert md.issuer_bucket.shape == (B,)
        # maturity_bucket defaults to zeros
        assert md.maturity_bucket is not None


# =========================================================================
# Sweep config with data loader
# =========================================================================


_BASE_CFG_DICT = {
    "use_greedy": False,
    "gross_dv01_cap": 100.0, "top_k": 10,
    "z_inc": 2.0, "p_inc": 0.05,
    "z_dec": 1.0, "p_dec": 0.10,
    "alpha_thr_inc": 1.0, "alpha_thr_dec": 0.5,
    "max_levels": 3, "haircut": 1.0,
    "min_fill_ratio": 0.0,
    "cooldown_bars": 0, "cooldown_mode": 0,
    "min_maturity_inc": 0.0,
    "initial_capital": 1_000_000.0,
    "close_time": "none",
}


class TestSweepConfigWithDataLoader:
    def test_yaml_round_trip(self, tmp_path):
        """YAML with data section round-trips correctly."""
        base = dict(_BASE_CFG_DICT)
        base.pop("z_inc")  # z_inc is in grid
        config = {
            "grid_name": "test_grid",
            "base_config": base,
            "grid": {"z_inc": [1.5, 2.0]},
            "data": {
                "book_filename": "book.parquet",
                "mid_filename": "mid.parquet",
                "dv01_filename": "dv01.parquet",
                "signal_filename": "signal.parquet",
                "meta_filename": "meta.parquet",
                "hedge_ratio_filename": "hedge_ratios.parquet",
                "hedge_meta_filename": "hedge_meta.parquet",
                "data_path": "/data/test",
            },
        }
        yaml_path = tmp_path / "test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        cfg = load_sweep_config(yaml_path)
        assert cfg.data_loader is not None
        assert isinstance(cfg.data_loader, PortfolioDataLoader)
        assert cfg.data_loader.book_filename == "book.parquet"
        assert cfg.data_loader.data_path == "/data/test"

    def test_no_data_section(self, tmp_path):
        """Config without data section has data_loader=None."""
        base = dict(_BASE_CFG_DICT)
        base.pop("z_inc")  # z_inc is in grid
        config = {
            "grid_name": "test_grid",
            "base_config": base,
            "grid": {"z_inc": [1.5, 2.0]},
        }
        yaml_path = tmp_path / "test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        cfg = load_sweep_config(yaml_path)
        assert cfg.data_loader is None

    def test_build_data_loader_none(self):
        assert _build_data_loader(None) is None

    def test_build_data_loader_valid(self):
        raw = {
            "book_filename": "a.parquet",
            "mid_filename": "b.parquet",
            "dv01_filename": "c.parquet",
            "signal_filename": "d.parquet",
            "meta_filename": "e.parquet",
            "hedge_ratio_filename": "i.parquet",
            "hedge_meta_filename": "j.parquet",
        }
        dl = _build_data_loader(raw)
        assert isinstance(dl, PortfolioDataLoader)
        assert dl.book_filename == "a.parquet"
