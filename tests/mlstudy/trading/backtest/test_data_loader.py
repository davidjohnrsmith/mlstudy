"""Tests for BacktestDataLoader and MarketData."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest.mean_reversion.data_loader import (
    BacktestDataLoader,
    MarketData,
    _detect_book_levels,
)


# ---------------------------------------------------------------------------
# Helpers — write synthetic parquet files
# ---------------------------------------------------------------------------

INSTRUMENTS = ["A", "B", "C"]
N_LEVELS = 2
DT_COL = "datetime"
INST_COL = "instrument_id"


def _make_datetimes(n: int, start: str = "2024-01-01 09:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="min")


def _make_book_df(
    datetimes: pd.DatetimeIndex,
    instruments: list[str],
    n_levels: int = N_LEVELS,
) -> pd.DataFrame:
    rows = []
    for dt in datetimes:
        for inst in instruments:
            row = {DT_COL: dt, INST_COL: inst}
            for l in range(n_levels):
                row[f"bid_price_l{l}"] = 100.0 - l * 0.01
                row[f"bid_size_l{l}"] = 10.0 + l
                row[f"ask_price_l{l}"] = 100.0 + (l + 1) * 0.01
                row[f"ask_size_l{l}"] = 10.0 + l
            rows.append(row)
    return pd.DataFrame(rows)


def _make_mid_df(
    datetimes: pd.DatetimeIndex,
    instruments: list[str],
) -> pd.DataFrame:
    rows = []
    for dt in datetimes:
        for inst in instruments:
            rows.append({DT_COL: dt, INST_COL: inst, "mid_px": 100.0})
    return pd.DataFrame(rows)


def _make_dv01_df(
    datetimes: pd.DatetimeIndex,
    instruments: list[str],
) -> pd.DataFrame:
    rows = []
    for dt in datetimes:
        for inst in instruments:
            rows.append({DT_COL: dt, INST_COL: inst, "dv01": 0.05})
    return pd.DataFrame(rows)


def _make_signal_df(
    datetimes: pd.DatetimeIndex,
    instruments: list[str],
    ref_instrument: str,
) -> pd.DataFrame:
    rows = []
    for dt in datetimes:
        for inst in instruments:
            rows.append({
                DT_COL: dt,
                INST_COL: inst,
                "zscore": 1.5 if inst == ref_instrument else 0.5,
                "expected_yield_pnl_bps": 2.0,
                "package_yield_bps": 3.0,
            })
    return pd.DataFrame(rows)


def _make_hedge_df(
    datetimes: pd.DatetimeIndex,
    instruments: list[str],
    ratios: list[float] | None = None,
) -> pd.DataFrame:
    if ratios is None:
        # Default: sum to 0
        ratios = [1.0, -0.5, -0.5]
    rows = []
    for dt in datetimes:
        for inst, r in zip(instruments, ratios):
            rows.append({DT_COL: dt, INST_COL: inst, "hedge_ratio": r})
    return pd.DataFrame(rows)


def _write_all_parquets(
    data_dir,
    datetimes: pd.DatetimeIndex | None = None,
    instruments: list[str] | None = None,
    n_levels: int = N_LEVELS,
    ref_instrument: str = "B",
    hedge_ratios: list[float] | None = None,
):
    """Write a complete set of parquet files to *data_dir*."""
    if datetimes is None:
        datetimes = _make_datetimes(10)
    if instruments is None:
        instruments = INSTRUMENTS

    book_df = _make_book_df(datetimes, instruments, n_levels)
    mid_df = _make_mid_df(datetimes, instruments)
    dv01_df = _make_dv01_df(datetimes, instruments)
    signal_df = _make_signal_df(datetimes, instruments, ref_instrument)
    hedge_df = _make_hedge_df(datetimes, instruments, hedge_ratios)

    book_df.to_parquet(data_dir / "book.parquet", index=False)
    mid_df.to_parquet(data_dir / "mid.parquet", index=False)
    dv01_df.to_parquet(data_dir / "dv01.parquet", index=False)
    signal_df.to_parquet(data_dir / "signal.parquet", index=False)
    hedge_df.to_parquet(data_dir / "hedge_ratios.parquet", index=False)


def _make_loader(data_dir, **overrides) -> BacktestDataLoader:
    defaults = dict(
        data_path=str(data_dir),
        book_filename="book.parquet",
        mid_filename="mid.parquet",
        dv01_filename="dv01.parquet",
        signal_filename="signal.parquet",
        hedge_ratio_filename="hedge_ratios.parquet",
        instrument_ids=INSTRUMENTS,
        ref_instrument_id="B",
    )
    defaults.update(overrides)
    return BacktestDataLoader(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectBookLevels:
    def test_two_levels(self):
        cols = pd.Index([
            "datetime", "instrument_id",
            "bid_price_l0", "bid_size_l0", "ask_price_l0", "ask_size_l0",
            "bid_price_l1", "bid_size_l1", "ask_price_l1", "ask_size_l1",
        ])
        assert _detect_book_levels(cols) == 2

    def test_one_level(self):
        cols = pd.Index([
            "bid_price_l0", "bid_size_l0", "ask_price_l0", "ask_size_l0",
        ])
        assert _detect_book_levels(cols) == 1

    def test_no_levels_raises(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            _detect_book_levels(pd.Index(["datetime", "price"]))

    def test_non_contiguous_raises(self):
        cols = pd.Index(["bid_price_l0", "bid_price_l2"])
        with pytest.raises(ValueError, match="not contiguous"):
            _detect_book_levels(cols)


class TestBasicLoad:
    """Basic end-to-end load with aligned data."""

    def test_shapes(self, tmp_path):
        T = 10
        dts = _make_datetimes(T)
        _write_all_parquets(tmp_path, datetimes=dts)
        loader = _make_loader(tmp_path)
        md = loader.load()

        N = len(INSTRUMENTS)
        L = N_LEVELS
        assert md.bid_px.shape == (T, N, L)
        assert md.bid_sz.shape == (T, N, L)
        assert md.ask_px.shape == (T, N, L)
        assert md.ask_sz.shape == (T, N, L)
        assert md.mid_px.shape == (T, N)
        assert md.dv01.shape == (T, N)
        assert md.zscore.shape == (T,)
        assert md.expected_yield_pnl_bps.shape == (T,)
        assert md.package_yield_bps.shape == (T,)
        assert md.hedge_ratios.shape == (N,)
        assert md.datetimes.shape == (T,)
        assert md.instrument_ids == INSTRUMENTS

    def test_values(self, tmp_path):
        _write_all_parquets(tmp_path, datetimes=_make_datetimes(5))
        md = _make_loader(tmp_path).load()

        # bid_px level 0 should be 100.0 for all instruments
        np.testing.assert_allclose(md.bid_px[:, :, 0], 100.0)
        # bid_px level 1 should be 99.99
        np.testing.assert_allclose(md.bid_px[:, :, 1], 99.99)
        # mid_px
        np.testing.assert_allclose(md.mid_px, 100.0)
        # zscore filtered to ref instrument "B" → 1.5
        np.testing.assert_allclose(md.zscore, 1.5)
        # hedge_ratios
        np.testing.assert_allclose(md.hedge_ratios, [1.0, -0.5, -0.5])

    def test_to_dict_keys(self, tmp_path):
        _write_all_parquets(tmp_path, datetimes=_make_datetimes(5))
        md = _make_loader(tmp_path).load()
        d = md.to_dict()
        expected_keys = {
            "bid_px", "bid_sz", "ask_px", "ask_sz", "mid_px",
            "dv01", "zscore", "expected_yield_pnl_bps",
            "package_yield_bps", "hedge_ratios",
        }
        assert set(d.keys()) == expected_keys


class TestSignalFiltering:
    def test_ref_instrument_filter(self, tmp_path):
        dts = _make_datetimes(5)
        _write_all_parquets(tmp_path, datetimes=dts, ref_instrument="A")
        # Load with ref_instrument_id="A" — zscore for A is 1.5
        md = _make_loader(tmp_path, ref_instrument_id="A").load()
        np.testing.assert_allclose(md.zscore, 1.5)

    def test_invalid_ref_instrument_raises(self, tmp_path):
        _write_all_parquets(tmp_path, datetimes=_make_datetimes(5))
        loader = _make_loader(tmp_path, ref_instrument_id="NONEXISTENT")
        with pytest.raises(ValueError, match="No signal data"):
            loader.load()


class TestDatetimeAlignment:
    """Test misaligned datetimes between sources."""

    def test_ffill_mode(self, tmp_path):
        """Different sources have different timestamps → ffill aligns them."""
        instruments = INSTRUMENTS
        dts_book = _make_datetimes(5, "2024-01-01 09:00")
        dts_mid = _make_datetimes(5, "2024-01-01 09:01")  # offset by 1 min

        # Write book with dts_book
        _make_book_df(dts_book, instruments).to_parquet(
            tmp_path / "book.parquet", index=False
        )
        # Write mid with dts_mid
        _make_mid_df(dts_mid, instruments).to_parquet(
            tmp_path / "mid.parquet", index=False
        )
        # Other files with union of timestamps
        all_dts = dts_book.union(dts_mid).sort_values()
        _make_dv01_df(all_dts, instruments).to_parquet(
            tmp_path / "dv01.parquet", index=False
        )
        _make_signal_df(all_dts, instruments, "B").to_parquet(
            tmp_path / "signal.parquet", index=False
        )
        _make_hedge_df(all_dts, instruments).to_parquet(
            tmp_path / "hedge_ratios.parquet", index=False
        )

        loader = _make_loader(tmp_path, fill_method="ffill")
        md = loader.load()

        # First bar (09:00) has book but not mid → mid NaN → leading NaN dropped
        # First valid bar is 09:01 where both have data after ffill
        # Total bars: 09:01 through 09:04 = 9 bars (from union, after dropping leading)
        assert md.bid_px.shape[0] == md.mid_px.shape[0]
        assert md.bid_px.shape[0] > 0
        assert not np.any(np.isnan(md.mid_px))

    def test_drop_mode(self, tmp_path):
        """In drop mode, only keep rows where ALL sources have data."""
        instruments = INSTRUMENTS
        dts_common = _make_datetimes(3, "2024-01-01 09:01")
        dts_book = _make_datetimes(3, "2024-01-01 09:00")  # 09:00, 09:01, 09:02

        _make_book_df(dts_book, instruments).to_parquet(
            tmp_path / "book.parquet", index=False
        )
        _make_mid_df(dts_common, instruments).to_parquet(
            tmp_path / "mid.parquet", index=False
        )
        _make_dv01_df(dts_common, instruments).to_parquet(
            tmp_path / "dv01.parquet", index=False
        )
        _make_signal_df(dts_common, instruments, "B").to_parquet(
            tmp_path / "signal.parquet", index=False
        )
        _make_hedge_df(dts_common, instruments).to_parquet(
            tmp_path / "hedge_ratios.parquet", index=False
        )

        loader = _make_loader(tmp_path, fill_method="drop")
        md = loader.load()

        # Only 09:01, 09:02 are common to both book and mid
        assert md.bid_px.shape[0] == 2


class TestAutoDetectLevels:
    def test_three_levels(self, tmp_path):
        dts = _make_datetimes(5)
        _write_all_parquets(tmp_path, datetimes=dts, n_levels=3)
        md = _make_loader(tmp_path).load()
        assert md.bid_px.shape[2] == 3

    def test_one_level(self, tmp_path):
        dts = _make_datetimes(5)
        _write_all_parquets(tmp_path, datetimes=dts, n_levels=1)
        md = _make_loader(tmp_path).load()
        assert md.bid_px.shape[2] == 1


class TestHedgeRatios:
    def test_constant_ratios(self, tmp_path):
        _write_all_parquets(
            tmp_path,
            datetimes=_make_datetimes(5),
            hedge_ratios=[1.0, -0.6, -0.4],
        )
        md = _make_loader(tmp_path).load()
        np.testing.assert_allclose(md.hedge_ratios, [1.0, -0.6, -0.4])

    def test_varying_ratios_warns(self, tmp_path):
        """Hedge ratios that change over time should produce a warning."""
        instruments = INSTRUMENTS
        dts = _make_datetimes(5)
        rows = []
        for i, dt in enumerate(dts):
            for j, inst in enumerate(instruments):
                # Vary the first instrument's ratio over time
                r = 1.0 + i * 0.1 if j == 0 else -0.5
                rows.append({DT_COL: dt, INST_COL: inst, "hedge_ratio": r})
        pd.DataFrame(rows).to_parquet(tmp_path / "hedge_ratios.parquet", index=False)
        # Write other files normally
        _make_book_df(dts, instruments).to_parquet(tmp_path / "book.parquet", index=False)
        _make_mid_df(dts, instruments).to_parquet(tmp_path / "mid.parquet", index=False)
        _make_dv01_df(dts, instruments).to_parquet(tmp_path / "dv01.parquet", index=False)
        _make_signal_df(dts, instruments, "B").to_parquet(tmp_path / "signal.parquet", index=False)

        loader = _make_loader(tmp_path)
        with pytest.warns(UserWarning, match="vary over time"):
            md = loader.load()
        # Should use last row
        np.testing.assert_allclose(md.hedge_ratios[0], 1.0 + 4 * 0.1)

    def test_non_zero_sum_warns(self, tmp_path):
        _write_all_parquets(
            tmp_path,
            datetimes=_make_datetimes(5),
            hedge_ratios=[1.0, 0.5, 0.5],  # sum = 2.0
        )
        loader = _make_loader(tmp_path)
        with pytest.warns(UserWarning, match="do not sum to zero"):
            loader.load()


class TestMarketDataToDict:
    def test_round_trip(self, tmp_path):
        _write_all_parquets(tmp_path, datetimes=_make_datetimes(5))
        md = _make_loader(tmp_path).load()
        d = md.to_dict()
        # All values should be numpy arrays
        for k, v in d.items():
            assert isinstance(v, np.ndarray), f"{k} is not ndarray"
        # datetimes and instrument_ids should NOT be in dict (not needed by engine)
        assert "datetimes" not in d
        assert "instrument_ids" not in d


class TestSweepConfigDataSection:
    """Test that sweep_config.py parses the ``data`` section."""

    def test_parse_data_section(self, tmp_path):
        import yaml
        from mlstudy.trading.backtest.mean_reversion.sweep_config import load_sweep_config

        config = {
            "grid_name": "test",
            "base_config": {"ref_leg_idx": 0},
            "grid": {"entry_z_threshold": [1.0, 2.0]},
            "data": {
                "data_path": str(tmp_path),
                "book_filename": "book.parquet",
                "mid_filename": "mid.parquet",
                "dv01_filename": "dv01.parquet",
                "signal_filename": "signal.parquet",
                "hedge_ratio_filename": "hedge_ratios.parquet",
                "instrument_ids": ["A", "B"],
                "ref_instrument_id": "A",
            },
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        cfg = load_sweep_config(yaml_path)
        assert cfg.data_loader is not None
        assert cfg.data_loader.ref_instrument_id == "A"
        assert cfg.data_loader.instrument_ids == ["A", "B"]

    def test_no_data_section(self, tmp_path):
        import yaml
        from mlstudy.trading.backtest.mean_reversion.sweep_config import load_sweep_config

        config = {
            "grid_name": "test",
            "base_config": {"ref_leg_idx": 0},
            "grid": {"entry_z_threshold": [1.0]},
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        cfg = load_sweep_config(yaml_path)
        assert cfg.data_loader is None
