"""Tests for LP portfolio backtest engine, config, results, and common helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mlstudy.trading.backtest.common.single_backtest.engine import (
    ensure_f64,
    validate_l2_shapes,
)
from mlstudy.trading.backtest.portfolio.configs.backtest_config import PortfolioBacktestConfig
from mlstudy.trading.backtest.portfolio.single_backtest.engine import run_backtest, _validate
from mlstudy.trading.backtest.portfolio.single_backtest.results import PortfolioBacktestResults


# =========================================================================
# Helpers
# =========================================================================

def _make_market(T, B, L=3, mid_base=100.0, spread_bps=10.0):
    mid_px = np.full((T, B), mid_base, dtype=np.float64)
    spread = mid_base * spread_bps / 1e4
    bid_px = np.zeros((T, B, L), dtype=np.float64)
    bid_sz = np.zeros((T, B, L), dtype=np.float64)
    ask_px = np.zeros((T, B, L), dtype=np.float64)
    ask_sz = np.zeros((T, B, L), dtype=np.float64)
    for lev in range(L):
        bid_px[:, :, lev] = mid_base - spread * (lev + 1)
        ask_px[:, :, lev] = mid_base + spread * (lev + 1)
        bid_sz[:, :, lev] = 1000.0
        ask_sz[:, :, lev] = 1000.0
    return bid_px, bid_sz, ask_px, ask_sz, mid_px


def _make_hedge_market(T, H, L=3, mid_base=100.0, spread_bps=10.0):
    mid_px = np.full((T, H), mid_base, dtype=np.float64)
    spread = mid_base * spread_bps / 1e4
    bid_px = np.zeros((T, H, L), dtype=np.float64)
    bid_sz = np.zeros((T, H, L), dtype=np.float64)
    ask_px = np.zeros((T, H, L), dtype=np.float64)
    ask_sz = np.zeros((T, H, L), dtype=np.float64)
    for lev in range(L):
        bid_px[:, :, lev] = mid_base - spread * (lev + 1)
        ask_px[:, :, lev] = mid_base + spread * (lev + 1)
        bid_sz[:, :, lev] = 10000.0
        ask_sz[:, :, lev] = 10000.0
    return bid_px, bid_sz, ask_px, ask_sz, mid_px


def _base_inputs(T=3, B=2):
    L = 3
    bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
    return dict(
        bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz, mid_px=mid_px,
        dv01=np.full((T, B), 0.01, dtype=np.float64),
        fair_price=mid_px + 1.0,
        zscore=np.full((T, B), 3.0),
        adf_p_value=np.full((T, B), 0.01),
        tradable=np.ones(B, dtype=np.int32),
        pos_limits_long=np.full(B, 1e6),
        pos_limits_short=np.full(B, -1e6),
    )


# =========================================================================
# Common helpers
# =========================================================================


class TestEnsureF64:
    def test_passthrough_f64(self):
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = ensure_f64(a)
        assert b is a  # no copy

    def test_converts_f32(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = ensure_f64(a)
        assert b.dtype == np.float64
        assert b is not a

    def test_converts_int(self):
        a = np.array([1, 2], dtype=np.int32)
        b = ensure_f64(a)
        assert b.dtype == np.float64


class TestValidateL2Shapes:
    def test_valid(self):
        T, N, L = 5, 3, 2
        bp = np.zeros((T, N, L))
        bs = np.zeros((T, N, L))
        ap = np.zeros((T, N, L))
        az = np.zeros((T, N, L))
        mp = np.zeros((T, N))
        assert validate_l2_shapes(bp, bs, ap, az, mp) == (T, N, L)

    def test_mismatched_ask_sz(self):
        T, N, L = 5, 3, 2
        bp = np.zeros((T, N, L))
        bs = np.zeros((T, N, L))
        ap = np.zeros((T, N, L))
        az = np.zeros((T, N, L + 1))  # wrong
        mp = np.zeros((T, N))
        with pytest.raises(ValueError, match="ask_sz"):
            validate_l2_shapes(bp, bs, ap, az, mp)

    def test_mismatched_mid(self):
        T, N, L = 5, 3, 2
        bp = np.zeros((T, N, L))
        bs = np.zeros((T, N, L))
        ap = np.zeros((T, N, L))
        az = np.zeros((T, N, L))
        mp = np.zeros((T, N + 1))
        with pytest.raises(ValueError, match="mid_px"):
            validate_l2_shapes(bp, bs, ap, az, mp)

    def test_label_in_error(self):
        bp = np.zeros((5, 3, 2))
        bs = np.zeros((5, 3, 2))
        ap = np.zeros((5, 3, 2))
        az = np.zeros((5, 3, 2))
        mp = np.zeros((5, 4))  # wrong
        with pytest.raises(ValueError, match="hedge_mid_px"):
            validate_l2_shapes(bp, bs, ap, az, mp, label="hedge_")


# =========================================================================
# Config
# =========================================================================


class TestConfig:
    def test_defaults(self):
        cfg = PortfolioBacktestConfig()
        assert cfg.gross_dv01_cap == 100.0
        assert cfg.top_k == 10
        assert cfg.initial_capital == 1_000_000.0

    def test_frozen(self):
        cfg = PortfolioBacktestConfig()
        with pytest.raises(AttributeError):
            cfg.top_k = 5

    def test_custom(self):
        cfg = PortfolioBacktestConfig(top_k=20, haircut=0.5)
        assert cfg.top_k == 20
        assert cfg.haircut == 0.5


# =========================================================================
# Validation
# =========================================================================


class TestValidation:
    def test_shape_mismatch_dv01(self):
        inputs = _base_inputs()
        inputs["dv01"] = np.zeros((10, 10))  # wrong shape
        with pytest.raises(ValueError, match="dv01"):
            _validate(**inputs,
                       hedge_bid_px=None, hedge_bid_sz=None,
                       hedge_ask_px=None, hedge_ask_sz=None,
                       hedge_mid_px=None, hedge_dv01=None,
                       hedge_ratios=None)

    def test_partial_hedge_arrays_rejected(self):
        T, B, H = 3, 2, 1
        inputs = _base_inputs(T, B)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H)
        with pytest.raises(ValueError, match="all-None or all-provided"):
            _validate(**inputs,
                       hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
                       hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
                       hedge_mid_px=h_mid, hedge_dv01=None,
                       hedge_ratios=None)

    def test_hedge_ratios_shape_mismatch(self):
        T, B, H = 3, 2, 1
        inputs = _base_inputs(T, B)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H)
        hedge_dv01 = np.full((T, H), 0.01)
        hedge_ratios = np.zeros((T, B + 1, H))  # wrong B dimension
        with pytest.raises(ValueError, match="hedge_ratios"):
            _validate(**inputs,
                       hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
                       hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
                       hedge_mid_px=h_mid, hedge_dv01=hedge_dv01,
                       hedge_ratios=hedge_ratios)

    def test_valid_inputs_pass(self):
        T, B, H = 3, 2, 1
        inputs = _base_inputs(T, B)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H)
        hedge_dv01 = np.full((T, H), 0.01)
        hedge_ratios = np.zeros((T, B, H))
        # Should not raise
        _validate(**inputs,
                   hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
                   hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
                   hedge_mid_px=h_mid, hedge_dv01=hedge_dv01,
                   hedge_ratios=hedge_ratios)


# =========================================================================
# run_backtest end-to-end
# =========================================================================


class TestRunBacktest:
    def test_no_hedge(self):
        """run_backtest with no hedge arrays returns valid results."""
        inputs = _base_inputs(T=3, B=2)
        res = run_backtest(**inputs)
        assert isinstance(res, PortfolioBacktestResults)
        assert res.n_trades > 0
        assert res.equity.shape == (3,)
        assert res.positions.shape == (3, 2)

    def test_with_hedge(self):
        T, B, H = 3, 2, 1
        inputs = _base_inputs(T, B)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H)
        res = run_backtest(
            **inputs,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=np.full((T, H), 0.01),
            hedge_ratios=np.full((T, B, H), -1.0),
        )
        assert res.n_trades > 0
        assert res.hedge_positions.shape == (T, H)
        # Hedge was active → positions non-zero
        assert np.any(np.abs(res.hedge_positions) > 1e-15)

    def test_custom_config(self):
        inputs = _base_inputs(T=3, B=1)
        cfg = PortfolioBacktestConfig(initial_capital=500_000.0)
        res = run_backtest(**inputs, cfg=cfg)
        # First bar with no trades → equity == initial_capital
        # But since fair >> ask, there will be trades. Just check equity plausible.
        assert res.equity[0] < 600_000.0
        assert res.equity[0] > 400_000.0

    def test_f32_input_converted(self):
        """Float32 inputs should be auto-converted without error."""
        inputs = _base_inputs(T=2, B=1)
        inputs["bid_px"] = inputs["bid_px"].astype(np.float32)
        inputs["dv01"] = inputs["dv01"].astype(np.float32)
        res = run_backtest(**inputs)
        assert isinstance(res, PortfolioBacktestResults)


# =========================================================================
# Results
# =========================================================================


class TestResults:
    def test_bar_df_columns(self):
        inputs = _base_inputs(T=3, B=2)
        res = run_backtest(**inputs)
        df = res.bar_df
        assert "equity" in df.columns
        assert "cash" in df.columns
        assert "pnl" in df.columns
        assert "position_0" in df.columns
        assert "position_1" in df.columns
        assert len(df) == 3

    def test_trade_df_columns(self):
        inputs = _base_inputs(T=3, B=2)
        res = run_backtest(**inputs)
        df = res.trade_df
        assert len(df) == res.n_trades
        assert "bar" in df.columns
        assert "instrument" in df.columns
        assert "side" in df.columns
        assert "vwap" in df.columns

    def test_hedge_columns_in_trade_df(self):
        T, B, H = 3, 1, 2
        inputs = _base_inputs(T, B)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H)
        res = run_backtest(
            **inputs,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=np.full((T, H), 0.01),
            hedge_ratios=np.full((T, B, H), -0.5),
        )
        df = res.trade_df
        assert "hedge_cost" in df.columns
        assert "hedge_size_0" in df.columns
        assert "hedge_size_1" in df.columns

    def test_bar_df_with_datetimes(self):
        inputs = _base_inputs(T=3, B=1)
        dts = np.array(["2024-01-01", "2024-01-02", "2024-01-03"],
                        dtype="datetime64[D]")
        res = run_backtest(**inputs, datetimes=dts)
        assert "datetime" in res.bar_df.columns

    def test_hedge_positions_in_bar_df(self):
        T, B, H = 3, 1, 1
        inputs = _base_inputs(T, B)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H)
        res = run_backtest(
            **inputs,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=np.full((T, H), 0.01),
            hedge_ratios=np.full((T, B, H), -1.0),
        )
        assert "hedge_position_0" in res.bar_df.columns
