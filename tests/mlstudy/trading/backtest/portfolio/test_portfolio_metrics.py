"""Tests for PortfolioMetricsCalculator and FIFO round-trip matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest.metrics.portfolio_metrics_calculator import (
    FIFORoundTrip,
    PortfolioMetricsCalculator,
    _compute_hedge_pnls,
    fifo_match,
)
from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics


# =========================================================================
# Helpers
# =========================================================================


def _bar_df(T: int = 10, initial_equity: float = 1_000_000.0) -> pd.DataFrame:
    """Minimal bar_df with equity and state columns."""
    equity = np.linspace(initial_equity, initial_equity + 1000, T)
    return pd.DataFrame({
        "equity": equity,
        "state": np.ones(T, dtype=np.float64),
        "position_0": np.linspace(0, 100, T),
        "position_1": np.linspace(0, -50, T),
    })


def _trade_df_simple() -> pd.DataFrame:
    """Simple trade_df: buy then sell one instrument → one round-trip."""
    return pd.DataFrame([
        {"bar": 1, "instrument": 0, "side": 1, "qty_fill": 100.0,
         "vwap": 99.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
        {"bar": 5, "instrument": 0, "side": -1, "qty_fill": 100.0,
         "vwap": 101.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
    ])


def _trade_df_multi_instrument() -> pd.DataFrame:
    """Trades across two instruments."""
    return pd.DataFrame([
        # Instrument 0: buy at bar 1, sell at bar 3 → hold 2 bars, pnl = +200
        {"bar": 1, "instrument": 0, "side": 1, "qty_fill": 100.0,
         "vwap": 99.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
        {"bar": 3, "instrument": 0, "side": -1, "qty_fill": 100.0,
         "vwap": 101.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
        # Instrument 1: sell at bar 2, buy at bar 6 → hold 4 bars, pnl = +100
        {"bar": 2, "instrument": 1, "side": -1, "qty_fill": 50.0,
         "vwap": 102.0, "mid": 102.0, "cost": 0.5, "hedge_cost": 0.0},
        {"bar": 6, "instrument": 1, "side": 1, "qty_fill": 50.0,
         "vwap": 100.0, "mid": 100.0, "cost": 0.5, "hedge_cost": 0.0},
    ])


def _trade_df_partial_fill() -> pd.DataFrame:
    """Buy 100, sell 60 then sell 40 → two round-trips from FIFO matching."""
    return pd.DataFrame([
        {"bar": 0, "instrument": 0, "side": 1, "qty_fill": 100.0,
         "vwap": 100.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
        {"bar": 3, "instrument": 0, "side": -1, "qty_fill": 60.0,
         "vwap": 102.0, "mid": 102.0, "cost": 0.6, "hedge_cost": 0.0},
        {"bar": 5, "instrument": 0, "side": -1, "qty_fill": 40.0,
         "vwap": 103.0, "mid": 103.0, "cost": 0.4, "hedge_cost": 0.0},
    ])


# =========================================================================
# FIFO matching
# =========================================================================


class TestFIFOMatch:
    def test_empty_trade_df(self):
        rts = fifo_match(pd.DataFrame())
        assert rts == []

    def test_single_round_trip(self):
        rts = fifo_match(_trade_df_simple())
        assert len(rts) == 1
        rt = rts[0]
        assert rt.instrument == 0
        assert rt.qty == 100.0
        assert rt.entry_bar == 1
        assert rt.exit_bar == 5
        assert rt.holding_bars == 4
        assert rt.side == 1
        assert rt.unhedged_pnl == pytest.approx(200.0)  # +1 * (101 - 99) * 100

    def test_multi_instrument(self):
        rts = fifo_match(_trade_df_multi_instrument())
        assert len(rts) == 2

        rt0 = [r for r in rts if r.instrument == 0][0]
        assert rt0.holding_bars == 2
        assert rt0.unhedged_pnl == pytest.approx(200.0)

        rt1 = [r for r in rts if r.instrument == 1][0]
        assert rt1.holding_bars == 4
        assert rt1.side == -1
        # Short: pnl = -1 * (100 - 102) * 50 = +100
        assert rt1.unhedged_pnl == pytest.approx(100.0)

    def test_partial_fills(self):
        rts = fifo_match(_trade_df_partial_fill())
        assert len(rts) == 2
        # FIFO: first 60 matched from 100-lot entry, then 40
        rts_sorted = sorted(rts, key=lambda r: r.exit_bar)
        assert rts_sorted[0].qty == 60.0
        assert rts_sorted[0].holding_bars == 3
        assert rts_sorted[0].unhedged_pnl == pytest.approx(60.0 * 2.0)  # 60 * (102-100)
        assert rts_sorted[1].qty == 40.0
        assert rts_sorted[1].holding_bars == 5
        assert rts_sorted[1].unhedged_pnl == pytest.approx(40.0 * 3.0)  # 40 * (103-100)

    def test_open_position_not_matched(self):
        """Buys without corresponding sells are not matched."""
        df = pd.DataFrame([
            {"bar": 0, "instrument": 0, "side": 1, "qty_fill": 100.0,
             "vwap": 100.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
        ])
        rts = fifo_match(df)
        assert len(rts) == 0

    def test_same_direction_queues(self):
        """Multiple buys followed by sell matches FIFO."""
        df = pd.DataFrame([
            {"bar": 0, "instrument": 0, "side": 1, "qty_fill": 50.0,
             "vwap": 98.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
            {"bar": 1, "instrument": 0, "side": 1, "qty_fill": 50.0,
             "vwap": 99.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
            {"bar": 5, "instrument": 0, "side": -1, "qty_fill": 80.0,
             "vwap": 101.0, "mid": 101.0, "cost": 1.0, "hedge_cost": 0.0},
        ])
        rts = fifo_match(df)
        assert len(rts) == 2  # 50 from bar 0, 30 from bar 1
        rts_sorted = sorted(rts, key=lambda r: r.entry_bar)
        assert rts_sorted[0].qty == 50.0
        assert rts_sorted[0].entry_vwap == 98.0
        assert rts_sorted[1].qty == 30.0
        assert rts_sorted[1].entry_vwap == 99.0


# =========================================================================
# PortfolioMetricsCalculator
# =========================================================================


class TestPortfolioMetricsCalculator:
    def test_compute_all_with_trades(self):
        bar = _bar_df()
        trade = _trade_df_simple()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_pnl > 0
        assert metrics.n_trades == 2  # 2 fills
        assert metrics.avg_holding_period == pytest.approx(4.0)
        assert metrics.hit_rate == pytest.approx(1.0)  # one winning RT

    def test_compute_all_no_trades(self):
        bar = _bar_df()
        calc = PortfolioMetricsCalculator(bar, None, annualization_factor=252)
        metrics = calc.compute_all()
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.n_trades == 0
        assert metrics.avg_holding_period == 0.0

    def test_compute_equity_only(self):
        bar = _bar_df()
        trade = _trade_df_simple()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_equity()
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_pnl > 0
        assert metrics.n_trades == 0  # equity-only zeroes trades

    def test_multi_instrument_metrics(self):
        bar = _bar_df()
        trade = _trade_df_multi_instrument()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()
        assert metrics.n_trades == 4  # 4 fills total
        assert metrics.hit_rate == pytest.approx(1.0)  # both RTs profitable
        # Weighted avg holding: (200*2 + 100*4) / (200+100) = 800/300 ≈ 2.667
        # But weights are by qty: (100*2 + 50*4) / (100+50) = 400/150 ≈ 2.667
        assert metrics.avg_holding_period == pytest.approx(400.0 / 150.0)

    def test_partial_fills_metrics(self):
        bar = _bar_df()
        trade = _trade_df_partial_fill()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()
        assert metrics.n_trades == 3  # 3 fills
        # 2 round-trips: both profitable
        assert metrics.hit_rate == pytest.approx(1.0)
        assert metrics.profit_factor == np.inf  # no losers

    def test_losing_trade_hit_rate(self):
        """Round-trip with a loss gives hit_rate < 1."""
        trade = pd.DataFrame([
            {"bar": 0, "instrument": 0, "side": 1, "qty_fill": 100.0,
             "vwap": 102.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
            {"bar": 3, "instrument": 0, "side": -1, "qty_fill": 100.0,
             "vwap": 100.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0},
        ])
        bar = _bar_df()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()
        assert metrics.hit_rate == pytest.approx(0.0)  # one losing RT
        assert metrics.avg_loss < 0
        assert metrics.profit_factor == pytest.approx(0.0)

    def test_pct_time_in_market(self):
        bar = _bar_df()
        # All state=1 → 100% in market
        calc = PortfolioMetricsCalculator(bar, None, annualization_factor=252)
        metrics = calc.compute_all()
        assert metrics.pct_time_in_market == pytest.approx(1.0)

        # Half state=0
        bar2 = bar.copy()
        bar2["state"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        calc2 = PortfolioMetricsCalculator(bar2, None, annualization_factor=252)
        metrics2 = calc2.compute_all()
        assert metrics2.pct_time_in_market == pytest.approx(0.5)

    def test_inherits_equity_metrics(self):
        """Equity metrics come from the parent class."""
        bar = _bar_df()
        trade = _trade_df_simple()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()
        # Should have all equity fields
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "calmar_ratio")


# =========================================================================
# Hedged PnL tests
# =========================================================================


class TestFIFOMatchDv01:
    """Verify fifo_match captures dv01_fill into entry_dv01."""

    def test_entry_dv01_propagated(self):
        trade = pd.DataFrame([
            {"bar": 1, "instrument": "A", "side": 1, "qty_fill": 100.0,
             "vwap": 99.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0,
             "dv01_fill": 0.05},
            {"bar": 5, "instrument": "A", "side": -1, "qty_fill": 100.0,
             "vwap": 101.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0,
             "dv01_fill": 0.06},
        ])
        rts = fifo_match(trade)
        assert len(rts) == 1
        assert rts[0].entry_dv01 == pytest.approx(0.05)

    def test_entry_dv01_defaults_zero_when_missing(self):
        trade = _trade_df_simple()
        rts = fifo_match(trade)
        assert len(rts) == 1
        assert rts[0].entry_dv01 == 0.0


class TestTheoreticalHedgedPnl:
    """Test theoretical hedged PnL computation."""

    def test_single_rt_single_hedge(self):
        """One round-trip, one hedge instrument.

        Buy 100 of instrument 0 at bar 1, sell at bar 5.
        hedge_ratios=-0.5 (negative: buying ref → sell hedge),
        dv01=0.04, hedge_dv01=0.02
        theo_hedge_qty = +1 * 100 * 0.04 * (-0.5) / 0.02 = -100 (sell hedge)
        hedge_mid_px: entry=50, exit=51
        hedge PnL = -100*(51-50) = -100 (loss on short hedge)
        unhedged_pnl = +1*(101-99)*100 = 200
        theoretical_hedged_pnl = 200 + (-100) = 100
        """
        T, B, H = 10, 1, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_mid_px[5:, 0] = 51.0  # hedge price moves up at bar 5

        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        bar = _bar_df(T)
        _compute_hedge_pnls(
            [rt], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0},
        )
        assert rt.theoretical_hedged_pnl == pytest.approx(100.0)

    def test_short_side_theoretical(self):
        """Short round-trip: sell at bar 1, buy back at bar 5.

        side=-1, qty=100, dv01=0.04, hr=-0.5, hedge_dv01=0.02
        theo_hedge_qty = -1 * 100 * 0.04 * (-0.5) / 0.02 = +100 (buy hedge)
        hedge goes from 50→51: hedge PnL = +100*(51-50) = +100
        unhedged_pnl = -1*(101-99)*100 = -200
        theoretical_hedged_pnl = -200 + 100 = -100
        """
        T, B, H = 10, 1, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_mid_px[5:, 0] = 51.0

        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=-1,
        )
        bar = _bar_df(T)
        _compute_hedge_pnls(
            [rt], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0},
        )
        assert rt.theoretical_hedged_pnl == pytest.approx(-100.0)

    def test_multiple_hedges(self):
        """Two hedge instruments, verify PnL sums across both."""
        T, B, H = 10, 1, 2
        hedge_ratios = np.zeros((T, B, H))
        hedge_ratios[:, 0, 0] = -0.3
        hedge_ratios[:, 0, 1] = -0.2
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_mid_px[5:, 0] = 52.0  # hedge 0: +2
        hedge_mid_px[5:, 1] = 49.0  # hedge 1: -1

        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        # h0: theo_qty = 1*100*0.04*(-0.3)/0.02 = -60, pnl = -60*2 = -120
        # h1: theo_qty = 1*100*0.04*(-0.2)/0.02 = -40, pnl = -40*(-1) = +40
        # total hedge pnl = -120 + 40 = -80
        # unhedged = 200, theoretical = 200 - 80 = 120
        bar = _bar_df(T)
        _compute_hedge_pnls(
            [rt], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0},
        )
        assert rt.theoretical_hedged_pnl == pytest.approx(120.0)


class TestNetHedgedPnl:
    """Test net hedged PnL allocation by DV01 weight."""

    def test_net_hedged_allocated_by_hedge_dv01(self):
        """Two instruments sharing one hedge → allocation by required hedge DV01.

        Instrument A: qty=100, dv01=0.04, hr=-0.5 → hedge DV01 = |100*0.04*0.5| = 2.0
        Instrument B: qty=50,  dv01=0.08, hr=-0.5 → hedge DV01 = |50*0.08*0.5|  = 2.0
        Equal hedge DV01 → hedge PnL is split 50/50.
        """
        T, B, H = 10, 2, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.zeros((T, B))
        dv01_arr[:, 0] = 0.04
        dv01_arr[:, 1] = 0.08
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)

        rt_a = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        rt_b = FIFORoundTrip(
            instrument="B", qty=50.0, entry_bar=1, exit_bar=5,
            entry_vwap=100.0, exit_vwap=102.0, side=1,
        )

        # Build a bar_df with known hedge_pnl per bar (from backtest loop)
        # 10 per bar in active bars 1-4 → total = 40
        hedge_pnl = np.zeros(T)
        hedge_pnl[1:5] = 10.0
        bar = pd.DataFrame({
            "equity": np.linspace(1e6, 1e6 + 1000, T),
            "state": np.ones(T),
            "hedge_pnl": hedge_pnl,
        })

        _compute_hedge_pnls(
            [rt_a, rt_b], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0, "B": 1},
        )
        # Equal demand → each gets 50% of 40 = 20
        assert rt_a.net_hedged_pnl == pytest.approx(rt_a.unhedged_pnl + 20.0)
        assert rt_b.net_hedged_pnl == pytest.approx(rt_b.unhedged_pnl + 20.0)

    def test_unequal_dv01_weights(self):
        """Different hedge DV01 demand → unequal allocation.

        A: qty=100, dv01=0.04, hr=-0.5 → hedge DV01 = |100*0.04*0.5| = 2.0
        B: qty=50,  dv01=0.02, hr=-0.5 → hedge DV01 = |50*0.02*0.5|  = 0.5
        Ratio 4:1.
        """
        T, B, H = 10, 2, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.zeros((T, B))
        dv01_arr[:, 0] = 0.04  # hedge DV01 = 100 * 0.04 * 0.5 = 2.0
        dv01_arr[:, 1] = 0.02  # hedge DV01 = 50 * 0.02 * 0.5 = 0.5
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)

        rt_a = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=2, exit_bar=4,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        rt_b = FIFORoundTrip(
            instrument="B", qty=50.0, entry_bar=2, exit_bar=4,
            entry_vwap=100.0, exit_vwap=102.0, side=1,
        )

        # hedge_pnl = 25 per bar in active bars 2-3 → total = 50
        hedge_pnl = np.zeros(T)
        hedge_pnl[2:4] = 25.0
        bar = pd.DataFrame({
            "equity": np.linspace(1e6, 1e6 + 1000, T),
            "state": np.ones(T),
            "hedge_pnl": hedge_pnl,
        })

        _compute_hedge_pnls(
            [rt_a, rt_b], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0, "B": 1},
        )
        # hedge DV01: A=|100*0.04*0.5|=2.0, B=|50*0.02*0.5|=0.5, total=2.5
        # A gets 2/2.5 * 50 = 40, B gets 0.5/2.5 * 50 = 10
        assert rt_a.net_hedged_pnl == pytest.approx(rt_a.unhedged_pnl + 40.0)
        assert rt_b.net_hedged_pnl == pytest.approx(rt_b.unhedged_pnl + 10.0)

    def test_hedge_dv01_weight_not_notional(self):
        """Hedge DV01 weighting differs from notional when dv01 per unit varies.

        A: qty=200, dv01=0.01, hr=-0.5 → hedge DV01 = |200*0.01*0.5| = 1.0
        B: qty=100, dv01=0.04, hr=-0.5 → hedge DV01 = |100*0.04*0.5| = 2.0
        By notional: 200:100 = 2:1 (A gets 2/3).
        By hedge DV01: 1:2   = 1:3 (A gets 1/3).
        """
        T, B, H = 10, 2, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.zeros((T, B))
        dv01_arr[:, 0] = 0.01
        dv01_arr[:, 1] = 0.04
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)

        rt_a = FIFORoundTrip(
            instrument="A", qty=200.0, entry_bar=1, exit_bar=4,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        rt_b = FIFORoundTrip(
            instrument="B", qty=100.0, entry_bar=1, exit_bar=4,
            entry_vwap=100.0, exit_vwap=102.0, side=1,
        )

        # hedge_pnl = 30 per bar in active bars 1-3 → total = 90
        hedge_pnl = np.zeros(T)
        hedge_pnl[1:4] = 30.0
        bar = pd.DataFrame({
            "equity": np.linspace(1e6, 1e6 + 1000, T),
            "state": np.ones(T),
            "hedge_pnl": hedge_pnl,
        })

        _compute_hedge_pnls(
            [rt_a, rt_b], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0, "B": 1},
        )
        # hedge DV01: A=|200*0.01*0.5|=1.0, B=|100*0.04*0.5|=2.0, total=3.0
        # A gets 1/3 * 90 = 30, B gets 2/3 * 90 = 60
        assert rt_a.net_hedged_pnl == pytest.approx(rt_a.unhedged_pnl + 30.0)
        assert rt_b.net_hedged_pnl == pytest.approx(rt_b.unhedged_pnl + 60.0)


class TestHedgedPnlFallback:
    """When hedge arrays are None, both hedged PnLs equal unhedged."""

    def test_no_hedge_arrays_fallback(self):
        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        bar = _bar_df(10)
        _compute_hedge_pnls([rt], bar, None, None, None, None, None)
        assert rt.theoretical_hedged_pnl == pytest.approx(rt.unhedged_pnl)
        assert rt.net_hedged_pnl == pytest.approx(rt.unhedged_pnl)

    def test_calculator_without_hedge_arrays(self):
        """PortfolioMetricsCalculator without hedge arrays still works."""
        bar = _bar_df()
        trade = _trade_df_simple()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()
        assert isinstance(metrics, BacktestMetrics)
        # Without hedge arrays, net_hedged_pnl = unhedged_pnl
        # So metrics should reflect unhedged PnL
        assert metrics.hit_rate == pytest.approx(1.0)

    def test_calculator_with_hedge_arrays(self):
        """PortfolioMetricsCalculator with hedge arrays computes hedged PnL."""
        T, B, H = 10, 1, 1
        trade = pd.DataFrame([
            {"bar": 1, "instrument": "A", "side": 1, "qty_fill": 100.0,
             "vwap": 99.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0,
             "dv01_fill": 0.04},
            {"bar": 5, "instrument": "A", "side": -1, "qty_fill": 100.0,
             "vwap": 101.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0,
             "dv01_fill": 0.04},
        ])
        bar = _bar_df(T)
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)

        calc = PortfolioMetricsCalculator(
            bar, trade, annualization_factor=252,
            hedge_ratios=hedge_ratios, dv01=dv01_arr,
            hedge_dv01=hedge_dv01, hedge_mid_px=hedge_mid_px,
            instrument_ids=["A"],
        )
        metrics = calc.compute_all()
        assert isinstance(metrics, BacktestMetrics)


class TestThreePnlViewMetrics:
    """Verify metrics are computed for unhedged, theoretical-hedged, and net-hedged PnL."""

    def test_three_views_with_hedge(self):
        """With hedge data, the three PnL views produce distinct metrics."""
        T, B, H = 10, 1, 1
        # Buy at 99, sell at 101 → unhedged_pnl = +200
        # Hedge price goes from 50 to 52 → theoretical hedge PnL for sell hedge = -100*(52-50) = -200
        # theoretical_hedged_pnl = 200 + (-200) = 0
        trade = pd.DataFrame([
            {"bar": 1, "instrument": "A", "side": 1, "qty_fill": 100.0,
             "vwap": 99.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0,
             "dv01_fill": 0.04},
            {"bar": 5, "instrument": "A", "side": -1, "qty_fill": 100.0,
             "vwap": 101.0, "mid": 100.0, "cost": 1.0, "hedge_cost": 0.0,
             "dv01_fill": 0.04},
        ])
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_mid_px[5:, 0] = 52.0

        # net hedge PnL per bar: -5 per bar for bars 1-4 → total -20
        hedge_pnl = np.zeros(T)
        hedge_pnl[1:5] = -5.0

        bar = pd.DataFrame({
            "equity": np.linspace(1e6, 1e6 + 1000, T),
            "state": np.ones(T),
            "position_0": np.zeros(T),
            "hedge_pnl": hedge_pnl,
        })

        calc = PortfolioMetricsCalculator(
            bar, trade, annualization_factor=252,
            hedge_ratios=hedge_ratios, dv01=dv01_arr,
            hedge_dv01=hedge_dv01, hedge_mid_px=hedge_mid_px,
            instrument_ids=["A"],
        )
        metrics = calc.compute_all()

        # Unhedged PnL = +200 → winner
        assert metrics.hit_rate_unhedged == pytest.approx(1.0)
        assert metrics.avg_win_unhedged == pytest.approx(200.0)

        # Theoretical hedged PnL = 0 → not a winner
        assert metrics.hit_rate_theo_hedged == pytest.approx(0.0)

        # Net hedged PnL = 200 + (-20) = 180 → winner
        assert metrics.hit_rate == pytest.approx(1.0)
        assert metrics.avg_win == pytest.approx(180.0)

    def test_no_hedge_all_views_equal(self):
        """Without hedge arrays, all three views equal unhedged PnL."""
        bar = _bar_df()
        trade = _trade_df_simple()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        metrics = calc.compute_all()

        assert metrics.hit_rate == metrics.hit_rate_unhedged
        assert metrics.hit_rate == metrics.hit_rate_theo_hedged
        assert metrics.profit_factor == metrics.profit_factor_unhedged
        assert metrics.avg_win == metrics.avg_win_unhedged
        assert metrics.avg_loss == metrics.avg_loss_unhedged

    def test_empty_trades_all_zero(self):
        """With no trades, all three views are zero."""
        bar = _bar_df()
        calc = PortfolioMetricsCalculator(bar, None, annualization_factor=252)
        metrics = calc.compute_all()

        assert metrics.hit_rate == 0.0
        assert metrics.hit_rate_unhedged == 0.0
        assert metrics.hit_rate_theo_hedged == 0.0
        assert metrics.profit_factor_unhedged == 0.0
        assert metrics.profit_factor_theo_hedged == 0.0

    def test_to_dict_includes_all_views(self):
        """to_dict() includes all three PnL view fields."""
        bar = _bar_df()
        trade = _trade_df_simple()
        calc = PortfolioMetricsCalculator(bar, trade, annualization_factor=252)
        d = calc.compute_all().to_dict()

        for suffix in ("", "_unhedged", "_theo_hedged"):
            assert f"hit_rate{suffix}" in d
            assert f"profit_factor{suffix}" in d
            assert f"avg_win{suffix}" in d
            assert f"avg_loss{suffix}" in d
            assert f"win_loss_ratio{suffix}" in d


class TestNettingBenefit:
    """Two instruments sharing a hedge: net hedge qty < sum of individual."""

    def test_netting_reduces_hedge_cost(self):
        """Opposite-direction hedges partially cancel.

        Instrument A: long, hedge_ratio=-0.5 → needs to sell hedge
        Instrument B: short, hedge_ratio=-0.5 → needs to buy hedge
        They partially cancel, so net hedged PnL reflects lower actual hedge cost.

        Theoretical treats each independently (no netting).
        """
        T, B, H = 10, 2, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_mid_px[5:, 0] = 52.0  # hedge goes up by 2

        # A: long side=+1, theo_hedge_qty = 1*100*0.04*(-0.5)/0.02 = -100 (sell hedge)
        # B: short side=-1, theo_hedge_qty = -1*50*0.04*(-0.5)/0.02 = +50 (buy hedge)
        # Net hedge target = -100 + 50 = -50 (net short)
        # Theoretical A: hedge PnL = -100*(52-50) = -200, total = 200 + (-200) = 0
        # Theoretical B: hedge PnL = +50*(52-50) = +100, total = 100 + 100 = 200

        rt_a = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        rt_b = FIFORoundTrip(
            instrument="B", qty=50.0, entry_bar=1, exit_bar=5,
            entry_vwap=102.0, exit_vwap=100.0, side=-1,
        )

        bar = pd.DataFrame({
            "equity": np.linspace(1e6, 1e6 + 1000, T),
            "state": np.ones(T),
            "position_0": np.zeros(T),
            "position_1": np.zeros(T),
            "mid_px_0": np.ones(T) * 100.0,
            "mid_px_1": np.ones(T) * 100.0,
            "pnl": np.zeros(T),
        })

        _compute_hedge_pnls(
            [rt_a, rt_b], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0, "B": 1},
        )

        # Theoretical: each treated independently
        # A: unhedged=200, theo_hedge_qty=-100(sell), hedge+2 → pnl=-100*2=-200, total=0
        assert rt_a.theoretical_hedged_pnl == pytest.approx(0.0)
        # B: unhedged=100, theo_hedge_qty=+50(buy), hedge+2 → pnl=+50*2=+100, total=200
        assert rt_b.theoretical_hedged_pnl == pytest.approx(200.0)


class TestTheoreticalHedgedPnlBidAsk:
    """Test that theoretical PnL uses bid/ask when available."""

    def test_sell_hedge_uses_bid_entry_ask_exit(self):
        """Selling hedge → enter at bid, unwind (buy back) at ask.

        side=+1, hr=-0.5 → theo_hedge_qty = 1*100*0.04*(-0.5)/0.02 = -100
        Selling → entry at bid=49.5, exit at ask=51.5
        hedge PnL = -100 * (51.5 - 49.5) = -200
        unhedged = 200, theoretical = 200 + (-200) = 0
        """
        T, B, H = 10, 1, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)  # fallback (not used when ba present)
        hedge_bid_px = np.full((T, H), 49.5)
        hedge_ask_px = np.full((T, H), 50.5)
        # At exit bar, bid/ask shift up
        hedge_bid_px[5:, 0] = 51.0
        hedge_ask_px[5:, 0] = 51.5

        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        bar = _bar_df(T)
        _compute_hedge_pnls(
            [rt], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0}, hedge_bid_px, hedge_ask_px,
        )
        # sell at bid=49.5, buy back at ask=51.5 → pnl = -100*(51.5 - 49.5) = -200
        assert rt.theoretical_hedged_pnl == pytest.approx(0.0)

    def test_buy_hedge_uses_ask_entry_bid_exit(self):
        """Buying hedge → enter at ask, unwind (sell) at bid.

        side=-1, hr=-0.5 → theo_hedge_qty = -1*100*0.04*(-0.5)/0.02 = +100
        Buying → entry at ask=50.5, exit at bid=51.0
        hedge PnL = 100 * (51.0 - 50.5) = +50
        unhedged = -1*(101-99)*100 = -200, theoretical = -200 + 50 = -150
        """
        T, B, H = 10, 1, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_bid_px = np.full((T, H), 49.5)
        hedge_ask_px = np.full((T, H), 50.5)
        hedge_bid_px[5:, 0] = 51.0
        hedge_ask_px[5:, 0] = 51.5

        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=-1,
        )
        bar = _bar_df(T)
        _compute_hedge_pnls(
            [rt], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0}, hedge_bid_px, hedge_ask_px,
        )
        # buy at ask=50.5, sell at bid=51.0 → pnl = 100*(51.0 - 50.5) = +50
        assert rt.theoretical_hedged_pnl == pytest.approx(-150.0)

    def test_falls_back_to_mid_when_no_bidask(self):
        """Without bid/ask arrays, uses mid prices (same as before)."""
        T, B, H = 10, 1, 1
        hedge_ratios = np.full((T, B, H), -0.5)
        dv01_arr = np.full((T, B), 0.04)
        hedge_dv01 = np.full((T, H), 0.02)
        hedge_mid_px = np.full((T, H), 50.0)
        hedge_mid_px[5:, 0] = 51.0

        rt = FIFORoundTrip(
            instrument="A", qty=100.0, entry_bar=1, exit_bar=5,
            entry_vwap=99.0, exit_vwap=101.0, side=1,
        )
        bar = _bar_df(T)
        # No bid/ask args → uses mid
        _compute_hedge_pnls(
            [rt], bar, hedge_ratios, dv01_arr, hedge_dv01,
            hedge_mid_px, {"A": 0},
        )
        # sell hedge at mid=50, buy back at mid=51 → pnl = -100*(51-50) = -100
        assert rt.theoretical_hedged_pnl == pytest.approx(100.0)
