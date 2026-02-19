"""Tests for PortfolioMetricsCalculator and FIFO round-trip matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest.metrics.portfolio_metrics_calculator import (
    FIFORoundTrip,
    PortfolioMetricsCalculator,
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
        assert rt.pnl == pytest.approx(200.0)  # +1 * (101 - 99) * 100

    def test_multi_instrument(self):
        rts = fifo_match(_trade_df_multi_instrument())
        assert len(rts) == 2

        rt0 = [r for r in rts if r.instrument == 0][0]
        assert rt0.holding_bars == 2
        assert rt0.pnl == pytest.approx(200.0)

        rt1 = [r for r in rts if r.instrument == 1][0]
        assert rt1.holding_bars == 4
        assert rt1.side == -1
        # Short: pnl = -1 * (100 - 102) * 50 = +100
        assert rt1.pnl == pytest.approx(100.0)

    def test_partial_fills(self):
        rts = fifo_match(_trade_df_partial_fill())
        assert len(rts) == 2
        # FIFO: first 60 matched from 100-lot entry, then 40
        rts_sorted = sorted(rts, key=lambda r: r.exit_bar)
        assert rts_sorted[0].qty == 60.0
        assert rts_sorted[0].holding_bars == 3
        assert rts_sorted[0].pnl == pytest.approx(60.0 * 2.0)  # 60 * (102-100)
        assert rts_sorted[1].qty == 40.0
        assert rts_sorted[1].holding_bars == 5
        assert rts_sorted[1].pnl == pytest.approx(40.0 * 3.0)  # 40 * (103-100)

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
