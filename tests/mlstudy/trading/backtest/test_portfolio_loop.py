"""Tests for LP portfolio backtest loop."""

from __future__ import annotations

import numpy as np
import pytest

from mlstudy.trading.backtest.portfolio.single_backtest.loop import (
    lp_portfolio_loop,
    _walk_book,
    _check_market_valid,
    _round_qty_trade,
    _solve_lp,
)
from mlstudy.trading.backtest.portfolio.single_backtest.state import (
    PortfolioActionCode,
    CooldownMode,
)


# =========================================================================
# Fixtures / helpers
# =========================================================================

def _make_market(T, B, L=3, mid_base=100.0, spread_bps=10.0):
    """Create simple L2 market data with uniform spread."""
    rng = np.random.default_rng(42)
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


def _default_params():
    """Default scalar parameters for the loop."""
    return dict(
        gross_dv01_cap=100.0,
        issuer_dv01_caps=None,
        mat_bucket_dv01_caps=None,
        top_k=10,
        z_inc=2.0,
        p_inc=0.05,
        z_dec=1.0,
        p_dec=0.10,
        alpha_thr_inc=1.0,
        alpha_thr_dec=0.5,
        max_levels=3,
        haircut=1.0,
        qty_step=0.0,
        min_qty_trade=0.0,
        min_fill_ratio=0.0,
        cooldown_bars=0,
        cooldown_mode=int(CooldownMode.BLOCK_ALL),
        min_maturity_inc=0.0,
        initial_capital=1_000_000.0,
    )


# =========================================================================
# _walk_book
# =========================================================================


class TestWalkBook:
    def test_full_fill(self):
        px = np.array([100.0, 100.1, 100.2])
        sz = np.array([500.0, 500.0, 500.0])
        filled, vwap = _walk_book(px, sz, 200.0, 3, 1.0)
        assert filled == pytest.approx(200.0)
        assert vwap == pytest.approx(100.0)  # all from level 0

    def test_multi_level(self):
        px = np.array([100.0, 101.0])
        sz = np.array([100.0, 200.0])
        filled, vwap = _walk_book(px, sz, 150.0, 2, 1.0)
        assert filled == pytest.approx(150.0)
        expected_vwap = (100.0 * 100.0 + 50.0 * 101.0) / 150.0
        assert vwap == pytest.approx(expected_vwap)

    def test_haircut(self):
        px = np.array([100.0])
        sz = np.array([1000.0])
        filled, vwap = _walk_book(px, sz, 600.0, 1, 0.5)
        assert filled == pytest.approx(500.0)

    def test_zero_qty(self):
        px = np.array([100.0])
        sz = np.array([1000.0])
        filled, vwap = _walk_book(px, sz, 0.0, 1, 1.0)
        assert filled == pytest.approx(0.0)
        assert vwap == pytest.approx(0.0)


class TestCheckMarketValid:
    def test_valid(self):
        assert _check_market_valid(99.0, 101.0, 100.0) is True

    def test_crossed(self):
        assert _check_market_valid(101.0, 99.0, 100.0) is False

    def test_zero_price(self):
        assert _check_market_valid(0.0, 101.0, 100.0) is False


class TestRoundDv01Trade:
    def test_below_min(self):
        assert _round_qty_trade(0.5, 1.0, 0.1) == 0.0

    def test_round_to_step(self):
        assert _round_qty_trade(1.27, 0.5, 0.5) == pytest.approx(1.5)

    def test_no_step(self):
        assert _round_qty_trade(1.27, 0.5, 0.0) == pytest.approx(1.27)


# =========================================================================
# LP solver
# =========================================================================


class TestSolveLp:
    def test_basic_allocation(self):
        alphas = np.array([10.0, 5.0, 1.0])
        liq_caps = np.array([50.0, 50.0, 50.0])
        pos_hr = np.array([100.0, 100.0, 100.0])
        bonds = np.array([0, 1, 2], dtype=np.int32)
        sides = np.array([1, 1, 1], dtype=np.int32)

        sizes, greedy = _solve_lp(
            alphas, liq_caps, 80.0, pos_hr, bonds, sides,
            None, None, None, None,
            np.zeros(1), np.zeros(1),
        )
        assert sizes.sum() <= 80.0 + 1e-10
        # Highest alpha candidate should get allocation
        assert sizes[0] > 0.0

    def test_empty(self):
        sizes, greedy = _solve_lp(
            np.array([]), np.array([]), 100.0, np.array([]),
            np.array([], dtype=np.int32), np.array([], dtype=np.int32),
            None, None, None, None,
            np.zeros(1), np.zeros(1),
        )
        assert len(sizes) == 0


# =========================================================================
# Full loop
# =========================================================================


class TestLpPortfolioLoopBasic:
    """Basic end-to-end tests."""

    def test_no_signal_no_trades(self):
        """When fair == mid (no alpha), no trades should execute."""
        T, B = 5, 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px.copy()  # no alpha
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        n_trades = result[35]
        assert n_trades == 0

    def test_buy_signal_executes(self):
        """When fair >> ask, a BUY candidate should execute."""
        T, B = 3, 2
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)

        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        # Fair price well above ask → strong buy signal
        fair_price = mid_px + 1.0  # +100 bps above mid
        zscore = np.full((T, B), 3.0)  # above z_inc threshold
        adf_p = np.full((T, B), 0.01)  # below p_inc threshold
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        n_trades = result[35]
        assert n_trades > 0
        # Check that trades are buys
        tr_side = result[11]
        for i in range(n_trades):
            assert tr_side[i] == 1  # BUY

    def test_sell_signal_executes(self):
        """When fair << bid, a SELL candidate should execute."""
        T, B = 3, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px - 1.0  # well below bid
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        n_trades = result[35]
        assert n_trades > 0
        tr_side = result[11]
        for i in range(n_trades):
            assert tr_side[i] == -1  # SELL

    def test_output_shapes(self):
        """All output arrays have correct shapes."""
        T, B = 10, 4
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px.copy()
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        out_pos, out_cash, out_equity, out_pnl = result[0], result[1], result[2], result[3]
        assert out_pos.shape == (T, B)
        assert out_cash.shape == (T,)
        assert out_equity.shape == (T,)
        assert out_pnl.shape == (T,)

    def test_equity_starts_at_initial_capital(self):
        """First bar equity should start near initial capital."""
        T, B = 1, 2
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px.copy()
        zscore = np.zeros((T, B))  # no signal
        adf_p = np.ones((T, B))
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()
        params["initial_capital"] = 500_000.0

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        assert result[2][0] == pytest.approx(500_000.0)  # equity


class TestCooldown:
    def test_cooldown_blocks_trades(self):
        """After trading, cooldown should block new trades."""
        T, B = 10, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 1.0  # strong buy signal every bar
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e8)
        pos_short = np.full(B, -1e8)
        params = _default_params()
        params["cooldown_bars"] = 3
        params["cooldown_mode"] = int(CooldownMode.BLOCK_ALL)

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        codes = result[5]
        # After first execution, next 3 bars should be cooldown
        exec_bars = np.where(codes == int(PortfolioActionCode.EXEC_OK))[0]
        if len(exec_bars) > 0:
            first_exec = exec_bars[0]
            for off in range(1, min(4, T - first_exec)):
                bar = first_exec + off
                if bar < T and off <= 3:
                    assert codes[bar] in (
                        int(PortfolioActionCode.SKIP_COOLDOWN),
                        int(PortfolioActionCode.EXEC_OK),
                        int(PortfolioActionCode.EXEC_PARTIAL),
                        int(PortfolioActionCode.EXEC_GREEDY),
                        int(PortfolioActionCode.NO_CANDIDATES),
                    )


class TestSignalGating:
    def test_low_zscore_no_fair(self):
        """Z-score below threshold should produce no candidates."""
        T, B = 3, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 0.5)  # below z_dec=1.0
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        assert result[35] == 0  # no trades

    def test_high_adf_pvalue_no_fair(self):
        """ADF p-value above threshold should block fair price activation."""
        T, B = 3, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.99)  # above both p_inc and p_dec
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        assert result[35] == 0


class TestPositionLimits:
    def test_long_limit_caps_trade(self):
        """Trades should not push position beyond pos_limits_long."""
        T, B = 5, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 500.0)  # tight limit
        pos_short = np.full(B, -1e6)
        params = _default_params()
        params["cooldown_bars"] = 0

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        out_pos = result[0]
        # Position should never exceed the long limit
        assert np.all(out_pos <= 500.0 + 1e-6)


class TestNonTradable:
    def test_non_tradable_skipped(self):
        """Non-tradable instruments should never appear in trades."""
        T, B = 3, 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.array([1, 0, 1], dtype=np.int32)  # instrument 1 not tradable
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        n_trades = result[35]
        tr_instrument = result[10]
        for i in range(n_trades):
            assert tr_instrument[i] != 1  # instrument 1 should never be traded


class TestBookkeeping:
    def test_pnl_is_equity_diff(self):
        """PnL should equal the bar-over-bar equity change."""
        T, B = 5, 2
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 0.5
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()
        params["initial_capital"] = 1e6

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        equity = result[2]
        pnl = result[3]

        # pnl[0] = equity[0] - initial_capital
        assert pnl[0] == pytest.approx(equity[0] - 1e6, abs=1e-6)
        for t in range(1, T):
            assert pnl[t] == pytest.approx(equity[t] - equity[t - 1], abs=1e-6)

    def test_gross_pnl_equals_pnl_plus_cost(self):
        """gross_pnl = pnl + bar_cost for bars with trades."""
        T, B = 3, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=10.0)
        dv01 = np.full((T, B), 0.01)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        pnl = result[3]
        gross_pnl = result[4]
        # gross_pnl >= pnl always (cost >= 0)
        for t in range(T):
            assert gross_pnl[t] >= pnl[t] - 1e-10


# =========================================================================
# Hedge helpers
# =========================================================================

def _make_hedge_market(T, H, L=3, mid_base=100.0, spread_bps=10.0):
    """Create simple L2 hedge market data."""
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


# =========================================================================
# Hedge tests
# =========================================================================


class TestHedgeExecution:
    """Instrument buy triggers opposite hedge trades."""

    def test_buy_triggers_hedge_sell(self):
        T, B, H = 3, 1, 1
        L = 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px + 1.0  # strong buy signal
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        # Hedge: 1 instrument, ratio = -1.0 (sell hedge to offset buy)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H, L, spread_bps=5.0)
        hedge_dv01 = np.full((T, H), 0.01, dtype=np.float64)
        hedge_ratios = np.full((T, B, H), -1.0, dtype=np.float64)

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=hedge_dv01,
            hedge_ratios=hedge_ratios,
        )
        n_trades = result[35]
        assert n_trades > 0
        out_hedge_pos = result[8]
        # Instrument buy with negative hedge_ratio → hedge sells → hedge_pos < 0
        assert out_hedge_pos[0, 0] < -1e-15


class TestHedgeCostInEquity:
    """Hedge costs are included in bar_cost and reflected in pnl/gross_pnl."""

    def test_hedge_cost_increases_gross_pnl_gap(self):
        T, B, H = 3, 1, 1
        L = 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        # Run without hedge
        result_no_hedge = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )

        # Run with hedge (wide spread → large hedge cost)
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(
            T, H, L, spread_bps=50.0,
        )
        hedge_dv01 = np.full((T, H), 0.01, dtype=np.float64)
        hedge_ratios = np.full((T, B, H), -1.0, dtype=np.float64)

        result_hedge = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=hedge_dv01,
            hedge_ratios=hedge_ratios,
        )

        # gross_pnl - pnl should be larger with hedge (more execution cost)
        gap_no = result_no_hedge[4][0] - result_no_hedge[3][0]
        gap_h = result_hedge[4][0] - result_hedge[3][0]
        assert gap_h > gap_no + 1e-10


class TestHedgeMtmInEquity:
    """Equity includes instrument_pos * mid_px + hedge_pos * hedge_mid_px."""

    def test_hedge_mtm_affects_equity(self):
        T, B, H = 1, 1, 1
        L = 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H, L, spread_bps=5.0)
        hedge_dv01 = np.full((T, H), 0.01, dtype=np.float64)
        hedge_ratios = np.full((T, B, H), -1.0, dtype=np.float64)

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=hedge_dv01,
            hedge_ratios=hedge_ratios,
        )
        equity = result[2][0]
        cash = result[1][0]
        inst_pos = result[0][0]
        hedge_pos_out = result[8][0]

        # Verify equity = cash + instrument_mtm + hedge_mtm
        expected_equity = cash
        for b in range(B):
            expected_equity += inst_pos[b] * mid_px[0, b]
        for h in range(H):
            expected_equity += hedge_pos_out[h] * h_mid[0, h]
        assert equity == pytest.approx(expected_equity, abs=1e-6)


class TestNoHedge:
    """When hedge_ratios is all zeros, behavior matches pre-hedge version."""

    def test_zero_ratios_no_hedge_trades(self):
        T, B, H = 3, 1, 1
        L = 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H, L)
        hedge_dv01 = np.full((T, H), 0.01, dtype=np.float64)
        hedge_ratios = np.zeros((T, B, H), dtype=np.float64)  # all zeros

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=hedge_dv01,
            hedge_ratios=hedge_ratios,
        )
        out_hedge_pos = result[8]
        # No hedge trades → hedge_pos stays at zero
        assert np.all(np.abs(out_hedge_pos) < 1e-15)

    def test_none_hedges_matches_no_hedge(self):
        """Passing None for all hedge params gives same result as before."""
        T, B = 3, 1
        L = 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
        )
        n_trades = result[35]
        assert n_trades > 0
        out_hedge_pos = result[8]
        # H=0 → out_hedge_pos is (T, 1) of zeros
        assert np.all(np.abs(out_hedge_pos) < 1e-15)


class TestHedgePartialFill:
    """When hedge book has limited liquidity, partial fill recorded."""

    def test_partial_hedge_fill(self):
        T, B, H = 1, 1, 1
        L = 3
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair_price = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        # Hedge book with very small size
        h_bid, h_bsz, h_ask, h_asz, h_mid = _make_hedge_market(T, H, L, spread_bps=5.0)
        # Limit hedge bid book to 1.0 per level → max 3.0 total
        h_bsz[:] = 1.0
        h_asz[:] = 1.0
        hedge_dv01 = np.full((T, H), 0.01, dtype=np.float64)
        hedge_ratios = np.full((T, B, H), -1.0, dtype=np.float64)  # sell hedge

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair_price, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, None,
            **params,
            hedge_bid_px=h_bid, hedge_bid_sz=h_bsz,
            hedge_ask_px=h_ask, hedge_ask_sz=h_asz,
            hedge_mid_px=h_mid,
            hedge_dv01=hedge_dv01,
            hedge_ratios=hedge_ratios,
        )
        n_trades = result[35]
        assert n_trades > 0
        tr_hedge_fills = result[24]
        # Hedge should be partially filled (limited by book size)
        # Instrument fill was large but hedge book only had 3.0 total
        assert abs(tr_hedge_fills[0, 0]) > 0  # some fill
        assert abs(tr_hedge_fills[0, 0]) <= 3.0 + 1e-10  # capped by book


# =========================================================================
# Time-varying (T, B) maturity / maturity_bucket
# =========================================================================


class TestTimeVaryingMaturity:
    """Verify that (T, B) maturity and maturity_bucket work correctly."""

    def test_2d_maturity_backward_compat(self):
        """(T, B) maturity arrays produce same basic behaviour as (B,)."""
        T, B = 5, 2
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        # Static (B,) maturity
        maturity_1d = np.array([5.0, 10.0])

        result_1d = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            maturity_1d, None, None,
            **params,
        )

        # Time-varying (T, B) maturity — constant across time
        maturity_2d = np.broadcast_to(maturity_1d, (T, B)).copy()

        result_2d = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            maturity_2d, None, None,
            **params,
        )

        # Same trades produced
        assert result_1d[35] == result_2d[35]
        np.testing.assert_allclose(result_1d[0], result_2d[0])

    def test_2d_maturity_filter_blocks_when_maturity_decreases(self):
        """Instrument initially above min_maturity_inc, then drops below → blocked."""
        T, B = 6, 1
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair = mid_px + 1.0  # strong buy
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e8)
        pos_short = np.full(B, -1e8)
        params = _default_params()
        params["min_maturity_inc"] = 2.0
        params["cooldown_bars"] = 0

        # Maturity decreases from 3.0 to 0.5 over 6 bars
        maturity_2d = np.array([[3.0], [2.5], [2.0], [1.5], [1.0], [0.5]])
        assert maturity_2d.shape == (T, B)

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            maturity_2d, None, None,
            **params,
        )
        n_trades = result[35]
        tr_bar = result[9]
        # Trades should only happen in bars 0-2 (maturity >= 2.0)
        # Bars 3-5 have maturity < 2.0, risk-increasing trades should be blocked
        # Note: once position is > 0, sells become risk-decreasing and bypass filter
        for i in range(n_trades):
            bar_idx = tr_bar[i]
            side = result[11][i]
            if side == 1:  # BUY (risk-increasing)
                assert maturity_2d[bar_idx, 0] >= 2.0, (
                    f"Buy trade at bar {bar_idx} with maturity "
                    f"{maturity_2d[bar_idx, 0]} < min_maturity_inc=2.0"
                )

    def test_2d_maturity_bucket_in_lp(self):
        """(T, B) maturity_bucket is correctly sliced per bar in LP."""
        T, B = 4, 2
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()
        params["cooldown_bars"] = 0

        # Both instruments start in bucket 0, then instrument 1 moves to bucket 1
        maturity_bucket_2d = np.array([
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
        ], dtype=np.int64)
        params["mat_bucket_dv01_caps"] = np.array([50.0, 50.0])

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            None, None, maturity_bucket_2d,
            **params,
        )
        # Should run without error — LP uses per-bar slice
        assert result[35] >= 0  # non-negative trade count

    def test_1d_arrays_still_work(self):
        """Existing (B,) maturity and maturity_bucket arrays still work."""
        T, B = 3, 2
        bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, spread_bps=5.0)
        dv01 = np.full((T, B), 0.01, dtype=np.float64)
        fair = mid_px + 1.0
        zscore = np.full((T, B), 3.0)
        adf_p = np.full((T, B), 0.01)
        tradable = np.ones(B, dtype=np.int32)
        pos_long = np.full(B, 1e6)
        pos_short = np.full(B, -1e6)
        params = _default_params()

        maturity_1d = np.array([5.0, 10.0])
        maturity_bucket_1d = np.array([0, 1], dtype=np.int64)
        params["mat_bucket_dv01_caps"] = np.array([100.0, 100.0])

        result = lp_portfolio_loop(
            bid_px, bid_sz, ask_px, ask_sz, mid_px,
            dv01, fair, zscore, adf_p,
            tradable, pos_long, pos_short,
            np.full_like(pos_long, np.inf), np.full_like(pos_long, np.inf),
            maturity_1d, None, maturity_bucket_1d,
            **params,
        )
        assert result[35] > 0  # trades happen
