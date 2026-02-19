"""
End-to-end tests for the mean-reversion backtester.

Covers:
  1. Successful LONG entry  (ActionCode.ENTRY_OK)
  2. Successful take-profit  (ActionCode.EXIT_TP_OK)
  3. Forced stop-loss exit   (ActionCode.EXIT_SL_OK)
  4. Failed entry – no liquidity (ActionCode.ENTRY_FAILED_NO_LIQUIDITY)
  5. Failed entry – too wide  (ActionCode.ENTRY_FAILED_TOO_WIDE)
  6. Quarantine / cooldown    (ActionCode.ENTRY_FAILED_IN_COOLDOWN)
  7. Python vs JIT output parity (if Numba is available)

Synthetic data
--------------
- 3 instruments (2-year, 5-year, 10-year proxy).
- Butterfly hedge ratios: [-0.5, 1.0, -0.5], ref = instrument 1.
- Yields follow a scripted path so we can predict exactly which bars
  trigger entry, TP, and SL.
- L2 book has 2 levels.  Sizes are controllable to test NO_LIQUIDITY.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.single_backtest.engine import run_backtest
from mlstudy.trading.backtest.mean_reversion.single_backtest.state import (
    ActionCode, State, TradeType,
)
from mlstudy.trading.backtest.mean_reversion.single_backtest.loop import HAS_NUMBA


# =========================================================================
# Helpers
# =========================================================================

def _cfg(**overrides):
    """Build an MRBacktestConfig with test defaults."""
    defaults = dict(
        target_notional_ref=100.0,
        ref_leg_idx=0,
        entry_z_threshold=2.0,
        take_profit_zscore_soft_threshold=0.5,
        take_profit_yield_change_soft_threshold=1.0,
        take_profit_yield_change_hard_threshold=3.0,
        stop_loss_yield_change_hard_threshold=5.0,
        max_holding_bars=0,
        expected_yield_pnl_bps_multiplier=1.0,
        entry_cost_premium_yield_bps=0.0,
        tp_cost_premium_yield_bps=0.0,
        sl_cost_premium_yield_bps=0.0,
        tp_quarantine_bars=0,
        sl_quarantine_bars=0,
        time_quarantine_bars=0,
        max_levels_to_cross=5,
        size_haircut=1.0,
        validate_scope="REF_ONLY",
        initial_capital=0.0,
        use_jit=False,
    )
    defaults.update(overrides)
    return MRBacktestConfig(**defaults)


def _make_book(mid_px, half_spread, level2_offset, base_sizes):
    """Build 2-level L2 book from mid prices.

    Parameters
    ----------
    mid_px : (T, N)
    half_spread : (N,)     half the bid-ask spread per instrument.
    level2_offset : (N,)   extra offset for level-2.
    base_sizes : (T, N)    displayed size per level.

    Returns
    -------
    bid_px, bid_sz, ask_px, ask_sz : (T, N, 2)
    """
    T, N = mid_px.shape
    bid0 = mid_px - half_spread[None, :]
    ask0 = mid_px + half_spread[None, :]
    bid1 = bid0 - level2_offset[None, :]
    ask1 = ask0 + level2_offset[None, :]

    bid_px = np.stack([bid0, bid1], axis=2)
    ask_px = np.stack([ask0, ask1], axis=2)
    bid_sz = np.stack([base_sizes, 0.5 * base_sizes], axis=2)
    ask_sz = np.stack([base_sizes, 0.5 * base_sizes], axis=2)
    return bid_px, bid_sz, ask_px, ask_sz


def _make_scripted_inputs(
    T: int = 60,
    *,
    entry_bar: int = 5,
    tp_bar: int = 20,
    sl_bar: int = 40,
    base_book_size: float = 1000.0,
    zero_liquidity_bar: int = -1,
):
    """Create a fully scripted dataset for deterministic testing.

    The z-score path is crafted so that:
      - Bars 0..entry_bar-1: z ~ 0 (no signal)
      - Bar entry_bar: z jumps > entry_threshold (triggers LONG entry)
      - Bars entry_bar+1..tp_bar-1: z stays high (in position)
      - Bar tp_bar: yield reverts enough to trigger TP
      - Bars tp_bar+1..sl_bar-1: z rises again, enters LONG again
      - Bar sl_bar: yield moves adversely -> SL

    Returns dict with all arrays + expected events.
    """
    N = 3
    L = 2
    ref_idx = 1
    hedge_ratios = np.tile(
        np.array([-0.5, 1.0, -0.5], dtype=np.float64), (T, 1)
    )

    # DV01: constant
    dv01_vals = np.array([0.02, 0.045, 0.08], dtype=np.float64)
    dv01 = np.tile(dv01_vals, (T, 1))

    # Mid prices: around par
    base_px = np.array([99.0, 98.0, 97.0], dtype=np.float64)
    mid_px = np.tile(base_px, (T, 1))
    # Add small drift for realism
    rng = np.random.default_rng(42)
    mid_px += np.cumsum(rng.normal(0, 0.001, (T, N)), axis=0)

    # L2 book
    half_spread = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    level2_off = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    base_sizes = np.full((T, N), base_book_size, dtype=np.float64)
    if zero_liquidity_bar >= 0:
        base_sizes[zero_liquidity_bar, :] = 0.0  # no liquidity on that bar

    bid_px, bid_sz, ask_px, ask_sz = _make_book(
        mid_px, half_spread, level2_off, base_sizes)

    # --- signals (scripted) ---
    zscore = np.zeros(T, dtype=np.float64)
    package_yield_bps = np.zeros(T, dtype=np.float64)
    expected_yield_pnl_bps = np.full(T, 5.0, dtype=np.float64)  # generous budget

    # Phase 1: flat (z ~ 0)
    zscore[:entry_bar] = 0.0

    # Phase 2: entry trigger (z > threshold)
    zscore[entry_bar] = 3.0
    zscore[entry_bar + 1: tp_bar] = 2.5  # stay above threshold but don't TP yet

    # package yield starts at some level at entry
    package_yield_bps[entry_bar] = 100.0
    package_yield_bps[entry_bar + 1: tp_bar] = 100.0

    # Phase 3: TP trigger
    # For LONG: adj_z = -zscore, so we need zscore to become negative (reverted)
    # and adj_yield_delta = -(yield[t] - yield[entry]) to be positive -> yield must decrease
    # TP hard: adj_yield_delta > tp_yield_hard
    zscore[tp_bar] = -1.0   # reverted
    package_yield_bps[tp_bar] = 94.0  # yield dropped 6 bps from entry (100)
    # adj_yield_delta = -1 * (94 - 100) = 6.0 > tp_yield_hard (3.0) -> TP

    # Phase 4: cooldown + re-enter
    re_entry_bar = tp_bar + 4  # after cooldown (tp_quarantine=2)
    zscore[tp_bar + 1: re_entry_bar] = 0.0  # no signal during cooldown
    zscore[re_entry_bar] = 3.5  # re-enter LONG
    package_yield_bps[re_entry_bar] = 100.0

    # Phase 5: SL trigger
    zscore[re_entry_bar + 1: sl_bar] = 2.0
    package_yield_bps[re_entry_bar + 1: sl_bar] = 100.0

    # SL: adj_yield_delta = -(yield[t] - entry_yield) < -sl_yield_hard
    # for LONG: yield INCREASED (bad) -> adj_yield_delta negative
    # yield[sl_bar] = 106 -> adj_yield_delta = -(106-100) = -6 < -5 -> SL!
    zscore[sl_bar] = 2.0
    package_yield_bps[sl_bar] = 106.0

    # After SL: fill remaining bars
    if sl_bar + 1 < T:
        zscore[sl_bar + 1:] = 0.0
        package_yield_bps[sl_bar + 1:] = 100.0

    return {
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
        "ref_idx": ref_idx,
        "entry_bar": entry_bar,
        "tp_bar": tp_bar,
        "re_entry_bar": re_entry_bar,
        "sl_bar": sl_bar,
        "T": T,
        "N": N,
    }


# =========================================================================
# Tests
# =========================================================================

class TestMRBacktestEndToEnd:
    """Full lifecycle test: entry -> TP -> re-entry -> SL."""

    def _run(self, **kw):
        d = _make_scripted_inputs(**kw)
        cfg = _cfg(
            ref_leg_idx=d["ref_idx"],
            tp_quarantine_bars=2,
            sl_quarantine_bars=3,
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
        )
        res = run_backtest(
            bid_px=d["bid_px"],
            bid_sz=d["bid_sz"],
            ask_px=d["ask_px"],
            ask_sz=d["ask_sz"],
            mid_px=d["mid_px"],
            dv01=d["dv01"],
            zscore=d["zscore"],
            expected_yield_pnl_bps=d["expected_yield_pnl_bps"],
            package_yield_bps=d["package_yield_bps"],
            hedge_ratios=d["hedge_ratios"],
            cfg=cfg,
        )
        return res, d

    def test_shapes(self):
        res, d = self._run()
        T, N = d["T"], d["N"]
        assert res.positions.shape == (T, N)
        assert res.cash.shape == (T,)
        assert res.equity.shape == (T,)
        assert res.pnl.shape == (T,)
        assert res.codes.shape == (T,)
        assert res.state.shape == (T,)
        assert res.holding.shape == (T,)

    def test_entry_ok(self):
        res, d = self._run()
        eb = d["entry_bar"]
        assert res.codes[eb] == ActionCode.ENTRY_OK
        assert res.state[eb] == State.STATE_LONG
        # Position should be nonzero after entry
        assert np.any(np.abs(res.positions[eb]) > 1e-10)

    def test_tp_ok(self):
        res, d = self._run()
        tb = d["tp_bar"]
        assert res.codes[tb] == ActionCode.EXIT_TP_OK
        # tp_quarantine_bars=2 > 0 → enters TP_COOLDOWN state
        assert res.state[tb] == State.STATE_TP_COOLDOWN
        # Position should be flat after TP
        assert np.all(np.abs(res.positions[tb]) < 1e-10)

    def test_sl_forced(self):
        res, d = self._run()
        sb = d["sl_bar"]
        assert res.codes[sb] == ActionCode.EXIT_SL_OK
        # sl_quarantine_bars=3 > 0 → enters SL_COOLDOWN state
        assert res.state[sb] == State.STATE_SL_COOLDOWN

    def test_cooldown_after_tp(self):
        """After TP at bar tp_bar, next 2 bars should be in cooldown."""
        res, d = self._run()
        tb = d["tp_bar"]
        # tp_quarantine_bars=2 means bars tp_bar+1 and tp_bar+2 are cooldown
        # Bar tp_bar+1: cooldown starts (value=2, decremented to 1)
        # Bar tp_bar+2: cooldown continues (value=1, decremented to 0)
        # Bar tp_bar+3: cooldown expired, can enter again

        # Check that entry signals during cooldown produce ActionCode.ENTRY_FAILED_IN_COOLDOWN
        # Our scripted data has zscore=0 during cooldown, so we expect NO_ACTION
        # rather than IN_COOLDOWN (no signal to trigger entry).
        # Verify state is TP_COOLDOWN during cooldown:
        assert res.state[tb + 1] == State.STATE_TP_COOLDOWN
        assert res.state[tb + 2] == State.STATE_TP_COOLDOWN

    def test_cooldown_after_sl(self):
        """After SL, sl_quarantine_bars=3 bars should block entry."""
        res, d = self._run()
        sb = d["sl_bar"]
        for offset in range(1, 4):
            if sb + offset < d["T"]:
                assert res.state[sb + offset] == State.STATE_SL_COOLDOWN

    def test_re_entry_after_cooldown(self):
        """After TP cooldown, re-entry should happen."""
        res, d = self._run()
        reb = d["re_entry_bar"]
        assert res.codes[reb] == ActionCode.ENTRY_OK
        assert res.state[reb] == State.STATE_LONG

    def test_no_signal_before_entry(self):
        """Before the first entry, bars with z=0 should be ActionCode.NO_ACTION_NO_SIGNAL."""
        res, d = self._run()
        for t in range(d["entry_bar"]):
            assert res.codes[t] == ActionCode.NO_ACTION_NO_SIGNAL

    def test_equity_finite(self):
        res, _ = self._run()
        assert np.all(np.isfinite(res.equity))

    def test_pnl_sums_to_equity_change(self):
        """Cumulative PnL should equal equity[T-1] - initial_capital."""
        res, _ = self._run()
        assert abs(float(np.sum(res.pnl)) - float(res.equity[-1])) < 1e-8

    def test_trade_records(self):
        """Check trade record arrays are populated correctly."""
        res, d = self._run()
        # Should have at least: entry, TP exit, re-entry, SL exit = 4 trades
        assert res.n_trades >= 4

        # First trade should be entry
        assert res.tr_type[0] == TradeType.TRADE_ENTRY
        assert res.tr_code[0] == ActionCode.ENTRY_OK
        assert res.tr_bar[0] == d["entry_bar"]
        assert res.tr_side[0] == 1  # LONG

        # Second trade should be TP exit
        assert res.tr_type[1] == TradeType.TRADE_EXIT_TP
        assert res.tr_code[1] == ActionCode.EXIT_TP_OK
        assert res.tr_bar[1] == d["tp_bar"]

    def test_basket_sizing_dv01_consistent(self):
        """Leg sizes should maintain DV01 hedge ratios."""
        res, d = self._run()
        if res.n_trades == 0:
            pytest.skip("No trades to verify")

        entry_idx = 0
        sizes = res.tr_sizes[entry_idx]
        t = int(res.tr_bar[entry_idx])
        dv01_at_t = d["dv01"][t]
        hr = d["hedge_ratios"][t]

        # DV01 contributions
        dv01_contrib = sizes * dv01_at_t
        # Should be proportional to hedge_ratios
        ref = d["ref_idx"]
        scale = dv01_contrib[ref] / hr[ref]
        for i in range(d["N"]):
            if abs(hr[i]) > 1e-10:
                expected = hr[i] * scale
                assert abs(dv01_contrib[i] - expected) < 1e-6, (
                    f"Leg {i}: dv01_contrib={dv01_contrib[i]:.6f}, "
                    f"expected={expected:.6f}"
                )


class TestMRBacktestNoLiquidity:
    """Test ActionCode.ENTRY_FAILED_NO_LIQUIDITY when book has zero depth."""

    def test_no_liquidity_blocks_entry(self):
        d = _make_scripted_inputs(zero_liquidity_bar=5)
        cfg = _cfg(
            ref_leg_idx=d["ref_idx"],
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
        )
        res = run_backtest(
            bid_px=d["bid_px"],
            bid_sz=d["bid_sz"],
            ask_px=d["ask_px"],
            ask_sz=d["ask_sz"],
            mid_px=d["mid_px"],
            dv01=d["dv01"],
            zscore=d["zscore"],
            expected_yield_pnl_bps=d["expected_yield_pnl_bps"],
            package_yield_bps=d["package_yield_bps"],
            hedge_ratios=d["hedge_ratios"],
            cfg=cfg,
        )
        # Bar 5 should have entry attempt blocked by liquidity
        # (zero book sizes -> walk returns 0 filled)
        assert res.codes[5] == ActionCode.ENTRY_FAILED_NO_LIQUIDITY
        assert res.state[5] == State.STATE_FLAT


class TestMRBacktestTooWide:
    """Test ActionCode.ENTRY_FAILED_TOO_WIDE when execution cost exceeds budget."""

    def test_too_wide_blocks_entry(self):
        d = _make_scripted_inputs()
        cfg = _cfg(
            ref_leg_idx=d["ref_idx"],
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
            # Tiny expected PnL + large cost premium -> negative budget
            expected_yield_pnl_bps_multiplier=0.01,
            entry_cost_premium_yield_bps=100.0,
        )
        res = run_backtest(
            bid_px=d["bid_px"],
            bid_sz=d["bid_sz"],
            ask_px=d["ask_px"],
            ask_sz=d["ask_sz"],
            mid_px=d["mid_px"],
            dv01=d["dv01"],
            zscore=d["zscore"],
            expected_yield_pnl_bps=d["expected_yield_pnl_bps"],
            package_yield_bps=d["package_yield_bps"],
            hedge_ratios=d["hedge_ratios"],
            cfg=cfg,
        )
        # Bar 5 has z > threshold but cost budget is negative -> TOO_WIDE
        assert res.codes[d["entry_bar"]] == ActionCode.ENTRY_FAILED_TOO_WIDE
        assert res.state[d["entry_bar"]] == State.STATE_FLAT


class TestMRBacktestCooldownWithSignal:
    """Test ActionCode.ENTRY_FAILED_IN_COOLDOWN when signal is present during cooldown."""

    def test_in_cooldown_code(self):
        """Craft data so z > threshold during cooldown bars after SL."""
        T = 30
        N = 3
        ref_idx = 1
        hedge_ratios = np.tile(
            np.array([-0.5, 1.0, -0.5], dtype=np.float64), (T, 1)
        )

        dv01 = np.tile(np.array([0.02, 0.045, 0.08]), (T, 1))
        mid_px = np.tile(np.array([99.0, 98.0, 97.0]), (T, 1))

        half_sp = np.array([0.01, 0.01, 0.01])
        l2_off = np.array([0.01, 0.01, 0.01])
        base_sz = np.full((T, N), 1000.0)
        bid_px, bid_sz, ask_px, ask_sz = _make_book(mid_px, half_sp, l2_off, base_sz)

        zscore = np.zeros(T)
        package_yield_bps = np.full(T, 100.0)
        expected_yield_pnl_bps = np.full(T, 10.0)

        # Bar 2: enter LONG
        zscore[2] = 3.0
        # Bar 5: SL (yield goes up 6 bps)
        zscore[3:6] = 2.0
        package_yield_bps[5] = 106.0
        # Bars 6-8: cooldown (sl_quarantine=3) with z > threshold
        zscore[6] = 4.0
        zscore[7] = 4.0
        zscore[8] = 4.0
        # Bar 9: cooldown expired, can enter
        zscore[9] = 3.0

        cfg = _cfg(
            ref_leg_idx=ref_idx,
            sl_quarantine_bars=3,
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
        )

        res = run_backtest(
            bid_px=bid_px, bid_sz=bid_sz,
            ask_px=ask_px, ask_sz=ask_sz,
            mid_px=mid_px, dv01=dv01,
            zscore=zscore,
            expected_yield_pnl_bps=expected_yield_pnl_bps,
            package_yield_bps=package_yield_bps,
            hedge_ratios=hedge_ratios,
            cfg=cfg,
        )

        assert res.codes[2] == ActionCode.ENTRY_OK
        assert res.codes[5] == ActionCode.EXIT_SL_OK
        # Bars 6, 7, 8: cooldown with signal present -> ActionCode.ENTRY_FAILED_IN_COOLDOWN
        assert res.codes[6] == ActionCode.ENTRY_FAILED_IN_COOLDOWN
        assert res.codes[7] == ActionCode.ENTRY_FAILED_IN_COOLDOWN
        assert res.codes[8] == ActionCode.ENTRY_FAILED_IN_COOLDOWN
        # Bar 9: cooldown expired -> can enter
        assert res.codes[9] == ActionCode.ENTRY_OK


class TestMRBacktestMaxHolding:
    """Test max_holding_bars forced exit (EXIT_TIME_FORCED)."""

    def test_max_holding_forces_exit(self):
        T = 30
        N = 3
        ref_idx = 1
        hedge_ratios = np.tile(
            np.array([-0.5, 1.0, -0.5], dtype=np.float64), (T, 1)
        )
        dv01 = np.tile(np.array([0.02, 0.045, 0.08]), (T, 1))
        mid_px = np.tile(np.array([99.0, 98.0, 97.0]), (T, 1))

        half_sp = np.array([0.01, 0.01, 0.01])
        l2_off = np.array([0.01, 0.01, 0.01])
        base_sz = np.full((T, N), 1000.0)
        bid_px, bid_sz, ask_px, ask_sz = _make_book(mid_px, half_sp, l2_off, base_sz)

        zscore = np.zeros(T)
        package_yield_bps = np.full(T, 100.0)
        expected_yield_pnl_bps = np.full(T, 10.0)

        # Bar 2: enter LONG
        zscore[2] = 3.0
        # Stay in position (z stays high, no TP/SL trigger)
        zscore[3:] = 2.0


        cfg = _cfg(
            ref_leg_idx=ref_idx,
            stop_loss_yield_change_hard_threshold=50.0,  # very high, won't trigger
            max_holding_bars=5,
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
        )

        res = run_backtest(
            bid_px=bid_px, bid_sz=bid_sz,
            ask_px=ask_px, ask_sz=ask_sz,
            mid_px=mid_px, dv01=dv01,
            zscore=zscore,
            expected_yield_pnl_bps=expected_yield_pnl_bps,
            package_yield_bps=package_yield_bps,
            hedge_ratios=hedge_ratios,
            cfg=cfg,
        )

        assert res.codes[2] == ActionCode.ENTRY_OK
        # Holding starts at 0 on entry bar, increments each bar.
        # max_holding_bars=5: exit when holding >= 5
        # Bar 2: entry, holding=0
        # Bar 3: holding=1, Bar 4: holding=2, ..., Bar 7: holding=5 -> exit
        assert res.codes[7] == ActionCode.EXIT_TIME_OK
        assert res.state[7] == State.STATE_FLAT


class TestMRBacktestInactiveLegs:
    """Test that a zero-hedge-ratio leg does not block entry."""

    def test_zero_hedge_ratio_leg_entry_ok(self):
        """Entry should succeed even when one leg has hedge_ratio=0 and bad dv01."""
        T = 30
        N = 3
        ref_idx = 1

        # Hedge ratios: leg 2 is inactive (0.0)
        hedge_ratios = np.tile(
            np.array([-1.0, 1.0, 0.0], dtype=np.float64), (T, 1)
        )

        # DV01: leg 2 has zero dv01 (would fail if checked)
        dv01 = np.tile(np.array([0.02, 0.045, 0.0]), (T, 1))
        mid_px = np.tile(np.array([99.0, 98.0, 0.0]), (T, 1))

        half_sp = np.array([0.01, 0.01, 0.01])
        l2_off = np.array([0.01, 0.01, 0.01])
        base_sz = np.full((T, N), 1000.0)
        bid_px, bid_sz, ask_px, ask_sz = _make_book(mid_px, half_sp, l2_off, base_sz)

        zscore = np.zeros(T)
        package_yield_bps = np.full(T, 100.0)
        expected_yield_pnl_bps = np.full(T, 10.0)

        # Bar 2: enter LONG
        zscore[2] = 3.0
        zscore[3:] = 2.0

        cfg = _cfg(
            ref_leg_idx=ref_idx,
            stop_loss_yield_change_hard_threshold=50.0,
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
        )

        res = run_backtest(
            bid_px=bid_px, bid_sz=bid_sz,
            ask_px=ask_px, ask_sz=ask_sz,
            mid_px=mid_px, dv01=dv01,
            zscore=zscore,
            expected_yield_pnl_bps=expected_yield_pnl_bps,
            package_yield_bps=package_yield_bps,
            hedge_ratios=hedge_ratios,
            cfg=cfg,
        )

        # Entry should succeed despite leg 2 having zero dv01
        assert res.codes[2] == ActionCode.ENTRY_OK
        assert res.state[2] == State.STATE_LONG
        # Leg 2 should have zero position
        assert abs(res.positions[2, 2]) < 1e-15
        # Active legs should have nonzero positions
        assert abs(res.positions[2, 0]) > 1e-10
        assert abs(res.positions[2, 1]) > 1e-10


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
class TestMRBacktestJITParity:
    """Verify Python and JIT loops produce identical output."""

    def test_py_vs_jit_match(self):
        d = _make_scripted_inputs()
        inputs = dict(
            bid_px=d["bid_px"],
            bid_sz=d["bid_sz"],
            ask_px=d["ask_px"],
            ask_sz=d["ask_sz"],
            mid_px=d["mid_px"],
            dv01=d["dv01"],
            zscore=d["zscore"],
            expected_yield_pnl_bps=d["expected_yield_pnl_bps"],
            package_yield_bps=d["package_yield_bps"],
            hedge_ratios=d["hedge_ratios"],
        )
        shared = dict(
            ref_leg_idx=d["ref_idx"],
            tp_quarantine_bars=2,
            sl_quarantine_bars=3,
            max_levels_to_cross=2,
            validate_scope="ALL_LEGS",
        )

        res_py = run_backtest(**inputs, cfg=_cfg(**shared, use_jit=False))
        res_jit = run_backtest(**inputs, cfg=_cfg(**shared, use_jit=True))

        np.testing.assert_array_equal(res_py.codes, res_jit.codes)
        np.testing.assert_array_equal(res_py.state, res_jit.state)
        np.testing.assert_allclose(res_py.positions, res_jit.positions, atol=1e-10)
        np.testing.assert_allclose(res_py.cash, res_jit.cash, atol=1e-10)
        np.testing.assert_allclose(res_py.equity, res_jit.equity, atol=1e-10)
        np.testing.assert_allclose(res_py.pnl, res_jit.pnl, atol=1e-10)
        assert res_py.n_trades == res_jit.n_trades
        np.testing.assert_array_equal(res_py.tr_bar, res_jit.tr_bar)
        np.testing.assert_array_equal(res_py.tr_code, res_jit.tr_code)
