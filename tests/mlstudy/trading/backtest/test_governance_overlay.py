# tests/test_governance_overlay.py
#
# Tests for GovernanceOverlay (implementability/tradability logic):
# - liquidity-gated entry (skip mode)
# - stop-loss -> cooldown blocks re-entry
# - max holding -> cooldown blocks re-entry
#
# These tests do NOT require engine integration yet.
# They validate the overlay state machine in isolation.

from __future__ import annotations

import numpy as np

from mlstudy.trading.backtest.core.types import (
    GovernanceConfig,
    GovernanceState,
    StopMode,
    ExitReason,
    EntryBlockReason,
)
from mlstudy.trading.backtest.lifecycle.overlay import GovernanceOverlay


def _mk_book_sizes(T: int, N: int = 3, L: int = 2, *, base: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create constant bid/ask sizes (T,N,L).
    """
    bid_sz = np.full((T, N, L), base, dtype=np.float64)
    ask_sz = np.full((T, N, L), base, dtype=np.float64)
    return bid_sz, ask_sz


def _mk_desired_targets_switch(
    T: int,
    *,
    q_front: float,
    q_back: float,
    enter_t: int,
    exit_t: int,
    N: int = 3,
) -> np.ndarray:
    """
    Simple desired target path:
      - flat until enter_t
      - hold a 2-leg switch [front, belly, back] from enter_t .. exit_t-1
      - flat from exit_t onward
    """
    desired = np.zeros((T, N), dtype=np.float64)
    desired[enter_t:exit_t, 0] = q_front
    desired[enter_t:exit_t, 1] = 0.0
    desired[enter_t:exit_t, 2] = q_back
    return desired


def test_overlay_skip_if_entry_not_fillable_no_stop_no_cooldown():
    """
    If entry is blocked by liquidity (skip mode), we remain flat.
    Later equity moves should not trigger stops/cooldowns because we never entered.
    """
    T = 10
    N, L = 3, 2
    bid_sz, ask_sz = _mk_book_sizes(T, N, L, base=1.0)  # tiny liquidity

    cfg = GovernanceConfig(
        enabled=True,
        mode="IMPLEMENTABILITY_OWNED",
        entry_liquidity_gate=True,
        exit_liquidity_gate=False,
        stop_mode=StopMode.PER_TRADE_PNL,
        stop_value=10.0,
        cooldown_after_stop=20,
        gate_size_haircut=1.0,
        gate_max_levels_to_cross=2,
    )
    ov = GovernanceOverlay(cfg=cfg, state=GovernanceState())
    ov.reset(initial_equity=100.0)

    # Desired wants to enter at t=1 and hold, but liquidity is insufficient
    desired = _mk_desired_targets_switch(T, q_front=50.0, q_back=-50.0, enter_t=1, exit_t=9, N=N)

    pos = np.zeros(N, dtype=np.float64)
    equity = 100.0

    entered_any = False
    for t in range(T):
        # equity collapses at t=5, but we never entered
        if t == 5:
            equity = 50.0

        dec = ov.step(
            t=t,
            desired_target_leg=desired[t],
            current_pos_leg=pos,
            bid_sz_t=bid_sz[t],
            ask_sz_t=ask_sz[t],
            equity_t=equity,
            execution_size_haircut=1.0,
            execution_max_levels_to_cross=2,
        )

        # Overlay should keep us flat because entry isn't fillable
        assert np.allclose(dec.target_leg, 0.0)

        # If signal asked to enter, we should see liquidity block reason
        if t >= 1 and not np.allclose(desired[t], 0.0):
            assert dec.entry_block_reason == int(EntryBlockReason.LIQUIDITY)

        # No stop-loss exit should occur because never in position
        assert dec.exit_reason == int(ExitReason.NONE)
        assert dec.cooldown_remaining == 0

        # Apply target to pos (in a real engine fills decide; here we mirror target for overlay-only test)
        pos = dec.target_leg.copy()

        entered_any = entered_any or (not np.allclose(pos, 0.0))

    assert entered_any is False


def test_overlay_stop_loss_triggers_cooldown_and_blocks_entry_until_expiry():
    """
    - t1 entry feasible => overlay accepts entry, sets entry_equity
    - t5 equity drawdown crosses per-trade stop => overlay forces flat and sets cooldown
    - during cooldown, desired may want to enter, but overlay blocks and stays flat
    - at cooldown expiry, entry can happen again if feasible
    """
    T = 40
    N, L = 3, 2
    bid_sz, ask_sz = _mk_book_sizes(T, N, L, base=1_000.0)  # ample liquidity

    cfg = GovernanceConfig(
        enabled=True,
        mode="IMPLEMENTABILITY_OWNED",
        entry_liquidity_gate=True,
        exit_liquidity_gate=False,
        stop_mode=StopMode.PER_TRADE_PNL,
        stop_value=10.0,            # stop if loss >= 10 currency
        cooldown_after_stop=20,     # block entry for 20 bars after stop
        gate_size_haircut=1.0,
        gate_max_levels_to_cross=2,
    )
    ov = GovernanceOverlay(cfg=cfg, state=GovernanceState())
    ov.reset(initial_equity=100.0)

    # Strategy "intends": enter at t=1, exit at t=30 (but we will stop out at t=5)
    desired = _mk_desired_targets_switch(T, q_front=50.0, q_back=-50.0, enter_t=1, exit_t=30, N=N)

    pos = np.zeros(N, dtype=np.float64)
    equity = 100.0

    stopped_at = None
    reentered_at = None

    for t in range(T):
        # Trigger stop at t=5 by dropping equity
        if t == 5:
            equity = 89.0  # loss = 11 >= 10 => stop

        dec = ov.step(
            t=t,
            desired_target_leg=desired[t],
            current_pos_leg=pos,
            bid_sz_t=bid_sz[t],
            ask_sz_t=ask_sz[t],
            equity_t=equity,
            execution_size_haircut=1.0,
            execution_max_levels_to_cross=2,
        )

        # Apply overlay target to pos (overlay-only test)
        pos = dec.target_leg.copy()

        # Entry accepted at t=1 (since feasible)
        if t == 1:
            assert not np.allclose(pos, 0.0)
            assert dec.entry_block_reason == int(EntryBlockReason.NONE)

        # Stop triggers at t=5 and forces flat + cooldown
        if t == 5:
            assert np.allclose(pos, 0.0)
            assert dec.exit_reason == int(ExitReason.STOP_LOSS)
            assert dec.cooldown_remaining == cfg.cooldown_after_stop
            stopped_at = t

        # During cooldown: forced flat and entry blocked
        if stopped_at is not None and 6 <= t <= 5 + cfg.cooldown_after_stop:
            assert np.allclose(pos, 0.0)
            assert dec.entry_block_reason in (int(EntryBlockReason.COOLDOWN), int(EntryBlockReason.NONE))

        # First possible re-entry bar is t = 5 + cooldown_after_stop + 1
        # (because we decrement cooldown at step start)
        if reentered_at is None and t >= 5 + cfg.cooldown_after_stop + 1:
            # If desired still wants non-zero and liquidity is fine, we should be able to re-enter
            if not np.allclose(desired[t], 0.0):
                assert not np.allclose(pos, 0.0)
                reentered_at = t

    assert stopped_at == 5
    assert reentered_at is not None
    assert reentered_at >= 5 + cfg.cooldown_after_stop + 1


def test_overlay_max_holding_forces_exit_and_time_cooldown():
    """
    - enter feasible at t=1
    - max holding = 5 bars => at t=6 (t-entry_bar >= 5) overlay forces exit
    - apply cooldown_after_time
    """
    T = 30
    N, L = 3, 2
    bid_sz, ask_sz = _mk_book_sizes(T, N, L, base=1_000.0)

    cfg = GovernanceConfig(
        enabled=True,
        mode="IMPLEMENTABILITY_OWNED",
        entry_liquidity_gate=True,
        stop_mode=StopMode.NONE,
        max_holding_bars=5,
        cooldown_after_time=7,
        gate_size_haircut=1.0,
        gate_max_levels_to_cross=2,
    )
    ov = GovernanceOverlay(cfg=cfg, state=GovernanceState())
    ov.reset(initial_equity=100.0)

    desired = _mk_desired_targets_switch(T, q_front=10.0, q_back=-10.0, enter_t=1, exit_t=25, N=N)

    pos = np.zeros(N, dtype=np.float64)
    equity = 100.0

    time_exit_at = None

    for t in range(T):
        dec = ov.step(
            t=t,
            desired_target_leg=desired[t],
            current_pos_leg=pos,
            bid_sz_t=bid_sz[t],
            ask_sz_t=ask_sz[t],
            equity_t=equity,
            execution_size_haircut=1.0,
            execution_max_levels_to_cross=2,
        )
        pos = dec.target_leg.copy()

        if t == 1:
            assert not np.allclose(pos, 0.0)

        # entry_bar is 1, so max holding triggers when t - 1 >= 5 => t >= 6
        if t == 6:
            assert np.allclose(pos, 0.0)
            assert dec.exit_reason == int(ExitReason.MAX_HOLDING)
            assert dec.cooldown_remaining == cfg.cooldown_after_time
            time_exit_at = t

        # During time cooldown: stay flat
        if time_exit_at is not None and 7 <= t <= time_exit_at + cfg.cooldown_after_time:
            assert np.allclose(pos, 0.0)

    assert time_exit_at == 6
