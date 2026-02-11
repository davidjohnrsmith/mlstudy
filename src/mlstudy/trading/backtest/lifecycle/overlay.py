# backtest/lifecycle/overlay.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core.types import (
    GovernanceConfig,
    GovernanceState,
    GovernanceDecision,
    ExitReason,
    EntryBlockReason,
    StopMode,
    EntryTrigger,
)


def _dec_nonneg(x: int) -> int:
    return x - 1 if x > 0 else 0


def _available_qty_from_l2(
    *,
    side: int,                 # +1 buy (hit ask), -1 sell (hit bid)
    bid_sz_levels: np.ndarray,  # (L,)
    ask_sz_levels: np.ndarray,  # (L,)
    max_levels: int,
    haircut: float,
) -> float:
    L = int(bid_sz_levels.shape[0])
    k = min(L, int(max_levels))
    if side > 0:
        return float(np.sum(ask_sz_levels[:k]) * haircut)
    return float(np.sum(bid_sz_levels[:k]) * haircut)


def _is_basket_feasible(
    *,
    trade_leg: np.ndarray,      # (N,) signed trade sizes
    bid_sz_t: np.ndarray,       # (N,L)
    ask_sz_t: np.ndarray,       # (N,L)
    max_levels: int,
    haircut: float,
    legs_to_check: Optional[np.ndarray] = None,
) -> bool:
    """
    Atomic feasibility check: every non-zero trade leg must be feasible.
    """
    if legs_to_check is None:
        idxs = np.where(np.abs(trade_leg) > 0)[0]
    else:
        idxs = legs_to_check

    for i in idxs:
        q = float(trade_leg[i])
        if q == 0.0:
            continue
        side = +1 if q > 0 else -1
        avail = _available_qty_from_l2(
            side=side,
            bid_sz_levels=bid_sz_t[i],
            ask_sz_levels=ask_sz_t[i],
            max_levels=max_levels,
            haircut=haircut,
        )
        if abs(q) > avail + 1e-9:
            return False
    return True


@dataclass
class GovernanceOverlay:
    """
    Governance overlay for a *single* basket (one portfolio) controlling leg targets.

    You call .step(...) each bar with:
    - desired_target_leg: what the strategy "wants" (signal-owned)
    - current_pos_leg: what you currently hold (legs)
    - L2 sizes at time t: bid_sz_t, ask_sz_t
    - equity_t: current equity (for stop logic)

    It outputs executable target_leg and records reasons.
    """

    cfg: GovernanceConfig
    state: GovernanceState

    def reset(self, *, initial_equity: float = 0.0) -> None:
        self.state.pos_side = 0
        self.state.entry_bar = -1
        self.state.entry_equity = float(initial_equity)
        self.state.cooldown_remaining = 0
        self.state.peak_equity = float(initial_equity)

    def _apply_stop_and_time_exits(
        self,
        *,
        t: int,
        current_pos_leg: np.ndarray,
        equity_t: float,
    ) -> Tuple[bool, int]:
        """
        Returns (should_flatten, exit_reason_int).
        """
        # update peak equity for drawdown stop
        self.state.peak_equity = float(max(self.state.peak_equity, equity_t))

        in_pos = bool(np.any(np.abs(current_pos_leg) > 0))
        if not in_pos:
            return False, int(ExitReason.NONE)

        # Max holding
        if self.cfg.max_holding_bars and self.cfg.max_holding_bars > 0 and self.state.entry_bar >= 0:
            if (t - self.state.entry_bar) >= int(self.cfg.max_holding_bars):
                return True, int(ExitReason.MAX_HOLDING)

        # Stops
        if self.cfg.stop_mode == StopMode.PER_TRADE_PNL:
            # stop_value is positive currency loss threshold
            loss = float(self.state.entry_equity - equity_t)
            if self.cfg.stop_value > 0 and loss >= float(self.cfg.stop_value):
                return True, int(ExitReason.STOP_LOSS)

        if self.cfg.stop_mode == StopMode.EQUITY_DRAWDOWN:
            peak = max(self.state.peak_equity, 1e-12)
            dd = (peak - float(equity_t)) / peak
            if self.cfg.stop_value > 0 and dd >= float(self.cfg.stop_value):
                return True, int(ExitReason.STOP_LOSS)

        return False, int(ExitReason.NONE)

    def _cooldown_for_exit_reason(self, exit_reason: int) -> int:
        if exit_reason == int(ExitReason.TAKE_PROFIT) or exit_reason == int(ExitReason.SIGNAL_EXIT):
            return int(self.cfg.cooldown_after_tp)
        if exit_reason == int(ExitReason.STOP_LOSS):
            return int(self.cfg.cooldown_after_stop)
        if exit_reason == int(ExitReason.MAX_HOLDING):
            return int(self.cfg.cooldown_after_time)
        return 0

    def step(
        self,
        *,
        t: int,
        desired_target_leg: np.ndarray,      # (N,) signal-owned desired
        current_pos_leg: np.ndarray,         # (N,)
        bid_sz_t: np.ndarray,                # (N,L)
        ask_sz_t: np.ndarray,                # (N,L)
        equity_t: float,
        execution_size_haircut: float,
        execution_max_levels_to_cross: int,
        # optional: for CROSSING trigger you pass z[t], z[t-1]
        # but if you already precomputed desired_target_leg as crossing-based, you can ignore these.
    ) -> GovernanceDecision:
        cfg = self.cfg
        st = self.state

        N = int(desired_target_leg.shape[0])
        desired = np.asarray(desired_target_leg, dtype=np.float64)
        pos = np.asarray(current_pos_leg, dtype=np.float64)

        # Default: executable target equals desired
        target = desired.copy()
        exit_reason = int(ExitReason.NONE)
        entry_block = int(EntryBlockReason.NONE)

        # In SIGNAL_OWNED mode, overlay does nothing.
        if (not cfg.enabled) or (cfg.mode == "SIGNAL_OWNED"):
            st.cooldown_remaining = _dec_nonneg(st.cooldown_remaining)
            return GovernanceDecision(
                target_leg=target,
                exit_reason=exit_reason,
                entry_block_reason=entry_block,
                cooldown_remaining=st.cooldown_remaining,
            )

        # IMPLEMENTABILITY_OWNED
        st.cooldown_remaining = _dec_nonneg(st.cooldown_remaining)

        # If in cooldown: force flat and block entry (this matches your “skip”)
        if st.cooldown_remaining > 0:
            entry_block = int(EntryBlockReason.COOLDOWN)
            target[:] = 0.0
            return GovernanceDecision(
                target_leg=target,
                exit_reason=exit_reason,
                entry_block_reason=entry_block,
                cooldown_remaining=st.cooldown_remaining,
            )

        # Apply stops/time exits based on realized path
        should_flatten, stop_reason = self._apply_stop_and_time_exits(
            t=t,
            current_pos_leg=pos,
            equity_t=equity_t,
        )
        if should_flatten:
            target[:] = 0.0
            exit_reason = stop_reason

            # start cooldown based on reason
            st.cooldown_remaining = self._cooldown_for_exit_reason(exit_reason)

            # update basket state to flat
            st.pos_side = 0
            st.entry_bar = -1
            st.entry_equity = float(equity_t)

            return GovernanceDecision(
                target_leg=target,
                exit_reason=exit_reason,
                entry_block_reason=int(EntryBlockReason.NONE),
                cooldown_remaining=st.cooldown_remaining,
            )

        # Determine if this bar is an ENTRY attempt, EXIT attempt, or HOLD
        is_flat = bool(np.all(np.abs(pos) < 1e-12))
        desired_flat = bool(np.all(np.abs(desired) < 1e-12))

        # If not flat already and desired becomes flat => treat as TP/signal exit (for cooldown)
        if (not is_flat) and desired_flat:
            # optionally gate exit by liquidity; default is False
            if cfg.exit_liquidity_gate:
                haircut = float(cfg.gate_size_haircut) if cfg.gate_size_haircut is not None else float(execution_size_haircut)
                max_lv = int(cfg.gate_max_levels_to_cross) if cfg.gate_max_levels_to_cross is not None else int(execution_max_levels_to_cross)

                trade = -pos  # what we need to do to flatten
                ok = _is_basket_feasible(
                    trade_leg=trade,
                    bid_sz_t=bid_sz_t,
                    ask_sz_t=ask_sz_t,
                    max_levels=max_lv,
                    haircut=haircut,
                )
                if not ok:
                    # cannot exit fully now; hold
                    entry_block = int(EntryBlockReason.LIQUIDITY)
                    target = pos.copy()
                    return GovernanceDecision(
                        target_leg=target,
                        exit_reason=int(ExitReason.NONE),
                        entry_block_reason=entry_block,
                        cooldown_remaining=st.cooldown_remaining,
                    )

            # accept exit
            target[:] = 0.0
            exit_reason = int(ExitReason.SIGNAL_EXIT)
            st.cooldown_remaining = self._cooldown_for_exit_reason(exit_reason)
            st.pos_side = 0
            st.entry_bar = -1
            st.entry_equity = float(equity_t)

            return GovernanceDecision(
                target_leg=target,
                exit_reason=exit_reason,
                entry_block_reason=int(EntryBlockReason.NONE),
                cooldown_remaining=st.cooldown_remaining,
            )

        # ENTRY attempt: currently flat but desired is non-flat
        if is_flat and (not desired_flat):
            if not cfg.entry_liquidity_gate:
                # accept entry immediately
                st.pos_side = 1  # sign doesn’t matter here; you can derive from desired vector if needed
                st.entry_bar = int(t)
                st.entry_equity = float(equity_t)
                return GovernanceDecision(
                    target_leg=target,
                    exit_reason=int(ExitReason.NONE),
                    entry_block_reason=int(EntryBlockReason.NONE),
                    cooldown_remaining=st.cooldown_remaining,
                )

            # liquidity feasibility: must be able to execute the ENTRY basket
            haircut = float(cfg.gate_size_haircut) if cfg.gate_size_haircut is not None else float(execution_size_haircut)
            max_lv = int(cfg.gate_max_levels_to_cross) if cfg.gate_max_levels_to_cross is not None else int(execution_max_levels_to_cross)

            trade = desired - pos  # pos is ~0, so trade ~ desired
            ok = _is_basket_feasible(
                trade_leg=trade,
                bid_sz_t=bid_sz_t,
                ask_sz_t=ask_sz_t,
                max_levels=max_lv,
                haircut=haircut,
            )

            if not ok:
                # SKIP: stay flat; do not set cooldown; next bar can try again if signal says so
                entry_block = int(EntryBlockReason.LIQUIDITY)
                target[:] = 0.0
                st.pos_side = 0
                st.entry_bar = -1
                st.entry_equity = float(equity_t)
                return GovernanceDecision(
                    target_leg=target,
                    exit_reason=int(ExitReason.NONE),
                    entry_block_reason=entry_block,
                    cooldown_remaining=st.cooldown_remaining,
                )

            # accept entry
            st.pos_side = 1
            st.entry_bar = int(t)
            st.entry_equity = float(equity_t)
            return GovernanceDecision(
                target_leg=target,
                exit_reason=int(ExitReason.NONE),
                entry_block_reason=int(EntryBlockReason.NONE),
                cooldown_remaining=st.cooldown_remaining,
            )

        # HOLD or rebalance while already in position:
        # In implementability mode you can optionally forbid “rebalances” and only allow hold/exit.
        # For now: pass through desired.
        return GovernanceDecision(
            target_leg=target,
            exit_reason=int(ExitReason.NONE),
            entry_block_reason=int(EntryBlockReason.NONE),
            cooldown_remaining=st.cooldown_remaining,
        )
