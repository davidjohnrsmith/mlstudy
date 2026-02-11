"""
backtest/lifecycle/momentum.py

Momentum / trend-following position manager using precomputed momentum scores.

Typical rules (simple):
- Enter long when score >= entry
- Enter short when score <= -entry
- Exit when |score| <= exit (trend faded)
- Optional max holding time
- Optional cooldown after exits/stops

This is intentionally simple and strategy-agnostic; you can evolve it into
MA-cross, breakout, trailing stop, etc., by changing the entry/exit conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import PMState, StepOutput, PositionManager
from .common_guards import LifeReason, dec_nonneg, start_cooldown, enforce_time_stop


@dataclass(frozen=True)
class MomentumConfig:
    entry: float = 1.0
    exit: float = 0.2

    unit_size: float = 1.0
    max_hold_bars: int = 0

    # Optional stop on adverse flip/weakening (simple version)
    stop_on_flip: bool = True
    cooldown_after_exit: int = 0
    cooldown_after_stop: int = 0
    cooldown_after_time: int = 0

    treat_nan_as_flat: bool = True


class MomentumPositionManager(PositionManager):
    def __init__(self, momentum_score: np.ndarray, cfg: MomentumConfig):
        """
        momentum_score: (T, M) array, precomputed
        """
        self.s = momentum_score
        self.cfg = cfg

    def reset(self, M: int, *, initial_pos: np.ndarray | None = None) -> PMState:
        pos = np.zeros(M, dtype=np.float64) if initial_pos is None else initial_pos.astype(np.float64).copy()
        holding = np.zeros(M, dtype=np.int32)
        cooldown = np.zeros(M, dtype=np.int32)
        entry_ref = np.full(M, np.nan, dtype=np.float64)
        return PMState(pos=pos, holding_bars=holding, cooldown_bars=cooldown, entry_ref=entry_ref, aux=None)

    def step(self, t: int, state: PMState) -> StepOutput:
        cfg = self.cfg

        st = self.s[t].astype(np.float64, copy=False)
        M = st.shape[0]
        reason = np.full(M, int(LifeReason.NONE), dtype=np.int16)

        dec_nonneg(state.cooldown_bars)

        in_pos = state.pos != 0.0
        state.holding_bars[in_pos] += 1
        state.holding_bars[~in_pos] = 0

        if cfg.treat_nan_as_flat:
            nan_mask = ~np.isfinite(st)
        else:
            nan_mask = np.zeros(M, dtype=bool)

        target = state.pos.copy()

        # TIME STOP
        time_stop = enforce_time_stop(target, state.holding_bars, cfg.max_hold_bars)
        if np.any(time_stop):
            target[time_stop] = 0.0
            reason[time_stop] = int(LifeReason.EXIT_TIME)
            start_cooldown(state.cooldown_bars, np.where(time_stop)[0], cfg.cooldown_after_time)
            state.entry_ref[time_stop] = np.nan

        active = (target != 0.0) & ~time_stop

        # STOP on flip (optional): if long and score strongly negative, or short and score strongly positive
        if cfg.stop_on_flip:
            flip_stop = active & (
                ((target > 0.0) & (st <= -cfg.entry)) |
                ((target < 0.0) & (st >= cfg.entry))
            )
            if np.any(flip_stop):
                target[flip_stop] = 0.0
                reason[flip_stop] = int(LifeReason.EXIT_STOP)
                start_cooldown(state.cooldown_bars, np.where(flip_stop)[0], cfg.cooldown_after_stop)
                state.entry_ref[flip_stop] = np.nan
                active = (target != 0.0) & ~(time_stop | flip_stop)

        # EXIT when trend fades
        exit_mask = active & (np.abs(st) <= cfg.exit)
        if np.any(exit_mask):
            target[exit_mask] = 0.0
            reason[exit_mask] = int(LifeReason.EXIT_SIGNAL)
            start_cooldown(state.cooldown_bars, np.where(exit_mask)[0], cfg.cooldown_after_exit)
            state.entry_ref[exit_mask] = np.nan

        # NaN handling
        if np.any(nan_mask):
            target[nan_mask] = 0.0
            reason[nan_mask] = int(LifeReason.BLOCKED_MISSING)
            state.entry_ref[nan_mask] = np.nan

        # ENTRY (flat, not cooldown)
        flat = (target == 0.0)
        can_enter = flat & (state.cooldown_bars == 0) & ~nan_mask

        long_entry = can_enter & (st >= cfg.entry)
        short_entry = can_enter & (st <= -cfg.entry)

        if np.any(long_entry):
            target[long_entry] = +cfg.unit_size
            reason[long_entry] = int(LifeReason.ENTRY_LONG)
            state.entry_ref[long_entry] = st[long_entry]

        if np.any(short_entry):
            target[short_entry] = -cfg.unit_size
            reason[short_entry] = int(LifeReason.ENTRY_SHORT)
            state.entry_ref[short_entry] = st[short_entry]

        # Tag cooldown blocks (optional)
        blocked = flat & (state.cooldown_bars > 0) & ~nan_mask
        reason[(reason == int(LifeReason.NONE)) & blocked] = int(LifeReason.BLOCKED_COOLDOWN)

        state.pos[:] = target
        return StepOutput(target_pos=target, reason=reason)
