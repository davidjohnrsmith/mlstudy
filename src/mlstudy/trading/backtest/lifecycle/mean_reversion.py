"""
backtest/lifecycle/mean_reversion.py

Mean reversion position manager using precomputed z-scores.

Typical rules:
- Enter when z >= entry => short
- Enter when z <= -entry => long
- Exit when |z| <= exit
- Stop when |z| >= stop
- Optional max holding time + cooldown/quarantine

This outputs portfolio-space target positions (M,).
Sizing here is intentionally simple (fixed unit). You can:
- scale by |z| upstream and pass targets directly (skip lifecycle), OR
- extend this manager with a sizing function.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import PMState, StepOutput, PositionManager
from .common_guards import LifeReason, dec_nonneg, start_cooldown, enforce_time_stop


@dataclass(frozen=True)
class MeanReversionConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5

    unit_size: float = 1.0            # position size in portfolio units when in trade
    max_hold_bars: int = 0            # 0 disables time stop

    cooldown_after_tp: int = 0
    cooldown_after_stop: int = 0
    cooldown_after_time: int = 0

    treat_nan_as_flat: bool = True    # if z is nan, force flat


class MeanReversionPositionManager(PositionManager):
    def __init__(self, z_scores: np.ndarray, cfg: MeanReversionConfig):
        """
        z_scores: (T, M) array, precomputed
        """
        self.z = z_scores
        self.cfg = cfg

    def reset(self, M: int, *, initial_pos: np.ndarray | None = None) -> PMState:
        pos = np.zeros(M, dtype=np.float64) if initial_pos is None else initial_pos.astype(np.float64).copy()
        holding = np.zeros(M, dtype=np.int32)
        cooldown = np.zeros(M, dtype=np.int32)
        entry_ref = np.full(M, np.nan, dtype=np.float64)
        return PMState(pos=pos, holding_bars=holding, cooldown_bars=cooldown, entry_ref=entry_ref, aux=None)

    def step(self, t: int, state: PMState) -> StepOutput:
        cfg = self.cfg

        zt = self.z[t].astype(np.float64, copy=False)  # (M,)
        M = zt.shape[0]

        reason = np.full(M, int(LifeReason.NONE), dtype=np.int16)

        # Decrement cooldown at start of bar
        dec_nonneg(state.cooldown_bars)

        # Update holding time for active positions
        in_pos = state.pos != 0.0
        state.holding_bars[in_pos] += 1
        state.holding_bars[~in_pos] = 0

        # Handle NaNs
        if cfg.treat_nan_as_flat:
            nan_mask = ~np.isfinite(zt)
        else:
            nan_mask = np.zeros(M, dtype=bool)

        target = state.pos.copy()

        # TIME STOP
        time_stop = enforce_time_stop(state.pos, state.holding_bars, cfg.max_hold_bars)
        if np.any(time_stop):
            target[time_stop] = 0.0
            reason[time_stop] = int(LifeReason.EXIT_TIME)
            start_cooldown(state.cooldown_bars, np.where(time_stop)[0], cfg.cooldown_after_time)
            state.entry_ref[time_stop] = np.nan

        # STOP LOSS by z
        # Only apply if currently in position and not already time-stopped this bar
        active = (target != 0.0) & ~time_stop
        stop_mask = active & (np.abs(zt) >= cfg.stop_z)
        if np.any(stop_mask):
            target[stop_mask] = 0.0
            reason[stop_mask] = int(LifeReason.EXIT_STOP)
            start_cooldown(state.cooldown_bars, np.where(stop_mask)[0], cfg.cooldown_after_stop)
            state.entry_ref[stop_mask] = np.nan

        # EXIT (take profit / mean-reverted)
        active = (target != 0.0) & ~(time_stop | stop_mask)
        exit_mask = active & (np.abs(zt) <= cfg.exit_z)
        if np.any(exit_mask):
            target[exit_mask] = 0.0
            reason[exit_mask] = int(LifeReason.EXIT_TP)
            start_cooldown(state.cooldown_bars, np.where(exit_mask)[0], cfg.cooldown_after_tp)
            state.entry_ref[exit_mask] = np.nan

        # Force flat on NaN if configured
        if np.any(nan_mask):
            target[nan_mask] = 0.0
            reason[nan_mask] = int(LifeReason.BLOCKED_MISSING)
            state.entry_ref[nan_mask] = np.nan

        # ENTRY (only if flat and not in cooldown and not NaN)
        flat = (target == 0.0)
        can_enter = flat & (state.cooldown_bars == 0) & ~nan_mask

        long_entry = can_enter & (zt <= -cfg.entry_z)
        short_entry = can_enter & (zt >= cfg.entry_z)

        if np.any(long_entry):
            target[long_entry] = +cfg.unit_size
            reason[long_entry] = int(LifeReason.ENTRY_LONG)
            state.entry_ref[long_entry] = zt[long_entry]

        if np.any(short_entry):
            target[short_entry] = -cfg.unit_size
            reason[short_entry] = int(LifeReason.ENTRY_SHORT)
            state.entry_ref[short_entry] = zt[short_entry]

        # If blocked by cooldown (optional reason tagging)
        blocked = flat & (state.cooldown_bars > 0) & ~nan_mask
        reason[(reason == int(LifeReason.NONE)) & blocked] = int(LifeReason.BLOCKED_COOLDOWN)

        # Commit state.pos to new target
        state.pos[:] = target

        return StepOutput(target_pos=target, reason=reason)
