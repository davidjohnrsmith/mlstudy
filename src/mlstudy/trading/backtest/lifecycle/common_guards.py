
"""
backtest/lifecycle/common_guards.py

Shared helper utilities for lifecycle state machines.
Keep these simple (NumPy-friendly) so they can be ported to Numba later if desired.
"""

from __future__ import annotations

from enum import IntEnum
import numpy as np


class LifeReason(IntEnum):
    NONE = 0

    # entries
    ENTRY_LONG = 10
    ENTRY_SHORT = 11

    # exits
    EXIT_SIGNAL = 20
    EXIT_TP = 21
    EXIT_STOP = 22
    EXIT_TIME = 23

    # blocked / prevented
    BLOCKED_COOLDOWN = 30
    BLOCKED_MISSING = 31


def dec_nonneg(x: np.ndarray) -> None:
    """Decrement positive integers in-place, clamp at 0."""
    x[x > 0] -= 1


def apply_min_trade_deadzone(q: np.ndarray, min_abs: float) -> np.ndarray:
    """Zero out tiny target positions."""
    if min_abs <= 0:
        return q
    out = q.copy()
    out[np.abs(out) < min_abs] = 0.0
    return out


def start_cooldown(cooldown: np.ndarray, idx: np.ndarray, bars: int) -> None:
    """Set cooldown bars for a subset of portfolios."""
    if bars <= 0:
        return
    cooldown[idx] = np.maximum(cooldown[idx], bars)


def enforce_time_stop(pos: np.ndarray, holding_bars: np.ndarray, max_hold: int) -> np.ndarray:
    """Return a boolean mask of positions that should be time-stopped."""
    if max_hold <= 0:
        return np.zeros_like(holding_bars, dtype=bool)
    return (pos != 0.0) & (holding_bars >= max_hold)
