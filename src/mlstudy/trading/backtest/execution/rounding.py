"""
backtest/execution/rounding.py

Order/target rounding utilities.

These operate in "size units" (contracts, notional, shares, etc.)
as defined by cfg.instrument and your strategy controls.
"""

from __future__ import annotations

import numpy as np


def apply_rounding(
    q: np.ndarray,
    *,
    min_trade_size: float,
    round_to_size: float,
    lot_size: float,
) -> np.ndarray:
    """
    Apply per-element rounding + deadzone.

    - min_trade_size: if |q| < min_trade_size, set to 0
    - round_to_size: if >0, round to nearest increment
    - lot_size: if >0, round to nearest lot (applied after round_to_size)

    Returns a new array (does not modify q in-place).
    """
    out = q.astype(np.float64, copy=True)

    if min_trade_size > 0:
        out[np.abs(out) < min_trade_size] = 0.0

    if round_to_size and round_to_size > 0:
        out = np.round(out / round_to_size) * round_to_size

    if lot_size and lot_size > 0:
        out = np.round(out / lot_size) * lot_size

    return out
