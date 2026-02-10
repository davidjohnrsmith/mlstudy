"""
backtest/execution/liquidity.py

Liquidity throttles, e.g. participation caps vs traded volume / ADV.

This module does NOT decide fill prices; it just clips desired size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.types import RejectReason


def apply_participation_cap(
    q: float,
    *,
    participation_cap: Optional[float],
    traded_volume: Optional[float],
    adv: Optional[float],
    allow_partial: bool,
) -> Tuple[float, int]:
    """
    Apply participation cap to a single desired trade size q (signed).
    Returns (clipped_q, reject_reason_int).
    - If cap is None -> unchanged, reason NONE
    - If liquidity source missing -> unchanged, reason NONE (caller may decide)
    """
    if participation_cap is None:
        return q, int(RejectReason.NONE)

    liq = None
    if traded_volume is not None and traded_volume > 0:
        liq = traded_volume
    elif adv is not None and adv > 0:
        liq = adv

    if liq is None:
        # cannot apply cap without a liquidity proxy; treat as no-op
        return q, int(RejectReason.NONE)

    cap = float(participation_cap) * float(liq)
    if cap <= 0:
        return 0.0, int(RejectReason.PARTICIPATION_CAP)

    if abs(q) <= cap:
        return q, int(RejectReason.NONE)

    if allow_partial:
        return float(np.sign(q) * cap), int(RejectReason.NONE)

    return 0.0, int(RejectReason.PARTICIPATION_CAP)
