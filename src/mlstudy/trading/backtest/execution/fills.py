"""
backtest/execution/fills.py

L2 order book fill models.

Supported:
- TOP_OF_BOOK: fill at best bid/ask, optionally limited by top size * haircut
- ORDERBOOK_WALK: consume depth across levels to compute VWAP

This module is written NumPy-friendly and can be ported to Numba easily:
- avoid Python objects in hot loops
- keep signatures simple and numeric
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.types import Side, RejectReason


def top_of_book_fill(
    *,
    side: int,
    qty: float,
    bid_px_levels: np.ndarray,
    bid_sz_levels: np.ndarray,
    ask_px_levels: np.ndarray,
    ask_sz_levels: np.ndarray,
    size_haircut: float,
    allow_partial: bool,
) -> Tuple[float, float, int]:
    """
    Execute against best bid/ask only.
    Inputs are (L,) arrays but we only use level 0.
    Returns (filled_qty_signed, fill_px, reject_reason_int).
    """
    if qty <= 0:
        return 0.0, 0.0, int(RejectReason.MIN_TRADE)

    if side == int(Side.BUY):
        px = float(ask_px_levels[0])
        avail = float(ask_sz_levels[0]) * float(size_haircut)
    else:
        px = float(bid_px_levels[0])
        avail = float(bid_sz_levels[0]) * float(size_haircut)

    if avail <= 0:
        return 0.0, 0.0, int(RejectReason.INSUFFICIENT_DEPTH)

    fill = min(qty, avail) if allow_partial else (qty if qty <= avail else 0.0)
    if fill <= 0:
        return 0.0, 0.0, int(RejectReason.INSUFFICIENT_DEPTH)

    return float(side) * float(fill), px, int(RejectReason.NONE)


def walk_book_fill(
    *,
    side: int,
    qty: float,
    bid_px_levels: np.ndarray,
    bid_sz_levels: np.ndarray,
    ask_px_levels: np.ndarray,
    ask_sz_levels: np.ndarray,
    size_haircut: float,
    max_levels_to_cross: int,
    allow_partial: bool,
    reject_if_insufficient_depth: bool,
) -> Tuple[float, float, int, int]:
    """
    Walk the book to fill qty.
    Returns (filled_qty_signed, vwap_px, levels_used, reject_reason_int).
    """
    if qty <= 0:
        return 0.0, 0.0, 0, int(RejectReason.MIN_TRADE)

    if side == int(Side.BUY):
        px_levels = ask_px_levels
        sz_levels = ask_sz_levels
    else:
        px_levels = bid_px_levels
        sz_levels = bid_sz_levels

    L = int(px_levels.shape[0])
    take_levels = min(L, int(max_levels_to_cross))

    rem = float(qty)
    filled = 0.0
    notional = 0.0
    used = 0

    for k in range(take_levels):
        avail = float(sz_levels[k]) * float(size_haircut)
        if avail <= 0:
            continue
        take = avail if avail < rem else rem
        if take <= 0:
            break
        px = float(px_levels[k])
        notional += take * px
        filled += take
        rem -= take
        used = k + 1
        if rem <= 1e-12:
            break

    if filled <= 0:
        return 0.0, 0.0, used, int(RejectReason.INSUFFICIENT_DEPTH)

    if (filled < qty) and reject_if_insufficient_depth and (not allow_partial):
        # reject entirely if you require full fill and cannot
        return 0.0, 0.0, used, int(RejectReason.INSUFFICIENT_DEPTH)

    if (filled < qty) and reject_if_insufficient_depth and allow_partial:
        # reject partial fills if configured to reject on insufficient depth
        return 0.0, 0.0, used, int(RejectReason.INSUFFICIENT_DEPTH)

    vwap = notional / filled
    return float(side) * float(filled), float(vwap), used, int(RejectReason.NONE)
