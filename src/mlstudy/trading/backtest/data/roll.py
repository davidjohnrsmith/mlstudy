"""
backtest/data/roll.py

Instrument roll utilities.

In this architecture, "legs" are the underlying instruments dimension N.
Rolling can mean:
- replacing instrument i with instrument j after a roll date
- forcing flatten before switching
- building active_mask accordingly

This file provides a minimal "apply roll map" utility that rewrites arrays by mapping
old columns to new columns over time.

You can keep roll logic upstream; this is optional.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

from ..core.types import BacktestInputs, OrderBookL2, MarketState, Availability


def apply_roll_map(
    inputs: BacktestInputs,
    *,
    roll_dates_int: np.ndarray,
    roll_from_to: np.ndarray,
    mode: str = "AUTO_SWITCH",
) -> BacktestInputs:
    """
    Apply a roll mapping to the *column dimension N* over time.

    Args:
      roll_dates_int: (R,) increasing dates when roll becomes effective (inclusive)
      roll_from_to: (R,2) integer pairs (from_idx, to_idx) applied at each roll date
      mode:
        - "AUTO_SWITCH": after roll date, data/quotes for 'from' are replaced by 'to'
        - "FREEZE": do nothing (placeholder)
        - "FLATTEN_THEN_SWITCH": not handled here (position-level behavior belongs in engine/lifecycle)

    Returns:
      New BacktestInputs with rolled market/book arrays.
    """
    if mode == "FREEZE":
        return inputs
    if mode not in ("AUTO_SWITCH", "FLATTEN_THEN_SWITCH"):
        raise ValueError(f"Unknown roll mode: {mode}")

    dt = inputs.datetimes_int
    T = int(dt.shape[0])
    bid_px = inputs.orderbook.bid_px
    _, N, L = bid_px.shape

    roll_dates_int = np.asarray(roll_dates_int).astype(np.int64)
    roll_from_to = np.asarray(roll_from_to).astype(np.int64)
    if roll_from_to.ndim != 2 or roll_from_to.shape[1] != 2:
        raise ValueError("roll_from_to must be (R,2)")

    # Make copies
    ob = inputs.orderbook
    mk = inputs.market
    av = inputs.availability

    bid_px_new = ob.bid_px.copy()
    bid_sz_new = ob.bid_sz.copy()
    ask_px_new = ob.ask_px.copy()
    ask_sz_new = ob.ask_sz.copy()

    def _copy_2d(opt: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if opt is None else opt.copy()

    last_px_new = _copy_2d(ob.last_px)
    last_sz_new = _copy_2d(ob.last_sz)
    traded_volume_new = _copy_2d(ob.traded_volume)
    adv_new = _copy_2d(ob.adv)

    mtm_px_new = _copy_2d(mk.mtm_px)
    yields_new = _copy_2d(mk.yields)
    dv01s_new = _copy_2d(mk.dv01s)
    accrual_new = _copy_2d(mk.accrual_factors)

    active_new = _copy_2d(av.active_mask)
    tradable_new = _copy_2d(av.tradable_mask)

    # For each roll event, swap columns from -> to after effective date
    for k in range(roll_from_to.shape[0]):
        eff_date = int(roll_dates_int[k])
        frm, to = int(roll_from_to[k, 0]), int(roll_from_to[k, 1])
        if not (0 <= frm < N and 0 <= to < N):
            raise ValueError(f"Invalid roll mapping indices: {frm}->{to} for N={N}")

        # find start index in time
        start = int(np.searchsorted(dt, eff_date, side="left"))
        if start >= T:
            continue

        bid_px_new[start:, frm, :] = bid_px_new[start:, to, :]
        bid_sz_new[start:, frm, :] = bid_sz_new[start:, to, :]
        ask_px_new[start:, frm, :] = ask_px_new[start:, to, :]
        ask_sz_new[start:, frm, :] = ask_sz_new[start:, to, :]

        if last_px_new is not None:
            last_px_new[start:, frm] = last_px_new[start:, to]
        if last_sz_new is not None:
            last_sz_new[start:, frm] = last_sz_new[start:, to]
        if traded_volume_new is not None:
            traded_volume_new[start:, frm] = traded_volume_new[start:, to]
        if adv_new is not None:
            adv_new[start:, frm] = adv_new[start:, to]

        if mtm_px_new is not None:
            mtm_px_new[start:, frm] = mtm_px_new[start:, to]
        if yields_new is not None:
            yields_new[start:, frm] = yields_new[start:, to]
        if dv01s_new is not None:
            dv01s_new[start:, frm] = dv01s_new[start:, to]
        if accrual_new is not None:
            accrual_new[start:, frm] = accrual_new[start:, to]

        # Active/tradable masks: if present, copy too
        if active_new is not None:
            active_new[start:, frm] = active_new[start:, to]
        if tradable_new is not None:
            tradable_new[start:, frm] = tradable_new[start:, to]

    new_inputs = BacktestInputs(
        datetimes_int=inputs.datetimes_int,
        orderbook=OrderBookL2(
            bid_px=bid_px_new,
            bid_sz=bid_sz_new,
            ask_px=ask_px_new,
            ask_sz=ask_sz_new,
            last_px=last_px_new,
            last_sz=last_sz_new,
            traded_volume=traded_volume_new,
            adv=adv_new,
        ),
        market=MarketState(
            mtm_px=mtm_px_new,
            yields=yields_new,
            dv01s=dv01s_new,
            coupons=mk.coupons,
            accrual_factors=accrual_new,
            repo_rates=mk.repo_rates,
        ),
        mapping=inputs.mapping,
        controls=inputs.controls,
        availability=Availability(active_mask=active_new, tradable_mask=tradable_new),
    )
    return new_inputs
