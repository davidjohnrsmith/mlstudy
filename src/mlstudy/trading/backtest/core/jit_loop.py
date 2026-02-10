"""
backtest/core/jit_loop.py

Placeholder for a numba-compiled loop.
Right now: pure numpy loop implementing minimal execution + MTM.

Execution features included (minimal):
- control_mode: target_positions or orders
- mapping portfolios->legs via W (optional)
- execution_mode:
    - MID: execute at mtm_px
    - TOP_OF_BOOK: execute at best bid/ask for full qty (no depth)
    - ORDERBOOK_WALK: walk L2 levels for VWAP (with size haircuts)
    - LAST: execute at last_px
- optional participation cap vs traded_volume or adv (if provided)
- optional partial fills
- simple fees: fee_bps (per notional) or fee_per_unit (per size unit)

Risk/data policy are *not fully implemented* here; add inside this module or in separate JIT helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .types import (
    ControlMode,
    ExecMode,
    RejectReason,
)


def _walk_book_one_side(px_levels: np.ndarray, sz_levels: np.ndarray, qty: float, *, max_levels: int, haircut: float) -> Tuple[float, float, int, int]:
    """
    Walk one side of the book for a marketable order.
    Returns (filled_qty, vwap_px, levels_used, reject_reason_int).
    px_levels, sz_levels are (L,) for the side you hit (ask for buys, bid for sells).
    qty is positive desired fill size.
    """
    if qty <= 0:
        return 0.0, 0.0, 0, int(RejectReason.MIN_TRADE)

    L = int(px_levels.shape[0])
    take_levels = min(L, max_levels)
    rem = qty
    notional = 0.0
    filled = 0.0
    used = 0

    for k in range(take_levels):
        avail = float(sz_levels[k]) * haircut
        if avail <= 0:
            continue
        take = avail if avail < rem else rem
        if take <= 0:
            break
        price = float(px_levels[k])
        notional += take * price
        filled += take
        rem -= take
        used = k + 1
        if rem <= 1e-12:
            break

    if filled <= 0:
        return 0.0, 0.0, used, int(RejectReason.INSUFFICIENT_DEPTH)

    vwap = notional / filled
    return filled, vwap, used, int(RejectReason.NONE)


def run(
    *,
    datetimes_int: np.ndarray,
    # L2 book
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    # MTM
    mtm_px: np.ndarray,
    # optional microstructure
    last_px: Optional[np.ndarray],
    traded_volume: Optional[np.ndarray],
    adv: Optional[np.ndarray],
    tradable_mask: Optional[np.ndarray],
    # rates optional
    yields: Optional[np.ndarray],
    dv01s: Optional[np.ndarray],
    # controls + mapping
    controls: np.ndarray,
    W: Optional[np.ndarray],
    # packed params
    params_float: np.ndarray,
    params_int: np.ndarray,
    flags: np.ndarray,
    # init
    initial_capital: float,
    initial_positions: Optional[np.ndarray],
) -> Dict[str, Any]:
    T = int(datetimes_int.shape[0])
    N = int(bid_px.shape[1])
    M = int(controls.shape[1])

    # --- unpack a few params by index (must match packer.py order) ---
    contract_multiplier = float(params_float[0])
    lot_size = float(params_float[1])
    size_haircut = float(params_float[2])
    fee_bps = float(params_float[3])
    fee_per_unit = float(params_float[4])
    impact_k_bps = float(params_float[5])  # not applied here (placeholder)
    impact_alpha = float(params_float[6])  # not applied here
    noise_bps_std = float(params_float[7]) # not applied here
    min_trade = float(params_float[13])
    round_to = float(params_float[14])
    participation_cap = float(params_float[15])  # -1 means disabled

    control_mode = int(params_int[0])
    exec_mode = int(params_int[1])
    exec_lag = int(params_int[2])
    max_levels = int(params_int[3])

    allow_partials = bool(flags[0])
    carry_unfilled = bool(flags[1])
    reject_if_insufficient_depth = bool(flags[2])

    # --- outputs ---
    positions_leg = np.zeros((T, N), dtype=np.float64)
    cash = np.zeros(T, dtype=np.float64)
    equity = np.zeros(T, dtype=np.float64)
    pnl = np.zeros(T, dtype=np.float64)

    # very simple fills array (grow list; in production use preallocated structured array)
    fills = []  # list of tuples: (t, leg_i, side, qty, px, reason)

    # --- init state ---
    pos_leg = np.zeros(N, dtype=np.float64)
    cash_prev = float(initial_capital)

    if initial_positions is not None:
        init = np.asarray(initial_positions, dtype=np.float64)
        # interpret: if mapping enabled, init is (M,) portfolio; else (N,) legs
        if W is None:
            if init.shape[0] != N:
                raise ValueError("initial_positions must be (N,) when mapping is None")
            pos_leg[:] = init
        else:
            if init.shape[0] != M:
                raise ValueError("initial_positions must be (M,) when mapping is provided")
            pos_leg[:] = init @ W[0]

    # initial equity
    equity_prev = cash_prev + float(np.dot(pos_leg, mtm_px[0]) * contract_multiplier)

    # Pending orders if carry_unfilled
    pending_leg = np.zeros(N, dtype=np.float64)

    for t in range(T):
        # Determine which bar we execute on (lagged)
        te = t + exec_lag
        if te >= T:
            # no more execution possible; just mark-to-market
            mtm_t = mtm_px[t]
            eq = cash_prev + float(np.dot(pos_leg, mtm_t) * contract_multiplier)
            positions_leg[t] = pos_leg
            cash[t] = cash_prev
            equity[t] = eq
            pnl[t] = eq - equity_prev
            equity_prev = eq
            continue

        # Tradability check (if provided)
        tradable_ok = True
        if tradable_mask is not None:
            # require all legs tradable at execution time by default
            tradable_ok = bool(np.all(tradable_mask[te].astype(bool)))

        # Build desired trade in portfolio space or legs
        if control_mode == int(ControlMode.TARGET_POSITIONS):
            target = controls[t].astype(np.float64, copy=False)
            if W is None:
                if target.shape[0] != N:
                    raise ValueError("controls (target_positions) must be (T,N) when mapping is None")
                target_leg = target
            else:
                target_leg = target @ W[te]  # use execution-time weights
            desired_trade = target_leg - pos_leg
        else:
            order = controls[t].astype(np.float64, copy=False)
            if W is None:
                if order.shape[0] != N:
                    raise ValueError("controls (orders) must be (T,N) when mapping is None")
                desired_trade = order
            else:
                desired_trade = order @ W[te]

        # Add any pending (carry-unfilled)
        if carry_unfilled:
            desired_trade = desired_trade + pending_leg
            pending_leg[:] = 0.0

        # Rounding / min trade filter (per-leg)
        trade_leg = desired_trade.copy()
        for i in range(N):
            q = trade_leg[i]
            if abs(q) < min_trade:
                trade_leg[i] = 0.0
                continue
            if round_to > 0:
                trade_leg[i] = np.round(q / round_to) * round_to
            # enforce lot_size if specified
            if lot_size > 0:
                trade_leg[i] = np.round(trade_leg[i] / lot_size) * lot_size

        # If not tradable, either skip trading (simple behavior)
        if not tradable_ok:
            # skip trading => no fills; just MTM
            mtm_t = mtm_px[t]
            eq = cash_prev + float(np.dot(pos_leg, mtm_t) * contract_multiplier)
            positions_leg[t] = pos_leg
            cash[t] = cash_prev
            equity[t] = eq
            pnl[t] = eq - equity_prev
            equity_prev = eq
            continue

        # Execute each leg independently against book snapshot at te
        cash_now = cash_prev
        pos_now = pos_leg.copy()

        for i in range(N):
            q = float(trade_leg[i])
            if abs(q) <= 0:
                continue

            # participation cap
            if participation_cap >= 0:
                liq = None
                if traded_volume is not None:
                    liq = float(traded_volume[te, i])
                elif adv is not None:
                    liq = float(adv[te, i])
                if liq is not None and liq > 0:
                    cap = participation_cap * liq
                    if abs(q) > cap:
                        if allow_partials:
                            q = np.sign(q) * cap
                        else:
                            # reject
                            fills.append((te, i, int(np.sign(q)), 0.0, 0.0, int(RejectReason.PARTICIPATION_CAP)))
                            continue

            side = 1 if q > 0 else -1
            qty = abs(q)

            # choose execution price
            if exec_mode == int(ExecMode.MID):
                filled = qty
                px = float(mtm_px[te, i])
                reason = int(RejectReason.NONE)

            elif exec_mode == int(ExecMode.LAST):
                if last_px is None:
                    filled = 0.0
                    px = 0.0
                    reason = int(RejectReason.MISSING_BOOK)
                else:
                    filled = qty
                    px = float(last_px[te, i])
                    reason = int(RejectReason.NONE)

            elif exec_mode == int(ExecMode.TOP_OF_BOOK):
                if side > 0:
                    px = float(ask_px[te, i, 0])
                else:
                    px = float(bid_px[te, i, 0])
                # optionally haircut top size
                top_sz = float(ask_sz[te, i, 0] if side > 0 else bid_sz[te, i, 0]) * size_haircut
                if top_sz <= 0:
                    filled, px, _, reason = 0.0, 0.0, 0, int(RejectReason.INSUFFICIENT_DEPTH)
                else:
                    if qty <= top_sz or allow_partials:
                        filled = min(qty, top_sz)
                        reason = int(RejectReason.NONE) if filled > 0 else int(RejectReason.INSUFFICIENT_DEPTH)
                    else:
                        filled = 0.0
                        reason = int(RejectReason.INSUFFICIENT_DEPTH)

            elif exec_mode == int(ExecMode.ORDERBOOK_WALK):
                if side > 0:
                    filled, px, _, reason = _walk_book_one_side(
                        ask_px[te, i], ask_sz[te, i], qty, max_levels=max_levels, haircut=size_haircut
                    )
                else:
                    filled, px, _, reason = _walk_book_one_side(
                        bid_px[te, i], bid_sz[te, i], qty, max_levels=max_levels, haircut=size_haircut
                    )
                if reason != int(RejectReason.NONE):
                    if reject_if_insufficient_depth and reason == int(RejectReason.INSUFFICIENT_DEPTH):
                        filled = 0.0

            else:
                raise ValueError(f"Unknown exec_mode int: {exec_mode}")

            if filled <= 0:
                fills.append((te, i, side, 0.0, 0.0, reason))
                if carry_unfilled and allow_partials:
                    # carry full remaining
                    pending_leg[i] += np.sign(q) * qty
                continue

            signed_filled = side * filled

            # cash impact (simple): pay price*qty for buys, receive for sells
            trade_value = signed_filled * px * contract_multiplier
            cash_now -= trade_value

            # explicit fees: fee_bps per notional OR per unit
            if fee_bps != 0.0:
                # treat "notional" as px*qty*multiplier
                notional = abs(signed_filled) * px * contract_multiplier
                cash_now -= (fee_bps / 1e4) * notional
            if fee_per_unit != 0.0:
                cash_now -= abs(signed_filled) * fee_per_unit

            # update position
            pos_now[i] += signed_filled

            fills.append((te, i, side, signed_filled, px, reason))

            # carry remainder if partial and desired qty not fully filled
            if allow_partials and carry_unfilled and filled < qty:
                pending_leg[i] += np.sign(q) * (qty - filled)

        # Mark-to-market at time t (not te): this is a convention; you can switch to te if desired
        mtm_t = mtm_px[t]
        eq = cash_now + float(np.dot(pos_now, mtm_t) * contract_multiplier)

        positions_leg[t] = pos_now
        cash[t] = cash_now
        equity[t] = eq
        pnl[t] = eq - equity_prev

        # roll forward
        pos_leg = pos_now
        cash_prev = cash_now
        equity_prev = eq

    # build fills structured array (simple)
    fills_arr = None
    if len(fills) > 0:
        dtype = np.dtype(
            [
                ("t", np.int32),
                ("leg", np.int32),
                ("side", np.int8),
                ("qty", np.float64),
                ("px", np.float64),
                ("reason", np.int16),
            ]
        )
        fills_arr = np.array(fills, dtype=dtype)

    return {
        "positions_leg": positions_leg,
        "cash": cash,
        "equity": equity,
        "pnl": pnl,
        "fills": fills_arr,
        "meta": {"engine": "jit_loop_placeholder_numpy"},
    }
