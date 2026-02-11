"""
Mean-reversion backtester core loop.

Provides a Python reference implementation and an optional Numba JIT-compiled
version.  Both share identical logic; outputs must match on the same inputs.

Sign conventions
----------------
- pos_side: +1 = LONG, -1 = SHORT
- zscore > 0 -> cheap  -> enter LONG
- zscore < 0 -> expensive -> enter SHORT
- All TP / SL thresholds are applied to **direction-adjusted** values::

      adj_yield_delta = -pos_side * (package_yield_bps[t] - entry_pkg_yield)
      adj_z           = -pos_side * zscore[t]

  Positive adj_yield_delta = profitable; positive adj_z = z-score reverted.

DV01 / sizing convention
------------------------
- dv01[t, i]: price change per 1 bp yield change per 1 unit of par notional.
- size_i = target_notional_ref * dv01[t, ref] * hedge_ratios[i] / dv01[t, i]
- basket_dv01_ref = target_notional_ref * dv01[t, ref]  (ref-leg DV01 exposure)
- acceptable_cost (price units) = acceptable_yield_bps * basket_dv01_ref

Basket execution cost
---------------------
- Per-leg cost (always positive) = |vwap_i - mid_i| * |size_i|
- Basket cost = sum of per-leg costs.

No lookahead
------------
All signal inputs (zscore, expected_yield_pnl_bps, package_yield_bps)
are assumed to be already lagged by the caller.  No internal lagging is
performed; at bar *t* the loop only reads ``signals[t]``.
"""

from __future__ import annotations

import numpy as np

# Attempt-code constants are plain ints so the same source compiles under
# Numba ``@njit`` and runs in pure Python.
#
# We inline the values rather than importing from types.py because Numba
# cannot import from a non-JIT module inside an ``@njit`` function.  The
# values are verified in tests against the canonical types.py definitions.

_NO_ACTION = 0

_ENTRY_OK = 100
_ENTRY_NO_SIGNAL = 101
_ENTRY_INVALID_BOOK = 102
_ENTRY_NO_LIQUIDITY = 103
_ENTRY_TOO_WIDE = 104
_ENTRY_IN_COOLDOWN = 105

_EXIT_TP_OK = 200
_EXIT_TP_INVALID_BOOK = 202
_EXIT_TP_NO_LIQUIDITY = 203
_EXIT_TP_TOO_WIDE = 204

_EXIT_SL_FORCED = 307
_EXIT_SL_NO_LIQUIDITY = 303

_EXIT_TIME_FORCED = 407
_EXIT_TIME_NO_LIQUIDITY = 403

_VALIDATE_REF_ONLY = 0

_TRADE_ENTRY = 0
_TRADE_EXIT_TP = 1
_TRADE_EXIT_SL = 2
_TRADE_EXIT_TIME = 3

# ---------------------------------------------------------------------------
# Numba availability
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit  # type: ignore[import-untyped]
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[misc]
        """Identity decorator when Numba is not installed."""
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


# ======================================================================
# Helpers (Numba-safe: only scalars, arrays, and basic control flow)
# ======================================================================

def _walk_book(px, sz, qty, max_levels, haircut):
    """Walk one side of an L2 book.

    Parameters
    ----------
    px : (L,) prices at each level (best first).
    sz : (L,) sizes at each level.
    qty : unsigned quantity to fill.
    max_levels : max levels to cross.
    haircut : fraction of displayed size usable (0-1).

    Returns
    -------
    filled : unsigned filled quantity.
    vwap : volume-weighted average price (0.0 if nothing filled).
    """
    filled = 0.0
    notional = 0.0
    remaining = qty
    n_levels = len(px)
    if max_levels < n_levels:
        n_levels = max_levels
    for lev in range(n_levels):
        if remaining <= 1e-15:
            break
        avail = sz[lev] * haircut
        if avail <= 1e-15:
            continue
        take = remaining if remaining < avail else avail
        filled += take
        notional += take * px[lev]
        remaining -= take
    vwap = notional / filled if filled > 1e-15 else 0.0
    return filled, vwap


def _check_market_valid(bid0, ask0, mid, scope, ref_idx, N):
    """Return True if bid0 <= mid <= ask0 (and both positive).

    scope == 0 -> check reference leg only.
    scope != 0 -> check all legs.
    """
    if scope == _VALIDATE_REF_ONLY:
        lo = ref_idx
        hi = ref_idx + 1
    else:
        lo = 0
        hi = N
    for i in range(lo, hi):
        if bid0[i] <= 0.0 or ask0[i] <= 0.0:
            return False
        if bid0[i] > mid[i] or mid[i] > ask0[i]:
            return False
    return True


# ======================================================================
# Core loop – pure Python reference
# ======================================================================

def mr_loop(
    # -- Market data -------------------------------------------------------
    bid_px,                     # (T, N, L)
    bid_sz,                     # (T, N, L)
    ask_px,                     # (T, N, L)
    ask_sz,                     # (T, N, L)
    mid_px,                     # (T, N) independent mid prices
    dv01,                       # (T, N)
    # -- Signals (already lagged) ------------------------------------------
    zscore,                     # (T,)
    expected_yield_pnl_bps,     # (T,)
    package_yield_bps,          # (T,)
    # -- Static inputs -----------------------------------------------------
    hedge_ratios,               # (N,) yield-space ratios, r_ref=1, sum=0
    ref_idx,                    # int – reference leg index
    # -- Config scalars ----------------------------------------------------
    target_notional_ref,        # float
    entry_z_threshold,          # float
    tp_zscore_soft,             # float
    tp_yield_soft,              # float
    tp_yield_hard,              # float
    sl_yield_hard,              # float
    max_holding_bars,           # int
    yield_pnl_multiplier,       # float
    entry_cost_premium,         # float (yield bps)
    tp_cost_premium,            # float (yield bps)
    sl_cost_premium,            # float (yield bps – unused for forced exits)
    tp_quarantine,              # int (bars)
    sl_quarantine,              # int (bars)
    time_quarantine,            # int (bars)
    max_levels,                 # int
    haircut,                    # float [0, 1]
    validate_scope,             # int (0=REF_ONLY, 1=ALL_LEGS)
    initial_capital,            # float
):
    """Run the mean-reversion backtest loop.

    Returns a tuple of numpy arrays – see the *Returns* section in
    :func:`mr_loop_jit` for the exact layout.
    """
    T = bid_px.shape[0]
    N = bid_px.shape[1]

    # ---- per-bar output arrays -------------------------------------------
    out_pos = np.zeros((T, N), dtype=np.float64)
    out_cash = np.zeros(T, dtype=np.float64)
    out_equity = np.zeros(T, dtype=np.float64)
    out_pnl = np.zeros(T, dtype=np.float64)
    out_codes = np.zeros(T, dtype=np.int32)
    out_state = np.zeros(T, dtype=np.int32)
    out_holding = np.zeros(T, dtype=np.int32)

    # ---- per-trade output arrays (pre-allocated, max T trades) -----------
    tr_bar = np.full(T, -1, dtype=np.int64)
    tr_type = np.zeros(T, dtype=np.int32)
    tr_side = np.zeros(T, dtype=np.int32)
    tr_sizes = np.zeros((T, N), dtype=np.float64)
    tr_vwaps = np.zeros((T, N), dtype=np.float64)
    tr_mids = np.zeros((T, N), dtype=np.float64)
    tr_cost = np.zeros(T, dtype=np.float64)
    tr_code = np.zeros(T, dtype=np.int32)
    tr_pkg_yield = np.zeros(T, dtype=np.float64)
    n_trades = 0

    # ---- mutable state ---------------------------------------------------
    pos = np.zeros(N, dtype=np.float64)
    cash = initial_capital
    state = 0          # 0 flat, +1 long, -1 short
    cooldown = 0
    entry_bar = -1
    entry_pkg_yield = 0.0
    holding = 0
    prev_equity = initial_capital

    for t in range(T):
        code = _NO_ACTION

        # ==================================================================
        # EXIT LOGIC (in position)
        # ==================================================================
        if state != 0:
            pos_side = state
            holding += 1

            raw_yield_delta = package_yield_bps[t] - entry_pkg_yield
            adj_yield_delta = -pos_side * raw_yield_delta
            adj_z = -pos_side * zscore[t]

            # ---- stop-loss check -----------------------------------------
            sl_triggered = adj_yield_delta < -sl_yield_hard

            # ---- max-holding check ---------------------------------------
            time_triggered = (max_holding_bars > 0) and (holding >= max_holding_bars)

            # ---- take-profit check ---------------------------------------
            tp_z_cond = adj_z > tp_zscore_soft
            tp_ys_cond = adj_yield_delta > tp_yield_soft
            tp_yh_cond = adj_yield_delta > tp_yield_hard
            tp_triggered = (tp_z_cond and tp_ys_cond) or tp_yh_cond

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FORCED EXIT (SL / TIME) – no cost acceptance check
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if sl_triggered or time_triggered:
                is_sl = sl_triggered  # SL has priority over TIME
                all_filled = True
                exit_vwaps = np.empty(N, dtype=np.float64)
                for i in range(N):
                    if abs(pos[i]) < 1e-15:
                        exit_vwaps[i] = mid_px[t, i]
                        continue
                    qty_abs = abs(pos[i])
                    if pos[i] > 0:  # sell
                        filled, vwap = _walk_book(
                            bid_px[t, i], bid_sz[t, i],
                            qty_abs, max_levels, haircut,
                        )
                    else:  # buy to cover
                        filled, vwap = _walk_book(
                            ask_px[t, i], ask_sz[t, i],
                            qty_abs, max_levels, haircut,
                        )
                    if filled < qty_abs - 1e-10:
                        all_filled = False
                        break
                    exit_vwaps[i] = vwap

                if all_filled:
                    # execute: update cash, record trade
                    bcost = 0.0
                    for i in range(N):
                        if abs(pos[i]) < 1e-15:
                            continue
                        cash += pos[i] * exit_vwaps[i]  # sell gets +, buy to cover gets -
                        # negate: cash -= trade_size * vwap where trade = -pos
                        # equiv: cash += pos * vwap  (selling what we have)
                        # Actually: cash change = -(-pos[i]) * vwap = pos[i]*vwap? No.
                        # Let trade_i = -pos[i] (the exit trade).
                        # cash -= trade_i * vwap_i => cash -= (-pos[i]) * vwap_i
                        #                         => cash += pos[i] * vwap_i
                        bcost += abs(exit_vwaps[i] - mid_px[t, i]) * abs(pos[i])
                    # NOTE: correct cash formula above: cash += pos[i]*vwap is wrong
                    # because we SELL when pos>0 (receive cash) and BUY when pos<0 (pay cash).
                    # sell: receive qty*vwap => cash += |pos|*vwap (pos>0)
                    # buy:  pay    qty*vwap => cash -= |pos|*vwap (pos<0)
                    # Combined: cash -= (-pos[i])*vwap = cash += pos[i]*vwap.  Correct.

                    # record trade
                    tr_bar[n_trades] = t
                    tr_type[n_trades] = _TRADE_EXIT_SL if is_sl else _TRADE_EXIT_TIME
                    tr_side[n_trades] = pos_side
                    for i in range(N):
                        tr_sizes[n_trades, i] = -pos[i]
                        tr_vwaps[n_trades, i] = exit_vwaps[i]
                        tr_mids[n_trades, i] = mid_px[t, i]
                    tr_cost[n_trades] = bcost
                    tr_code[n_trades] = _EXIT_SL_FORCED if is_sl else _EXIT_TIME_FORCED
                    tr_pkg_yield[n_trades] = package_yield_bps[t]
                    n_trades += 1

                    # flatten
                    for i in range(N):
                        pos[i] = 0.0
                    code = _EXIT_SL_FORCED if is_sl else _EXIT_TIME_FORCED
                    cooldown = sl_quarantine if is_sl else time_quarantine
                    state = 0
                    holding = 0
                    entry_bar = -1
                else:
                    code = _EXIT_SL_NO_LIQUIDITY if is_sl else _EXIT_TIME_NO_LIQUIDITY

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # GATED EXIT (TP) – with cost acceptance
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif tp_triggered:
                # market validity
                valid = _check_market_valid(
                    bid_px[t, :, 0], ask_px[t, :, 0], mid_px[t],
                    validate_scope, ref_idx, N,
                )
                if not valid:
                    code = _EXIT_TP_INVALID_BOOK
                else:
                    all_filled = True
                    exit_vwaps = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        if abs(pos[i]) < 1e-15:
                            exit_vwaps[i] = mid_px[t, i]
                            continue
                        qty_abs = abs(pos[i])
                        if pos[i] > 0:
                            filled, vwap = _walk_book(
                                bid_px[t, i], bid_sz[t, i],
                                qty_abs, max_levels, haircut,
                            )
                        else:
                            filled, vwap = _walk_book(
                                ask_px[t, i], ask_sz[t, i],
                                qty_abs, max_levels, haircut,
                            )
                        if filled < qty_abs - 1e-10:
                            all_filled = False
                            break
                        exit_vwaps[i] = vwap

                    if not all_filled:
                        code = _EXIT_TP_NO_LIQUIDITY
                    else:
                        # cost check (use actual position ref-leg size)
                        bcost = 0.0
                        for i in range(N):
                            bcost += abs(exit_vwaps[i] - mid_px[t, i]) * abs(pos[i])

                        ref_dv01_not = abs(pos[ref_idx]) * dv01[t, ref_idx]
                        acc_yield = (yield_pnl_multiplier
                                     * expected_yield_pnl_bps[t]
                                     - tp_cost_premium)
                        acc_cost = acc_yield * ref_dv01_not

                        if acc_cost <= 0.0 or bcost > acc_cost:
                            code = _EXIT_TP_TOO_WIDE
                        else:
                            # execute TP exit
                            for i in range(N):
                                if abs(pos[i]) < 1e-15:
                                    continue
                                cash += pos[i] * exit_vwaps[i]

                            tr_bar[n_trades] = t
                            tr_type[n_trades] = _TRADE_EXIT_TP
                            tr_side[n_trades] = pos_side
                            for i in range(N):
                                tr_sizes[n_trades, i] = -pos[i]
                                tr_vwaps[n_trades, i] = exit_vwaps[i]
                                tr_mids[n_trades, i] = mid_px[t, i]
                            tr_cost[n_trades] = bcost
                            tr_code[n_trades] = _EXIT_TP_OK
                            tr_pkg_yield[n_trades] = package_yield_bps[t]
                            n_trades += 1

                            for i in range(N):
                                pos[i] = 0.0
                            code = _EXIT_TP_OK
                            cooldown = tp_quarantine
                            state = 0
                            holding = 0
                            entry_bar = -1
            else:
                code = _NO_ACTION

        # ==================================================================
        # ENTRY LOGIC (flat)
        # ==================================================================
        elif state == 0:
            if cooldown > 0:
                cooldown -= 1
                if abs(zscore[t]) > entry_z_threshold:
                    code = _ENTRY_IN_COOLDOWN
                else:
                    code = _NO_ACTION
            elif abs(zscore[t]) > entry_z_threshold:
                intended_side = 1 if zscore[t] > 0 else -1

                # -- compute leg sizes ------------------------------------
                ref_dv01_t = dv01[t, ref_idx]
                if ref_dv01_t < 1e-15:
                    code = _ENTRY_INVALID_BOOK
                else:
                    sizes_ok = True
                    trade_sizes = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        if dv01[t, i] < 1e-15:
                            sizes_ok = False
                            break
                        trade_sizes[i] = (intended_side
                                          * target_notional_ref
                                          * ref_dv01_t
                                          * hedge_ratios[i]
                                          / dv01[t, i])
                    if not sizes_ok:
                        code = _ENTRY_INVALID_BOOK
                    else:
                        # -- acceptable cost ------------------------------
                        basket_dv01_ref = target_notional_ref * ref_dv01_t
                        acc_yield = (yield_pnl_multiplier
                                     * expected_yield_pnl_bps[t]
                                     - entry_cost_premium)
                        acc_cost = acc_yield * basket_dv01_ref

                        if acc_cost <= 0.0:
                            code = _ENTRY_TOO_WIDE
                        else:
                            # -- market validity --------------------------
                            valid = _check_market_valid(
                                bid_px[t, :, 0], ask_px[t, :, 0],
                                mid_px[t], validate_scope, ref_idx, N,
                            )
                            if not valid:
                                code = _ENTRY_INVALID_BOOK
                            else:
                                # -- L2 walk per leg ----------------------
                                fill_vwaps = np.empty(N, dtype=np.float64)
                                all_filled = True
                                for i in range(N):
                                    if abs(trade_sizes[i]) < 1e-15:
                                        fill_vwaps[i] = mid_px[t, i]
                                        continue
                                    qty_abs = abs(trade_sizes[i])
                                    if trade_sizes[i] > 0:
                                        filled, vwap = _walk_book(
                                            ask_px[t, i], ask_sz[t, i],
                                            qty_abs, max_levels, haircut,
                                        )
                                    else:
                                        filled, vwap = _walk_book(
                                            bid_px[t, i], bid_sz[t, i],
                                            qty_abs, max_levels, haircut,
                                        )
                                    if filled < qty_abs - 1e-10:
                                        all_filled = False
                                        break
                                    fill_vwaps[i] = vwap

                                if not all_filled:
                                    code = _ENTRY_NO_LIQUIDITY
                                else:
                                    # -- basket cost check ----------------
                                    bcost = 0.0
                                    for i in range(N):
                                        bcost += (abs(fill_vwaps[i] - mid_px[t, i])
                                                  * abs(trade_sizes[i]))
                                    if bcost > acc_cost:
                                        code = _ENTRY_TOO_WIDE
                                    else:
                                        # -- EXECUTE ENTRY ----------------
                                        for i in range(N):
                                            cash -= trade_sizes[i] * fill_vwaps[i]
                                            pos[i] += trade_sizes[i]

                                        tr_bar[n_trades] = t
                                        tr_type[n_trades] = _TRADE_ENTRY
                                        tr_side[n_trades] = intended_side
                                        for i in range(N):
                                            tr_sizes[n_trades, i] = trade_sizes[i]
                                            tr_vwaps[n_trades, i] = fill_vwaps[i]
                                            tr_mids[n_trades, i] = mid_px[t, i]
                                        tr_cost[n_trades] = bcost
                                        tr_code[n_trades] = _ENTRY_OK
                                        tr_pkg_yield[n_trades] = package_yield_bps[t]
                                        n_trades += 1

                                        code = _ENTRY_OK
                                        state = intended_side
                                        entry_bar = t
                                        entry_pkg_yield = package_yield_bps[t]
                                        holding = 0
            else:
                code = _ENTRY_NO_SIGNAL

        # ==================================================================
        # MTM & PnL (always, even on invalid bars)
        # ==================================================================
        equity_t = cash
        for i in range(N):
            equity_t += pos[i] * mid_px[t, i]

        pnl_t = equity_t - prev_equity
        prev_equity = equity_t

        # ---- store per-bar outputs ---------------------------------------
        for i in range(N):
            out_pos[t, i] = pos[i]
        out_cash[t] = cash
        out_equity[t] = equity_t
        out_pnl[t] = pnl_t
        out_codes[t] = code
        out_state[t] = state
        out_holding[t] = holding

    # ---- return as tuple (Numba-compatible) ------------------------------
    return (
        out_pos,        # 0  (T, N)
        out_cash,       # 1  (T,)
        out_equity,     # 2  (T,)
        out_pnl,        # 3  (T,)
        out_codes,      # 4  (T,) int32
        out_state,      # 5  (T,) int32
        out_holding,    # 6  (T,) int32
        tr_bar,         # 7  (T,) int64  (-1 = unused slot)
        tr_type,        # 8  (T,) int32
        tr_side,        # 9  (T,) int32
        tr_sizes,       # 10 (T, N)
        tr_vwaps,       # 11 (T, N)
        tr_mids,        # 12 (T, N)
        tr_cost,        # 13 (T,)
        tr_code,        # 14 (T,) int32
        tr_pkg_yield,   # 15 (T,)
        n_trades,       # 16 int
    )


# ======================================================================
# JIT-compiled version (identical logic, compiled if Numba is present)
# ======================================================================

# The helpers must be separately compiled for Numba to resolve them.
_walk_book_jit = _njit(cache=True)(_walk_book)
_check_market_valid_jit = _njit(cache=True)(_check_market_valid)


def _mr_loop_jit_impl(
    bid_px, bid_sz, ask_px, ask_sz, mid_px, dv01,
    zscore, expected_yield_pnl_bps, package_yield_bps,
    hedge_ratios, ref_idx,
    target_notional_ref, entry_z_threshold,
    tp_zscore_soft, tp_yield_soft, tp_yield_hard,
    sl_yield_hard, max_holding_bars,
    yield_pnl_multiplier, entry_cost_premium, tp_cost_premium,
    sl_cost_premium,
    tp_quarantine, sl_quarantine, time_quarantine,
    max_levels, haircut, validate_scope, initial_capital,
):
    """Numba-friendly duplicate of :func:`mr_loop`.

    The body is identical except helper calls go through the ``_jit`` versions
    so that Numba can inline them.
    """
    T = bid_px.shape[0]
    N = bid_px.shape[1]

    out_pos = np.zeros((T, N), dtype=np.float64)
    out_cash = np.zeros(T, dtype=np.float64)
    out_equity = np.zeros(T, dtype=np.float64)
    out_pnl = np.zeros(T, dtype=np.float64)
    out_codes = np.zeros(T, dtype=np.int32)
    out_state = np.zeros(T, dtype=np.int32)
    out_holding = np.zeros(T, dtype=np.int32)

    tr_bar = np.full(T, -1, dtype=np.int64)
    tr_type = np.zeros(T, dtype=np.int32)
    tr_side = np.zeros(T, dtype=np.int32)
    tr_sizes = np.zeros((T, N), dtype=np.float64)
    tr_vwaps = np.zeros((T, N), dtype=np.float64)
    tr_mids = np.zeros((T, N), dtype=np.float64)
    tr_cost = np.zeros(T, dtype=np.float64)
    tr_code = np.zeros(T, dtype=np.int32)
    tr_pkg_yield = np.zeros(T, dtype=np.float64)
    n_trades = 0

    pos = np.zeros(N, dtype=np.float64)
    cash = initial_capital
    state = 0
    cooldown = 0
    entry_bar = -1
    entry_pkg_yield = 0.0
    holding = 0
    prev_equity = initial_capital

    for t in range(T):
        code = _NO_ACTION

        if state != 0:
            pos_side = state
            holding += 1

            raw_yield_delta = package_yield_bps[t] - entry_pkg_yield
            adj_yield_delta = -pos_side * raw_yield_delta
            adj_z = -pos_side * zscore[t]

            sl_triggered = adj_yield_delta < -sl_yield_hard
            time_triggered = (max_holding_bars > 0) and (holding >= max_holding_bars)

            tp_z_cond = adj_z > tp_zscore_soft
            tp_ys_cond = adj_yield_delta > tp_yield_soft
            tp_yh_cond = adj_yield_delta > tp_yield_hard
            tp_triggered = (tp_z_cond and tp_ys_cond) or tp_yh_cond

            if sl_triggered or time_triggered:
                is_sl = sl_triggered
                all_filled = True
                exit_vwaps = np.empty(N, dtype=np.float64)
                for i in range(N):
                    if abs(pos[i]) < 1e-15:
                        exit_vwaps[i] = mid_px[t, i]
                        continue
                    qty_abs = abs(pos[i])
                    if pos[i] > 0:
                        filled, vwap = _walk_book_jit(
                            bid_px[t, i], bid_sz[t, i],
                            qty_abs, max_levels, haircut)
                    else:
                        filled, vwap = _walk_book_jit(
                            ask_px[t, i], ask_sz[t, i],
                            qty_abs, max_levels, haircut)
                    if filled < qty_abs - 1e-10:
                        all_filled = False
                        break
                    exit_vwaps[i] = vwap

                if all_filled:
                    bcost = 0.0
                    for i in range(N):
                        if abs(pos[i]) < 1e-15:
                            continue
                        cash += pos[i] * exit_vwaps[i]
                        bcost += abs(exit_vwaps[i] - mid_px[t, i]) * abs(pos[i])

                    tr_bar[n_trades] = t
                    tr_type[n_trades] = _TRADE_EXIT_SL if is_sl else _TRADE_EXIT_TIME
                    tr_side[n_trades] = pos_side
                    for i in range(N):
                        tr_sizes[n_trades, i] = -pos[i]
                        tr_vwaps[n_trades, i] = exit_vwaps[i]
                        tr_mids[n_trades, i] = mid_px[t, i]
                    tr_cost[n_trades] = bcost
                    tr_code[n_trades] = _EXIT_SL_FORCED if is_sl else _EXIT_TIME_FORCED
                    tr_pkg_yield[n_trades] = package_yield_bps[t]
                    n_trades += 1

                    for i in range(N):
                        pos[i] = 0.0
                    code = _EXIT_SL_FORCED if is_sl else _EXIT_TIME_FORCED
                    cooldown = sl_quarantine if is_sl else time_quarantine
                    state = 0
                    holding = 0
                    entry_bar = -1
                else:
                    code = _EXIT_SL_NO_LIQUIDITY if is_sl else _EXIT_TIME_NO_LIQUIDITY

            elif tp_triggered:
                valid = _check_market_valid_jit(
                    bid_px[t, :, 0], ask_px[t, :, 0], mid_px[t],
                    validate_scope, ref_idx, N)
                if not valid:
                    code = _EXIT_TP_INVALID_BOOK
                else:
                    all_filled = True
                    exit_vwaps = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        if abs(pos[i]) < 1e-15:
                            exit_vwaps[i] = mid_px[t, i]
                            continue
                        qty_abs = abs(pos[i])
                        if pos[i] > 0:
                            filled, vwap = _walk_book_jit(
                                bid_px[t, i], bid_sz[t, i],
                                qty_abs, max_levels, haircut)
                        else:
                            filled, vwap = _walk_book_jit(
                                ask_px[t, i], ask_sz[t, i],
                                qty_abs, max_levels, haircut)
                        if filled < qty_abs - 1e-10:
                            all_filled = False
                            break
                        exit_vwaps[i] = vwap

                    if not all_filled:
                        code = _EXIT_TP_NO_LIQUIDITY
                    else:
                        bcost = 0.0
                        for i in range(N):
                            bcost += abs(exit_vwaps[i] - mid_px[t, i]) * abs(pos[i])

                        ref_dv01_not = abs(pos[ref_idx]) * dv01[t, ref_idx]
                        acc_yield = (yield_pnl_multiplier
                                     * expected_yield_pnl_bps[t]
                                     - tp_cost_premium)
                        acc_cost = acc_yield * ref_dv01_not

                        if acc_cost <= 0.0 or bcost > acc_cost:
                            code = _EXIT_TP_TOO_WIDE
                        else:
                            for i in range(N):
                                if abs(pos[i]) < 1e-15:
                                    continue
                                cash += pos[i] * exit_vwaps[i]

                            tr_bar[n_trades] = t
                            tr_type[n_trades] = _TRADE_EXIT_TP
                            tr_side[n_trades] = pos_side
                            for i in range(N):
                                tr_sizes[n_trades, i] = -pos[i]
                                tr_vwaps[n_trades, i] = exit_vwaps[i]
                                tr_mids[n_trades, i] = mid_px[t, i]
                            tr_cost[n_trades] = bcost
                            tr_code[n_trades] = _EXIT_TP_OK
                            tr_pkg_yield[n_trades] = package_yield_bps[t]
                            n_trades += 1

                            for i in range(N):
                                pos[i] = 0.0
                            code = _EXIT_TP_OK
                            cooldown = tp_quarantine
                            state = 0
                            holding = 0
                            entry_bar = -1
            else:
                code = _NO_ACTION

        elif state == 0:
            if cooldown > 0:
                cooldown -= 1
                if abs(zscore[t]) > entry_z_threshold:
                    code = _ENTRY_IN_COOLDOWN
                else:
                    code = _NO_ACTION
            elif abs(zscore[t]) > entry_z_threshold:
                intended_side = 1 if zscore[t] > 0 else -1
                ref_dv01_t = dv01[t, ref_idx]
                if ref_dv01_t < 1e-15:
                    code = _ENTRY_INVALID_BOOK
                else:
                    sizes_ok = True
                    trade_sizes = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        if dv01[t, i] < 1e-15:
                            sizes_ok = False
                            break
                        trade_sizes[i] = (intended_side
                                          * target_notional_ref
                                          * ref_dv01_t
                                          * hedge_ratios[i]
                                          / dv01[t, i])
                    if not sizes_ok:
                        code = _ENTRY_INVALID_BOOK
                    else:
                        basket_dv01_ref = target_notional_ref * ref_dv01_t
                        acc_yield = (yield_pnl_multiplier
                                     * expected_yield_pnl_bps[t]
                                     - entry_cost_premium)
                        acc_cost = acc_yield * basket_dv01_ref

                        if acc_cost <= 0.0:
                            code = _ENTRY_TOO_WIDE
                        else:
                            valid = _check_market_valid_jit(
                                bid_px[t, :, 0], ask_px[t, :, 0],
                                mid_px[t], validate_scope, ref_idx, N)
                            if not valid:
                                code = _ENTRY_INVALID_BOOK
                            else:
                                fill_vwaps = np.empty(N, dtype=np.float64)
                                all_filled = True
                                for i in range(N):
                                    if abs(trade_sizes[i]) < 1e-15:
                                        fill_vwaps[i] = mid_px[t, i]
                                        continue
                                    qty_abs = abs(trade_sizes[i])
                                    if trade_sizes[i] > 0:
                                        filled, vwap = _walk_book_jit(
                                            ask_px[t, i], ask_sz[t, i],
                                            qty_abs, max_levels, haircut)
                                    else:
                                        filled, vwap = _walk_book_jit(
                                            bid_px[t, i], bid_sz[t, i],
                                            qty_abs, max_levels, haircut)
                                    if filled < qty_abs - 1e-10:
                                        all_filled = False
                                        break
                                    fill_vwaps[i] = vwap

                                if not all_filled:
                                    code = _ENTRY_NO_LIQUIDITY
                                else:
                                    bcost = 0.0
                                    for i in range(N):
                                        bcost += (abs(fill_vwaps[i] - mid_px[t, i])
                                                  * abs(trade_sizes[i]))
                                    if bcost > acc_cost:
                                        code = _ENTRY_TOO_WIDE
                                    else:
                                        for i in range(N):
                                            cash -= trade_sizes[i] * fill_vwaps[i]
                                            pos[i] += trade_sizes[i]

                                        tr_bar[n_trades] = t
                                        tr_type[n_trades] = _TRADE_ENTRY
                                        tr_side[n_trades] = intended_side
                                        for i in range(N):
                                            tr_sizes[n_trades, i] = trade_sizes[i]
                                            tr_vwaps[n_trades, i] = fill_vwaps[i]
                                            tr_mids[n_trades, i] = mid_px[t, i]
                                        tr_cost[n_trades] = bcost
                                        tr_code[n_trades] = _ENTRY_OK
                                        tr_pkg_yield[n_trades] = package_yield_bps[t]
                                        n_trades += 1

                                        code = _ENTRY_OK
                                        state = intended_side
                                        entry_bar = t
                                        entry_pkg_yield = package_yield_bps[t]
                                        holding = 0
            else:
                code = _ENTRY_NO_SIGNAL

        equity_t = cash
        for i in range(N):
            equity_t += pos[i] * mid_px[t, i]

        pnl_t = equity_t - prev_equity
        prev_equity = equity_t

        for i in range(N):
            out_pos[t, i] = pos[i]
        out_cash[t] = cash
        out_equity[t] = equity_t
        out_pnl[t] = pnl_t
        out_codes[t] = code
        out_state[t] = state
        out_holding[t] = holding

    return (
        out_pos, out_cash, out_equity, out_pnl, out_codes, out_state,
        out_holding, tr_bar, tr_type, tr_side, tr_sizes, tr_vwaps,
        tr_mids, tr_cost, tr_code, tr_pkg_yield, n_trades,
    )


mr_loop_jit = _njit(cache=True)(_mr_loop_jit_impl)
