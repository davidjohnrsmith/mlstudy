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
All signal inputs (zscore, signal_expected_pnl_yield_bps, package_yield_bps)
are assumed to be already lagged by the caller.  No internal lagging is
performed; at bar *t* the loop only reads ``signals[t]``.
"""

from __future__ import annotations

import numpy as np

from mlstudy.trading.backtest.mean_reversion.state import (
    NO_POSITION_STATES,
    State,
    ActionCode,
    TradeType,
)

# Plain-int duplicates of the IntEnum values in types.py.
#
# Numba ``@njit`` cannot resolve IntEnum members at compile time, so
# the JIT-compiled loop needs bare ints.  The canonical definitions
# live in :mod:`.types`; a test asserts that these stay in sync.


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


def _check_market_valid(
    bid0, ask0, mid, scope, ref_idx, N, is_validate_book_for_ref_only
):
    """Return True if bid0 <= mid <= ask0 (and both positive).

    scope == 0 -> check reference leg only.
    scope != 0 -> check all legs.
    """
    if scope == is_validate_book_for_ref_only:
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


# ======================================================================
# JIT-compiled version (identical logic, compiled if Numba is present)
# ======================================================================

# The helpers must be separately compiled for Numba to resolve them.
_walk_book_jit = _njit(cache=True)(_walk_book)
_check_market_valid_jit = _njit(cache=True)(_check_market_valid)


def _mr_loop_jit_impl(
    bid_px,
    bid_sz,
    ask_px,
    ask_sz,
    mid_px,
    dv01,
    zscore,
    signal_expected_pnl_yield_bps,
    package_yield_bps,
    hedge_ratios,
    ref_idx,
    target_notional_ref,
    entry_z_threshold,
    tp_zscore_soft,
    tp_yield_soft,
    tp_yield_hard,
    sl_yield_hard,
    max_holding_bars,
    signal_expected_pnl_yield_bps_multiplier,
    entry_cost_premium_yield_bps,
    tp_cost_premium,
    sl_cost_premium,
    tp_quarantine,
    sl_quarantine,
    time_quarantine,
    max_levels,
    haircut,
    validate_scope,
    initial_capital,
    is_validate_book_for_ref_only,
):
    """Core mean-reversion backtest loop.

    Iterates bar-by-bar through *T* timesteps, maintaining a three-state
    machine (FLAT / LONG / SHORT) with L2-book execution, cost gating, and
    quarantine (cooldown) periods after exits.

    This single implementation is used in two ways:

    * **Pure Python** — called directly as ``_mr_loop_jit_impl(...)`` when
      ``use_jit=False`` (or Numba is unavailable).
    * **Numba JIT** — wrapped by ``mr_loop_jit = _njit(cache=True)(_mr_loop_jit_impl)``
      at module level.  Helper calls use the ``_jit``-suffixed variants so
      Numba can inline them.

    State machine
    -------------
    ::

        FLAT ──(entry signal + liquidity + cost OK)──► LONG / SHORT
          ▲                                                │
          │   ◄──(TP / SL / time-exit + liquidity OK)──────┘
          │                                                │
          └─────────── cooldown countdown ◄────────────────┘

    Each bar the loop evaluates **exactly one** action:

    1. **In position** (``state != 0``): check stop-loss → time-exit →
       take-profit → no-action, in that priority order.
    2. **Flat** (``state == 0``): if in cooldown, decrement counter; else
       check for entry signal.

    Entry logic (flat, no cooldown)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. ``|zscore[t]| > entry_z_threshold`` — signal present.
    2. Determine ``intended_side``: +1 if zscore > 0 (cheap → go long),
       −1 if zscore < 0 (expensive → go short).
    3. Compute DV01-weighted leg sizes via hedge ratios.
    4. Validate market data (bid ≤ mid ≤ ask).
    5. Walk the L2 book (``_walk_book_jit``) to fill each leg.
    6. Compute basket execution cost; compare against the acceptable cost
       budget derived from ``signal_expected_pnl_yield_bps``.
    7. If all checks pass → execute entry, record trade, enter LONG/SHORT.

    Take-profit logic
    ~~~~~~~~~~~~~~~~~
    Direction-adjusted values::

        adj_yield_delta = -pos_side * (package_yield_bps[t] - entry_pkg_yield)
        adj_z           = -pos_side * zscore[t]

    TP fires when *either*:

    * ``adj_z > tp_zscore_soft AND adj_yield_delta > tp_yield_soft`` (soft), or
    * ``adj_yield_delta > tp_yield_hard`` (hard override).

    After signal, the exit still requires valid book, sufficient liquidity,
    and cost within budget (otherwise → ``EXIT_TP_INVALID_BOOK``,
    ``EXIT_TP_NO_LIQUIDITY``, or ``EXIT_TP_TOO_WIDE``).

    Stop-loss / time-exit logic
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * SL fires when ``adj_yield_delta < -sl_yield_hard`` (adverse move).
    * Time-exit fires when ``holding >= max_holding_bars`` (if enabled).
    * Both are **forced**: they attempt to fill but skip the cost gate.
      If liquidity is insufficient → ``EXIT_SL_NO_LIQUIDITY`` /
      ``EXIT_TIME_NO_LIQUIDITY`` (position kept, retried next bar).

    Parameters
    ----------
    bid_px : ndarray, shape (T, N, L)
        Bid prices per bar × instrument × book level (level 0 = best).
    bid_sz : ndarray, shape (T, N, L)
        Bid sizes.
    ask_px : ndarray, shape (T, N, L)
        Ask prices.
    ask_sz : ndarray, shape (T, N, L)
        Ask sizes.
    mid_px : ndarray, shape (T, N)
        Mid prices (independent of book; used for MTM and cost calc).
    dv01 : ndarray, shape (T, N)
        Price change per 1 bp yield move per unit par notional, per
        instrument.  Used to derive leg sizes from hedge ratios.
    zscore : ndarray, shape (T,)
        Pre-computed, *already lagged* z-score signal.  Positive = package
        is cheap (enter long); negative = expensive (enter short).
    signal_expected_pnl_yield_bps : ndarray, shape (T,)
        Expected yield P&L magnitude in bps (non-negative).  Scaled by
        ``signal_expected_pnl_yield_bps_multiplier`` to set the cost budget.
    package_yield_bps : ndarray, shape (T,)
        Package yield level in bps.  Tracked from entry to compute
        direction-adjusted yield deltas for TP / SL checks.
    hedge_ratios : ndarray, shape (N,)
        Yield-space hedge ratios.  ``hedge_ratios[ref_idx] == 1.0`` and
        ``sum(hedge_ratios) == 0`` (DV01-neutral).
    ref_idx : int
        Index of the reference leg in the instrument dimension.
    target_notional_ref : float
        Par notional for the reference leg.  Other legs are sized so that
        ``size_i = target_notional_ref * dv01[t, ref] * hedge_ratios[i] / dv01[t, i]``.
    entry_z_threshold : float
        Minimum ``|zscore[t]|`` to consider entering.
    tp_zscore_soft : float
        Take-profit soft z-score threshold (direction-adjusted).
    tp_yield_soft : float
        Take-profit soft yield-change threshold (bps, direction-adjusted).
    tp_yield_hard : float
        Take-profit hard yield-change threshold (bps); triggers TP
        unconditionally on yield alone.
    sl_yield_hard : float
        Stop-loss yield-change threshold (bps, direction-adjusted).
        SL fires when ``adj_yield_delta < -sl_yield_hard``.
    max_holding_bars : int
        Force exit after this many bars in position.  0 = disabled.
    signal_expected_pnl_yield_bps_multiplier : float
        Scales ``signal_expected_pnl_yield_bps`` when computing the acceptable
        cost budget.
    entry_cost_premium_yield_bps : float
        Yield bps subtracted from the entry cost budget
        (tighter budget → fewer entries).
    tp_cost_premium : float
        Yield bps subtracted from the take-profit cost budget.
    sl_cost_premium : float
        Yield bps subtracted from the stop-loss cost budget (unused for
        forced SL which skips the cost gate).
    tp_quarantine : int
        Bars of cooldown after a take-profit exit before re-entry is
        allowed.
    sl_quarantine : int
        Bars of cooldown after a stop-loss exit.
    time_quarantine : int
        Bars of cooldown after a time-exit.
    max_levels : int
        Maximum L2 book levels to walk when filling an order.
    haircut : float
        Fraction of displayed book size actually usable (0–1).
    validate_scope : int
        ``0`` (``VALIDATE_REF_ONLY``) checks only the reference leg;
        non-zero checks all N legs for ``bid ≤ mid ≤ ask``.
    initial_capital : float
        Starting cash balance.

    Returns
    -------
    tuple of 17 elements
        Per-bar arrays (length *T*):

        0. ``out_pos``     — (T, N) float64, leg positions at end of bar.
        1. ``out_cash``    — (T,) float64, cash balance.
        2. ``out_equity``  — (T,) float64, total equity (cash + MTM).
        3. ``out_pnl``     — (T,) float64, bar-over-bar P&L.
        4. ``out_codes``   — (T,) int32, attempt/outcome code (see ``types.py``).
        5. ``out_state``   — (T,) int32, state at end of bar (0/+1/−1).
        6. ``out_holding`` — (T,) int32, bars held in current position.

        Per-trade arrays (pre-allocated to length *T*, valid for ``[:n_trades]``):

        7.  ``tr_bar``       — bar index of the trade.
        8.  ``tr_type``      — trade type (0 entry, 1 TP, 2 SL, 3 time).
        9.  ``tr_side``      — +1 long / −1 short.
        10. ``tr_sizes``     — (T, N) signed leg sizes.
        11. ``tr_vwaps``     — (T, N) fill VWAPs.
        12. ``tr_mids``      — (T, N) mid prices at fill time.
        13. ``tr_cost``      — basket execution cost (≥ 0).
        14. ``tr_code``      — int32 outcome code for the trade.
        15. ``tr_pkg_yield`` — package_yield_bps at time of trade.

        Scalar:

        16. ``n_trades`` — number of valid rows in the trade arrays.
    """
    T = bid_px.shape[0]  # number of bars
    N = bid_px.shape[1]  # number of instruments (legs)

    # -- Per-bar output arrays (written once per bar at the bottom of the loop) --
    out_pos = np.zeros((T, N), dtype=np.float64)
    out_cash = np.zeros(T, dtype=np.float64)
    out_equity = np.zeros(T, dtype=np.float64)
    out_pnl = np.zeros(T, dtype=np.float64)
    out_codes = np.zeros(T, dtype=np.int32)
    out_state = np.zeros(T, dtype=np.int32)
    out_holding = np.zeros(T, dtype=np.int32)

    # -- Per-trade output arrays (pre-allocated to max possible = T trades) --
    # Only the first n_trades rows are valid after the loop completes.
    tr_bar = np.full(T, -1, dtype=np.int64)
    tr_type = np.zeros(T, dtype=np.int32)
    tr_side = np.zeros(T, dtype=np.int32)
    tr_sizes = np.zeros((T, N), dtype=np.float64)
    tr_risks = np.zeros((T, N), dtype=np.float64)
    tr_vwaps = np.zeros((T, N), dtype=np.float64)
    tr_mids = np.zeros((T, N), dtype=np.float64)
    tr_cost = np.zeros(T, dtype=np.float64)
    tr_code = np.zeros(T, dtype=np.int32)
    tr_pkg_yield = np.zeros(T, dtype=np.float64)
    n_trades = 0

    # -- Mutable state carried across bars --
    pos = np.zeros(N, dtype=np.float64)  # current signed position per leg
    cash = initial_capital  # running cash balance
    state = State.STATE_FLAT.value
    cooldown = 0  # bars remaining before re-entry is allowed
    entry_bar = -1  # bar index of the most recent entry (-1 = none)
    entry_pkg_yield = 0.0  # package_yield_bps at entry (for TP/SL delta)
    holding = 0  # bars held in current position (reset on entry/exit)
    prev_equity = initial_capital  # previous bar's equity (for PnL diff)

    for t in range(T):
        code = ActionCode.NO_ACTION.value

        # ==================================================================
        # BRANCH A: IN POSITION (state != 0)
        # Priority: stop-loss / time-exit  ▸  take-profit  ▸  no action
        # ==================================================================
        if state == State.STATE_LONG.value or state == State.STATE_SHORT.value:
            pos_side = state  # +1 (LONG) or -1 (SHORT)
            holding += 1

            # -- Direction-adjusted deltas for TP / SL evaluation --
            # raw_yield_delta: how much the package yield moved since entry.
            #   Positive = yield rose; negative = yield fell.
            raw_yield_delta = package_yield_bps[t] - entry_pkg_yield

            # adj_yield_delta: flips sign so that *positive* = profitable.
            #   LONG profits when yield *falls*  → -(+1)*(neg) = pos  ✓
            #   SHORT profits when yield *rises* → -(-1)*(pos) = pos  ✓
            adj_yield_delta = -pos_side * raw_yield_delta

            # adj_z: flips sign so that *positive* = z-score reverted toward 0.
            #   LONG entered on z>0; reversion means z falling → -(+1)*(neg) = pos  ✓
            #   SHORT entered on z<0; reversion means z rising → -(-1)*(pos) = pos  ✓
            adj_z = -pos_side * zscore[t]

            # -- Check forced-exit triggers (SL and time) --
            # SL: the position has lost more yield bps than the hard threshold.
            sl_triggered = adj_yield_delta < -sl_yield_hard
            # Time: held too long (0 = disabled).
            time_triggered = (max_holding_bars > 0) and (holding >= max_holding_bars)

            # -- Check take-profit triggers --
            # Soft TP: both z-score reverted AND yield moved favourably.
            tp_z_cond = adj_z > tp_zscore_soft
            tp_ys_cond = adj_yield_delta > tp_yield_soft
            # Hard TP: yield profit alone exceeds the hard threshold.
            tp_yh_cond = adj_yield_delta > tp_yield_hard
            tp_triggered = (tp_z_cond and tp_ys_cond) or tp_yh_cond

            # ---- SL / time-exit (forced – no cost gate) ----
            if sl_triggered or time_triggered:
                is_sl = sl_triggered  # SL takes priority when both fire
                # Try to fill the exit by walking the book for each leg.
                # Longs sell into the bid; shorts buy from the ask.
                all_filled = True
                exit_vwaps = np.empty(N, dtype=np.float64)
                for i in range(N):
                    if abs(pos[i]) < 1e-15:
                        # Zero-size leg: no fill needed, use mid for record.
                        exit_vwaps[i] = mid_px[t, i]
                        continue
                    qty_abs = abs(pos[i])
                    if pos[i] > 0:
                        # Long leg → sell into the bid side.
                        filled, vwap = _walk_book_jit(
                            bid_px[t, i], bid_sz[t, i], qty_abs, max_levels, haircut
                        )
                    else:
                        # Short leg → buy from the ask side.
                        filled, vwap = _walk_book_jit(
                            ask_px[t, i], ask_sz[t, i], qty_abs, max_levels, haircut
                        )
                    if filled < qty_abs - 1e-10:
                        # Not enough liquidity to close this leg; abort
                        # the entire exit and retry next bar.
                        all_filled = False
                        break
                    exit_vwaps[i] = vwap

                if all_filled:
                    # -- Execute the forced exit --
                    # Settle cash: for each leg, cash += pos[i] * vwap[i].
                    #   Long leg (pos>0) sold at vwap → cash increases.
                    #   Short leg (pos<0) bought at vwap → cash decreases.
                    # Basket cost: sum of |vwap - mid| * |size| across legs
                    #   (always ≥ 0, measures total market-impact / spread cost).
                    bcost = 0.0
                    for i in range(N):
                        if abs(pos[i]) < 1e-15:
                            continue
                        cash += pos[i] * exit_vwaps[i]
                        bcost += abs(exit_vwaps[i] - mid_px[t, i]) * abs(pos[i])

                    # Record the trade (sizes are negated: closing the position).
                    tr_bar[n_trades] = t
                    tr_type[n_trades] = (
                        TradeType.TRADE_EXIT_SL.value
                        if is_sl
                        else TradeType.TRADE_EXIT_TIME.value
                    )
                    tr_side[n_trades] = pos_side
                    for i in range(N):
                        tr_sizes[n_trades, i] = -pos[i]
                        tr_risks[n_trades, i] = -pos[i] * dv01[t, i]
                        tr_vwaps[n_trades, i] = exit_vwaps[i]
                        tr_mids[n_trades, i] = mid_px[t, i]
                    tr_cost[n_trades] = bcost
                    tr_code[n_trades] = (
                        ActionCode.EXIT_SL_OK.value
                        if is_sl
                        else ActionCode.EXIT_TIME_OK.value
                    )
                    tr_pkg_yield[n_trades] = package_yield_bps[t]
                    n_trades += 1

                    # Reset to FLAT with post-exit quarantine.
                    for i in range(N):
                        pos[i] = 0.0
                    code = (
                        ActionCode.EXIT_SL_OK.value
                        if is_sl
                        else ActionCode.EXIT_TIME_OK.value
                    )
                    cooldown = sl_quarantine if is_sl else time_quarantine
                    if cooldown > 0:
                        state = (
                            State.STATE_SL_COOLDOWN.value
                            if is_sl
                            else State.STATE_TIME_COOLDOWN.value
                        )
                    else:
                        state = State.STATE_FLAT.value
                    holding = 0
                    entry_bar = -1
                else:
                    # Liquidity insufficient — position kept, will retry.
                    code = (
                        ActionCode.EXIT_FAILED_SL_NO_LIQUIDITY.value
                        if is_sl
                        else ActionCode.EXIT_FAILED_TIME_NO_LIQUIDITY.value
                    )

            # ---- Take-profit (requires valid book, liquidity, and cost gate) ----
            elif tp_triggered:
                # Gate 1: market validity — bid ≤ mid ≤ ask on required legs.
                valid = _check_market_valid_jit(
                    bid_px[t, :, 0],
                    ask_px[t, :, 0],
                    mid_px[t],
                    validate_scope,
                    ref_idx,
                    N,
                    is_validate_book_for_ref_only,
                )
                if not valid:
                    code = ActionCode.EXIT_FAILED_TP_NO_LIQUIDITY.value
                else:
                    # Gate 2: liquidity — walk book to fill each leg.
                    all_filled = True
                    exit_vwaps = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        if abs(pos[i]) < 1e-15:
                            exit_vwaps[i] = mid_px[t, i]
                            continue
                        qty_abs = abs(pos[i])
                        if pos[i] > 0:
                            filled, vwap = _walk_book_jit(
                                bid_px[t, i], bid_sz[t, i], qty_abs, max_levels, haircut
                            )
                        else:
                            filled, vwap = _walk_book_jit(
                                ask_px[t, i], ask_sz[t, i], qty_abs, max_levels, haircut
                            )
                        if filled < qty_abs - 1e-10:
                            all_filled = False
                            break
                        exit_vwaps[i] = vwap

                    if not all_filled:
                        code = ActionCode.EXIT_FAILED_TP_NO_LIQUIDITY.value
                    else:
                        # Gate 3: cost check — basket cost must be within the
                        # acceptable budget derived from expected yield PnL.
                        # bcost = Σ |vwap_i - mid_i| * |pos_i|
                        bcost = 0.0
                        for i in range(N):
                            bcost += abs(exit_vwaps[i] - mid_px[t, i]) * abs(pos[i])

                        # acc_cost = acceptable cost in price units.
                        # ref_dv01_not = DV01 exposure of the ref leg in
                        #   price units per bp: |pos_ref| * dv01_ref.
                        # acc_yield = yield budget in bps after subtracting
                        #   the TP cost premium.
                        # acc_cost = acc_yield * ref_dv01_not  converts
                        #   from yield-bps space to price-unit space.
                        ref_dv01_not = abs(pos[ref_idx]) * dv01[t, ref_idx]
                        acc_yield = (
                            signal_expected_pnl_yield_bps_multiplier
                            * signal_expected_pnl_yield_bps[t]
                            - tp_cost_premium
                        )
                        acc_cost = acc_yield * ref_dv01_not

                        if acc_cost <= 0.0 or bcost > acc_cost:
                            code = ActionCode.EXIT_FAILED_TP_TOO_WIDE.value
                        else:
                            # -- Execute the TP exit --
                            # Settle cash for each leg.
                            for i in range(N):
                                if abs(pos[i]) < 1e-15:
                                    continue
                                cash += pos[i] * exit_vwaps[i]

                            # Record the trade.
                            tr_bar[n_trades] = t
                            tr_type[n_trades] = TradeType.TRADE_EXIT_TP.value
                            tr_side[n_trades] = pos_side
                            for i in range(N):
                                tr_sizes[n_trades, i] = -pos[i]
                                tr_risks[n_trades, i] = -pos[i] * dv01[t, i]
                                tr_vwaps[n_trades, i] = exit_vwaps[i]
                                tr_mids[n_trades, i] = mid_px[t, i]
                            tr_cost[n_trades] = bcost
                            tr_code[n_trades] = ActionCode.EXIT_TP_OK.value
                            tr_pkg_yield[n_trades] = package_yield_bps[t]
                            n_trades += 1

                            # Reset to FLAT with TP quarantine.
                            for i in range(N):
                                pos[i] = 0.0
                            code = ActionCode.EXIT_TP_OK.value
                            cooldown = tp_quarantine
                            if cooldown > 0:
                                state = State.STATE_TP_COOLDOWN.value
                            else:
                                state = State.STATE_FLAT.value
                            holding = 0
                            entry_bar = -1
            else:
                # In position but no exit trigger — hold.
                code = ActionCode.NO_ACTION_HOLD.value

        # ==================================================================
        # BRANCH B: FLAT (state == 0) — cooldown or entry attempt
        # ==================================================================
        elif state in NO_POSITION_STATES:
            if cooldown > 0:
                # Still in post-exit quarantine; decrement and skip.
                cooldown -= 1
                if abs(zscore[t]) > entry_z_threshold:
                    # Signal present but blocked by cooldown.
                    code = ActionCode.ENTRY_FAILED_IN_COOLDOWN.value
                else:
                    code = ActionCode.NO_ACTION_NO_SIGNAL.value

            elif abs(zscore[t]) > entry_z_threshold:
                # --- Entry attempt ---
                # Step 1: determine direction.
                #   zscore > 0 → package is cheap → go LONG (+1).
                #   zscore < 0 → package is expensive → go SHORT (-1).
                intended_side = 1 if zscore[t] > 0 else -1

                # Step 2: compute DV01-based leg sizes.
                # ref_dv01_t = price change of the ref leg per 1 bp yield
                #   move per 1 unit of par notional.
                ref_dv01_t = dv01[t, ref_idx]
                if ref_dv01_t < 1e-15:
                    # Ref leg DV01 is zero/missing — cannot size the basket.
                    code = ActionCode.ENTRY_FAILED_INVALID_DV01.value
                else:
                    sizes_ok = True
                    trade_sizes = np.empty(N, dtype=np.float64)
                    trade_risks = np.empty(N, dtype=np.float64)
                    for i in range(N):
                        if dv01[t, i] < 1e-15:
                            # Any leg with zero DV01 → cannot compute size.
                            sizes_ok = False
                            break
                        # size_i = side * notional_ref * dv01_ref * hr_i / dv01_i
                        # This ensures each leg's DV01 contribution matches
                        # the intended hedge ratio in yield space.
                        trade_sizes[i] = (
                            intended_side
                            * target_notional_ref
                            * ref_dv01_t
                            * hedge_ratios[i]
                            / dv01[t, i]
                        )
                        trade_risks[i] = trade_sizes[i] * dv01[t, i]
                    if not sizes_ok:
                        code = ActionCode.ENTRY_FAILED_INVALID_DV01.value
                    else:
                        # Step 3: compute the acceptable cost budget.
                        # basket_dv01_ref = total ref-leg DV01 exposure in
                        #   price units: notional * dv01_ref.
                        # acc_yield = yield budget (bps) after premium.
                        # acc_cost  = budget converted to price units.
                        basket_dv01_ref = target_notional_ref * ref_dv01_t
                        acc_yield = (
                            signal_expected_pnl_yield_bps_multiplier
                            * signal_expected_pnl_yield_bps[t]
                            - entry_cost_premium_yield_bps
                        )
                        acc_cost = acc_yield * basket_dv01_ref

                        if acc_cost <= 0.0:
                            # Budget is non-positive → too expensive to enter.
                            code = ActionCode.ENTRY_FAILED_TOO_WIDE.value
                        else:
                            # Step 4: validate market data (bid ≤ mid ≤ ask).
                            valid = _check_market_valid_jit(
                                bid_px[t, :, 0],
                                ask_px[t, :, 0],
                                mid_px[t],
                                validate_scope,
                                ref_idx,
                                N,
                                is_validate_book_for_ref_only,
                            )
                            if not valid:
                                code = ActionCode.ENTRY_FAILED_INVALID_BOOK.value
                            else:
                                # Step 5: walk the L2 book to fill each leg.
                                # Buys lift the ask; sells hit the bid.
                                fill_vwaps = np.empty(N, dtype=np.float64)
                                all_filled = True
                                for i in range(N):
                                    if abs(trade_sizes[i]) < 1e-15:
                                        fill_vwaps[i] = mid_px[t, i]
                                        continue
                                    qty_abs = abs(trade_sizes[i])
                                    if trade_sizes[i] > 0:
                                        # Buying → lift the ask.
                                        filled, vwap = _walk_book_jit(
                                            ask_px[t, i],
                                            ask_sz[t, i],
                                            qty_abs,
                                            max_levels,
                                            haircut,
                                        )
                                    else:
                                        # Selling → hit the bid.
                                        filled, vwap = _walk_book_jit(
                                            bid_px[t, i],
                                            bid_sz[t, i],
                                            qty_abs,
                                            max_levels,
                                            haircut,
                                        )
                                    if filled < qty_abs - 1e-10:
                                        all_filled = False
                                        break
                                    fill_vwaps[i] = vwap

                                if not all_filled:
                                    code = ActionCode.ENTRY_FAILED_NO_LIQUIDITY.value
                                else:
                                    # Step 6: cost gate.
                                    # bcost = Σ |vwap_i - mid_i| * |size_i|
                                    # Must not exceed acc_cost.
                                    bcost = 0.0
                                    for i in range(N):
                                        bcost += abs(
                                            fill_vwaps[i] - mid_px[t, i]
                                        ) * abs(trade_sizes[i])
                                    if bcost > acc_cost:
                                        code = ActionCode.ENTRY_FAILED_TOO_WIDE.value
                                    else:
                                        # Step 7: execute entry.
                                        # Pay cash for buys, receive cash for
                                        # sells: cash -= size_i * vwap_i.
                                        # Accumulate position.
                                        for i in range(N):
                                            cash -= trade_sizes[i] * fill_vwaps[i]
                                            pos[i] += trade_sizes[i]

                                        # Record the entry trade.
                                        tr_bar[n_trades] = t
                                        tr_type[n_trades] = TradeType.TRADE_ENTRY.value
                                        tr_side[n_trades] = intended_side
                                        for i in range(N):
                                            tr_sizes[n_trades, i] = trade_sizes[i]
                                            tr_risks[n_trades, i] = trade_risks[i]
                                            tr_vwaps[n_trades, i] = fill_vwaps[i]
                                            tr_mids[n_trades, i] = mid_px[t, i]
                                        tr_cost[n_trades] = bcost
                                        tr_code[n_trades] = ActionCode.ENTRY_OK.value
                                        tr_pkg_yield[n_trades] = package_yield_bps[t]
                                        n_trades += 1

                                        # Transition FLAT → LONG/SHORT.
                                        code = ActionCode.ENTRY_OK.value
                                        state = intended_side
                                        entry_bar = t
                                        entry_pkg_yield = package_yield_bps[t]
                                        holding = 0
            else:
                # No signal — nothing to do.
                code = ActionCode.NO_ACTION_NO_SIGNAL.value

        # ==================================================================
        # End-of-bar bookkeeping: MTM equity, PnL, snapshot to output arrays
        # ==================================================================
        # equity = cash + Σ pos_i * mid_i  (mark-to-market)
        equity_t = cash
        for i in range(N):
            equity_t += pos[i] * mid_px[t, i]

        # Bar PnL = change in equity from the previous bar.
        pnl_t = equity_t - prev_equity
        prev_equity = equity_t

        # Write this bar's snapshot to the output arrays.
        for i in range(N):
            out_pos[t, i] = pos[i]
        out_cash[t] = cash
        out_equity[t] = equity_t
        out_pnl[t] = pnl_t
        out_codes[t] = code
        out_state[t] = state
        out_holding[t] = holding

    return (
        out_pos,
        out_cash,
        out_equity,
        out_pnl,
        out_codes,
        out_state,
        out_holding,
        tr_bar,
        tr_type,
        tr_side,
        tr_sizes,
        tr_risks,
        tr_vwaps,
        tr_mids,
        tr_cost,
        tr_code,
        tr_pkg_yield,
        n_trades,
    )


mr_loop_jit = _njit(cache=True)(_mr_loop_jit_impl)
