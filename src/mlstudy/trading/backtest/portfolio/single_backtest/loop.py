"""
LP-based portfolio backtest core loop.

Implements a bar-by-bar backtest for a multi-instrument portfolio strategy where
trade candidates are generated from two signal-gated fair prices (risk-
increasing vs. risk-decreasing) and an LP maximises executable alpha subject
to DV01 / position / bucket constraints.

Sign conventions
----------------
- pos > 0 → long; pos < 0 → short  (per instrument, in par notional units)
- side +1 = BUY, -1 = SELL
- alpha_bps > 0 → profitable trade opportunity

Fair price gating
-----------------
Two fair prices per instrument:

  fair_inc  active when |zscore| > z_inc AND adf_p_value < p_inc  (stricter)
  fair_dec  active when |zscore| > z_dec AND adf_p_value < p_dec  (looser)

Risk classification:
  BUY  is risk-decreasing if current pos < 0, else risk-increasing
  SELL is risk-decreasing if current pos > 0, else risk-increasing

Risk-increasing trades must use fair_inc; risk-decreasing use fair_dec.

Executable alpha (bps):
  BUY:  fair - ask   (must be > alpha_thr_inc or alpha_thr_dec)
  SELL: bid - fair    (must be > alpha_thr_inc or alpha_thr_dec)

LP formulation
--------------
Maximise Σ alpha_i · dv01_trade_i  subject to:
  - 0 ≤ dv01_trade_i ≤ dv01_liq_cap_i   (per candidate)
  - Σ dv01_trade_i ≤ gross_dv01_cap       (total)
  - per-instrument position bounds          (long / short limits)
  - optional issuer / maturity bucket caps

L2 book execution
-----------------
After LP, each selected trade is executed via _walk_book on the appropriate
book side.  Partial fills are allowed; the filled amount updates pos/cash.

No lookahead
------------
At bar t only signals[t], market[t], risk[t] are used.  No internal lagging
is performed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoopState:
    """Mutable backtest state that carries across chunks."""
    pos: np.ndarray           # (B,)
    hedge_pos: np.ndarray     # (H,)
    cash: float
    cooldown_remaining: int
    prev_equity: float
    prev_hedge_mtm: float
    cum_instrument_cash_mid: float = 0.0
    cum_hedge_cash_mid: float = 0.0

from mlstudy.trading.backtest.portfolio.single_backtest.state import (
    PortfolioActionCode,
    TradeCode,
    CooldownMode,
    FairType,
)

# Plain-int duplicates for potential Numba compatibility.
_NO_ACTION = int(PortfolioActionCode.NO_ACTION)
_NO_CANDIDATES = int(PortfolioActionCode.NO_CANDIDATES)
_SKIP_COOLDOWN = int(PortfolioActionCode.SKIP_COOLDOWN)
_SKIP_COOLDOWN_RISK_ONLY = int(PortfolioActionCode.SKIP_COOLDOWN_RISK_ONLY)
_EXEC_OK = int(PortfolioActionCode.EXEC_OK)
_EXEC_PARTIAL = int(PortfolioActionCode.EXEC_PARTIAL)
_EXEC_NO_LIQUIDITY = int(PortfolioActionCode.EXEC_NO_LIQUIDITY)
_EXEC_GREEDY = int(PortfolioActionCode.EXEC_GREEDY)
_LP_INFEASIBLE = int(PortfolioActionCode.LP_INFEASIBLE)
_LP_NO_CANDIDATES = int(PortfolioActionCode.LP_NO_CANDIDATES)
_INVALID_BOOK = int(PortfolioActionCode.INVALID_BOOK)
_INVALID_DV01 = int(PortfolioActionCode.INVALID_DV01)

_FILL_OK = int(TradeCode.FILL_OK)
_FILL_PARTIAL = int(TradeCode.FILL_PARTIAL)
_FILL_FAILED_LIQUIDITY = int(TradeCode.FILL_FAILED_LIQUIDITY)
_FILL_FAILED_BOOK = int(TradeCode.FILL_FAILED_BOOK)
_FILL_BELOW_MIN = int(TradeCode.FILL_BELOW_MIN)

_COOLDOWN_BLOCK_ALL = int(CooldownMode.BLOCK_ALL)
_COOLDOWN_RISK_REDUCING = int(CooldownMode.RISK_REDUCING)

_FAIR_DEC = int(FairType.DEC)
_FAIR_INC = int(FairType.INC)


from scipy.optimize import linprog as _linprog


# ======================================================================
# Helpers
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


def _check_market_valid(bid0, ask0, mid):
    """Return True if bid0 <= mid <= ask0 and all positive for one instrument."""
    if bid0 <= 0.0 or ask0 <= 0.0 or mid <= 0.0:
        return False
    if bid0 > mid or mid > ask0:
        return False
    return True


def _round_qty_trade(qty, min_qty_trade, qty_step):
    """Round a trade size: zero out if below min, else snap to step."""
    if abs(qty) < min_qty_trade:
        return 0.0
    if qty_step > 1e-15:
        return round(qty / qty_step) * qty_step
    return qty


# ======================================================================
# LP / greedy solver
# ======================================================================


def _solve_lp(
    alphas,           # (K,) alpha_bps per candidate (positive)
    dv01_liq_caps,    # (K,) max dv01 fillable per candidate
    gross_dv01_cap,   # scalar: total dv01 budget
    pos_headroom,     # (K,) max dv01 increase allowed per candidate (position limits)
    inst_indices,     # (K,) int bond index for each candidate
    sides,            # (K,) +1 buy / -1 sell
    issuer_bucket,    # (B,) or None — issuer bucket per instrument
    maturity_bucket,  # (B,) or None — maturity bucket per instrument
    issuer_dv01_caps, # (n_issuers,) or None
    mat_bucket_dv01_caps,  # (n_buckets,) or None
    current_issuer_dv01,   # (n_issuers,) current signed dv01 per issuer
    current_mat_dv01,      # (n_buckets,) current signed dv01 per maturity bucket
):
    """Solve for optimal dv01 trade sizes.

    Returns (dv01_sizes, used_greedy) where dv01_sizes is (K,) non-negative
    and used_greedy is True if the greedy fallback was used.
    """
    K = len(alphas)
    if K == 0:
        return np.zeros(0, dtype=np.float64), False

    # Upper bounds per candidate: min of liquidity cap and position headroom
    ub = np.minimum(dv01_liq_caps, pos_headroom)
    ub = np.maximum(ub, 0.0)

    # Minimise -alpha·x  (maximise alpha·x)
    c = -alphas.copy()

    # Bounds: 0 <= x_i <= ub_i
    bounds = [(0.0, float(ub[i])) for i in range(K)]

    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []

    # Gross DV01 cap: sum(x) <= gross_dv01_cap
    A_rows.append(np.ones(K, dtype=np.float64))
    b_rows.append(gross_dv01_cap)

    # Issuer caps
    if (issuer_bucket is not None and issuer_dv01_caps is not None
            and len(issuer_dv01_caps) > 0):
        n_issuers = len(issuer_dv01_caps)
        for iss in range(n_issuers):
            row = np.zeros(K, dtype=np.float64)
            for k in range(K):
                b_idx = inst_indices[k]
                if issuer_bucket[b_idx] == iss:
                    row[k] = sides[k]  # signed: buy adds, sell subtracts
            cap = issuer_dv01_caps[iss]
            # current + row·x <= cap  =>  row·x <= cap - current
            A_rows.append(row)
            b_rows.append(cap - current_issuer_dv01[iss])
            # Also: current + row·x >= -cap  =>  -row·x <= cap + current
            A_rows.append(-row)
            b_rows.append(cap + current_issuer_dv01[iss])

    # Maturity bucket caps
    if (maturity_bucket is not None and mat_bucket_dv01_caps is not None
            and len(mat_bucket_dv01_caps) > 0):
        n_buckets = len(mat_bucket_dv01_caps)
        for bkt in range(n_buckets):
            row = np.zeros(K, dtype=np.float64)
            for k in range(K):
                b_idx = inst_indices[k]
                if maturity_bucket[b_idx] == bkt:
                    row[k] = sides[k]
            cap = mat_bucket_dv01_caps[bkt]
            A_rows.append(row)
            b_rows.append(cap - current_mat_dv01[bkt])
            A_rows.append(-row)
            b_rows.append(cap + current_mat_dv01[bkt])

    A_ub = np.array(A_rows, dtype=np.float64) if A_rows else None
    b_ub = np.array(b_rows, dtype=np.float64) if b_rows else None

    try:
        res = _linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                       method="highs")
        if res.success:
            return np.maximum(res.x, 0.0), False
    except Exception:
        pass

    # Greedy fallback: allocate by descending alpha, respecting caps
    order = np.argsort(-alphas)
    dv01_sizes = np.zeros(K, dtype=np.float64)
    remaining_budget = gross_dv01_cap

    for idx in order:
        if remaining_budget <= 1e-15:
            break
        alloc = min(ub[idx], remaining_budget)
        if alloc <= 1e-15:
            continue
        dv01_sizes[idx] = alloc
        remaining_budget -= alloc

    return dv01_sizes, True


# ======================================================================
# Core loop
# ======================================================================


def lp_portfolio_loop(
    # -- Market L2 --
    bid_px,             # (T, B, L) bid prices
    bid_sz,             # (T, B, L) bid sizes
    ask_px,             # (T, B, L) ask prices
    ask_sz,             # (T, B, L) ask sizes
    mid_px,             # (T, B)    mid prices
    # -- Risk --
    dv01,               # (T, B)    dv01 per unit notional
    # -- Signals --
    fair_price,         # (T, B)    predicted fair mid
    zscore,             # (T, B)    z-score per instrument
    adf_p_value,        # (T, B)    ADF p-value per instrument
    # -- Static meta --
    tradable,           # (B,)      bool/int tradable mask
    pos_limits_long,    # (B,)      max long notional per instrument
    pos_limits_short,   # (B,)      max short notional per instrument (negative)
    max_trade_notional_inc,  # (B,) max notional per risk-increasing trade
    max_trade_notional_dec,  # (B,) max notional per risk-decreasing trade
    # -- Optional meta (can be None) --
    maturity,           # (T, B) or (B,) or None  years to maturity
    issuer_bucket,      # (B,) or None  int issuer label
    maturity_bucket,    # (T, B) or (B,) or None  int maturity bucket label
    # -- LP constraint params --
    gross_dv01_cap,     # float     total gross dv01 budget per bar
    issuer_dv01_caps,   # (n_issuers,) or None
    mat_bucket_dv01_caps,  # (n_buckets,) or None
    top_k,              # int       max candidates to send to LP
    # -- Signal gating thresholds --
    z_inc,              # float     |zscore| threshold for risk-increasing fair
    p_inc,              # float     adf p-value threshold for risk-increasing
    z_dec,              # float     |zscore| threshold for risk-decreasing fair
    p_dec,              # float     adf p-value threshold for risk-decreasing
    # -- Alpha thresholds (bps) --
    alpha_thr_inc,      # float     min alpha for risk-increasing trades
    alpha_thr_dec,      # float     min alpha for risk-decreasing trades
    # -- Execution params --
    max_levels,         # int       L2 levels to walk
    haircut,            # float     fraction of displayed size usable
    qty_step,          # (B,)      per-instrument notional rounding step
    min_qty_trade,     # float     notional below this → skip
    min_fill_ratio,     # float     if filled/requested < this → skip (0 to disable)
    # -- Cooldown --
    cooldown_bars,      # int       bars of cooldown after trading
    cooldown_mode,      # int       0 = block all, 1 = allow risk-reducing
    # -- Maturity filter (optional, 0.0 to disable) --
    min_maturity_inc,   # float     min years-to-maturity for risk-increasing
    # -- Capital --
    initial_capital,    # float
    # -- Hedge market L2 --
    hedge_bid_px=None,      # (T, H, L) or None
    hedge_bid_sz=None,      # (T, H, L) or None
    hedge_ask_px=None,      # (T, H, L) or None
    hedge_ask_sz=None,      # (T, H, L) or None
    hedge_mid_px=None,      # (T, H)    or None
    # -- Hedge risk --
    hedge_dv01=None,        # (T, H)    or None
    # -- Hedge ratios --
    hedge_ratios=None,      # (T, B, H) or None
    # -- Hedge execution --
    hedge_qty_step=None,    # (H,) or None — per-hedge notional rounding step
    # -- Chunked state --
    initial_state=None,     # LoopState or None — resume from this state
    return_final_state=False,  # bool — if True, append LoopState to return
):
    """LP-based portfolio backtest core loop.

    Iterates bar-by-bar through *T* timesteps.  At each bar:

    1. Gate fair prices by (zscore, adf_p_value) thresholds.
    2. Classify each instrument's trade direction as risk-increasing or -decreasing.
    3. Compute executable alpha; filter by thresholds and eligibility.
    4. Rank by alpha, keep top K candidates.
    5. Solve LP (or greedy fallback) for optimal DV01 allocation.
    6. Round, then execute via L2 book walking.
    7. Update positions, cash, equity.

    Parameters
    ----------
    bid_px, bid_sz, ask_px, ask_sz : (T, B, L)
        L2 order book arrays.
    mid_px : (T, B)
        Mid prices for MTM and cost calculation.
    dv01 : (T, B)
        DV01 per unit par notional per instrument.
    fair_price : (T, B)
        Model fair price per instrument.
    zscore : (T, B)
        Z-score signal per instrument.
    adf_p_value : (T, B)
        ADF test p-value per instrument.
    tradable : (B,)
        Boolean/int mask; 1 = tradable.
    pos_limits_long : (B,)
        Maximum long notional per instrument (positive).
    pos_limits_short : (B,)
        Maximum short notional per instrument (negative number).
    maturity : (T, B) or (B,) or None
        Years to maturity per instrument.
    issuer_bucket : (B,) or None
        Integer issuer label per instrument.
    maturity_bucket : (T, B) or (B,) or None
        Integer maturity bucket label per instrument.
    gross_dv01_cap : float
        Maximum total DV01 tradeable per bar.
    issuer_dv01_caps : (n_issuers,) or None
        Max absolute DV01 per issuer bucket.
    mat_bucket_dv01_caps : (n_buckets,) or None
        Max absolute DV01 per maturity bucket.
    top_k : int
        Maximum candidates to pass to the LP solver.
    z_inc, p_inc : float
        Signal gates for risk-increasing fair price.
    z_dec, p_dec : float
        Signal gates for risk-decreasing fair price.
    alpha_thr_inc, alpha_thr_dec : float
        Minimum executable alpha (bps) for risk-increasing / -decreasing.
    max_levels : int
        L2 book levels to walk.
    haircut : float
        Fraction of displayed book size usable (0–1).
    qty_step : (B,) array
        Per-instrument notional rounding step for trade sizes.
    min_qty_trade : float
        Notional below this is zeroed out.
    min_fill_ratio : float
        Minimum filled/requested ratio (0 disables).
    cooldown_bars : int
        Bars of cooldown after executing trades.
    cooldown_mode : int
        0 = block all trades during cooldown; 1 = allow risk-reducing only.
    min_maturity_inc : float
        Minimum years to maturity for risk-increasing trades (0 disables).
    initial_capital : float
        Starting cash.

    Returns
    -------
    tuple
        Per-bar arrays, per-trade arrays, and n_trades_total scalar.
        See source for full layout.
    """
    T = bid_px.shape[0]
    B = bid_px.shape[1]

    # -- Per-instrument qty_step: broadcast scalar to (B,) for compat --
    if np.isscalar(qty_step) or (isinstance(qty_step, np.ndarray) and qty_step.ndim == 0):
        qty_step = np.full(B, float(qty_step), dtype=np.float64)

    # -- Hedge universe size --
    H = hedge_bid_px.shape[1] if hedge_bid_px is not None else 0
    has_hedge = H > 0

    # -- Per-hedge qty_step --
    if hedge_qty_step is None and has_hedge:
        hedge_qty_step = np.zeros(H, dtype=np.float64)
    elif hedge_qty_step is not None and np.isscalar(hedge_qty_step):
        hedge_qty_step = np.full(H, float(hedge_qty_step), dtype=np.float64)

    # -- Per-bar output arrays --
    out_pos = np.zeros((T, B), dtype=np.float64)
    out_cash = np.zeros(T, dtype=np.float64)
    out_equity = np.zeros(T, dtype=np.float64)
    out_pnl = np.zeros(T, dtype=np.float64)
    out_gross_pnl = np.zeros(T, dtype=np.float64)
    out_codes = np.zeros(T, dtype=np.int32)
    out_n_trades_bar = np.zeros(T, dtype=np.int32)
    out_cooldown = np.zeros(T, dtype=np.int32)

    # -- Per-trade output arrays --
    # Upper bound: at most top_k trades per bar.
    max_trades = T * min(B, max(top_k, 1))
    tr_bar = np.full(max_trades, -1, dtype=np.int64)
    tr_instrument = np.full(max_trades, -1, dtype=np.int32)
    tr_side = np.zeros(max_trades, dtype=np.int32)
    tr_qty_req = np.zeros(max_trades, dtype=np.float64)
    tr_qty_fill = np.zeros(max_trades, dtype=np.float64)
    tr_dv01_req = np.zeros(max_trades, dtype=np.float64)
    tr_dv01_fill = np.zeros(max_trades, dtype=np.float64)
    tr_alpha = np.zeros(max_trades, dtype=np.float64)
    tr_fair_type = np.zeros(max_trades, dtype=np.int32)
    tr_vwap = np.zeros(max_trades, dtype=np.float64)
    tr_mid = np.zeros(max_trades, dtype=np.float64)
    tr_cost = np.zeros(max_trades, dtype=np.float64)
    tr_code = np.zeros(max_trades, dtype=np.int32)
    # -- Per-trade hedge output arrays --
    tr_hedge_sizes = np.zeros((max_trades, max(H, 1)), dtype=np.float64)
    tr_hedge_vwaps = np.zeros((max_trades, max(H, 1)), dtype=np.float64)
    tr_hedge_fills = np.zeros((max_trades, max(H, 1)), dtype=np.float64)
    tr_hedge_cost = np.zeros(max_trades, dtype=np.float64)
    n_trades_total = 0

    # -- Per-bar hedge output --
    out_hedge_pos = np.zeros((T, max(H, 1)), dtype=np.float64)
    out_hedge_pnl = np.zeros(T, dtype=np.float64)

    # -- Per-bar MTM and cost breakdown --
    out_instrument_position_mtm = np.zeros(T, dtype=np.float64)
    out_hedge_position_mtm = np.zeros(T, dtype=np.float64)
    out_instrument_cash_mtm = np.zeros(T, dtype=np.float64)
    out_hedge_cash_mtm = np.zeros(T, dtype=np.float64)
    out_portfolio_mtm = np.zeros(T, dtype=np.float64)
    out_instrument_cost = np.zeros(T, dtype=np.float64)
    out_hedge_cost_bar = np.zeros(T, dtype=np.float64)
    out_portfolio_cost = np.zeros(T, dtype=np.float64)

    # -- Mutable state --
    if initial_state is not None:
        pos = initial_state.pos.copy()
        hedge_pos = initial_state.hedge_pos.copy()
        cash = initial_state.cash
        cooldown_remaining = initial_state.cooldown_remaining
        prev_equity = initial_state.prev_equity
        prev_hedge_mtm = initial_state.prev_hedge_mtm
        cum_instrument_cash_mid = initial_state.cum_instrument_cash_mid
        cum_hedge_cash_mid = initial_state.cum_hedge_cash_mid
    else:
        pos = np.zeros(B, dtype=np.float64)
        hedge_pos = np.zeros(max(H, 1), dtype=np.float64)
        cash = float(initial_capital)
        cooldown_remaining = 0
        prev_equity = float(initial_capital)
        prev_hedge_mtm = 0.0
        cum_instrument_cash_mid = 0.0
        cum_hedge_cash_mid = 0.0

    # Detect whether maturity / maturity_bucket are time-varying (T, B)
    maturity_2d = maturity is not None and maturity.ndim == 2
    maturity_bucket_2d = maturity_bucket is not None and maturity_bucket.ndim == 2

    # Pre-compute current bucket dv01 arrays (updated after each bar's trades)
    n_issuers = len(issuer_dv01_caps) if issuer_dv01_caps is not None else 0
    n_mat_buckets = len(mat_bucket_dv01_caps) if mat_bucket_dv01_caps is not None else 0
    current_issuer_dv01 = np.zeros(max(n_issuers, 1), dtype=np.float64)
    current_mat_dv01 = np.zeros(max(n_mat_buckets, 1), dtype=np.float64)

    for t in range(T):
        code = _NO_ACTION
        bar_cost = 0.0
        bar_position_cost = 0.0
        bar_hedge_cost_t = 0.0
        bar_n_trades = 0
        hedge_trade_cash = 0.0  # sum of h_signed * h_vwap for hedge trades this bar

        # ==============================================================
        # Step 0: Cooldown check
        # ==============================================================
        in_cooldown = cooldown_remaining > 0
        allow_risk_reducing_only = False
        if in_cooldown:
            cooldown_remaining -= 1
            if cooldown_mode == _COOLDOWN_BLOCK_ALL:
                # Skip all trading this bar
                code = _SKIP_COOLDOWN
                # Jump to bookkeeping
                _skip_to_bookkeeping = True
            else:
                # Allow risk-reducing trades only
                allow_risk_reducing_only = True
                _skip_to_bookkeeping = False
                code = _SKIP_COOLDOWN_RISK_ONLY
        else:
            _skip_to_bookkeeping = False

        if not _skip_to_bookkeeping:
            # ==========================================================
            # Step 1: Gate fair prices
            # ==========================================================
            # fair_inc active when |zscore| > z_inc AND adf_p < p_inc
            # fair_dec active when |zscore| > z_dec AND adf_p < p_dec
            fair_inc_active = np.zeros(B, dtype=np.bool_)
            fair_dec_active = np.zeros(B, dtype=np.bool_)
            for b in range(B):
                az = abs(zscore[t, b])
                ap = adf_p_value[t, b]
                if az > z_inc and ap < p_inc:
                    fair_inc_active[b] = True
                if az > z_dec and ap < p_dec:
                    fair_dec_active[b] = True

            # ==========================================================
            # Step 2-5: Build candidates
            # ==========================================================
            # Candidate arrays (pre-allocate to B, trim later)
            cand_inst = np.empty(B, dtype=np.int32)
            cand_side = np.empty(B, dtype=np.int32)
            cand_alpha = np.empty(B, dtype=np.float64)
            cand_fair_type = np.empty(B, dtype=np.int32)
            cand_dv01_liq = np.empty(B, dtype=np.float64)
            cand_pos_headroom = np.empty(B, dtype=np.float64)
            n_cand = 0

            for b in range(B):
                # -- Eligibility: tradable --
                if not tradable[b]:
                    continue

                # -- DV01 valid --
                dv01_b = dv01[t, b]
                if dv01_b < 1e-15:
                    continue

                # -- Market valid --
                bid0 = bid_px[t, b, 0]
                ask0 = ask_px[t, b, 0]
                mid_b = mid_px[t, b]
                if not _check_market_valid(bid0, ask0, mid_b):
                    continue

                fair_b = fair_price[t, b]
                if np.isnan(fair_b):
                    continue

                # -- Determine candidate directions --
                # Check both BUY and SELL opportunities for this instrument
                for side in (1, -1):
                    # Risk classification
                    if side == 1:  # BUY
                        is_risk_dec = pos[b] < -1e-15  # closing/reducing short
                    else:  # SELL
                        is_risk_dec = pos[b] > 1e-15   # closing/reducing long

                    # In cooldown with risk-reducing only: skip risk-increasing
                    if allow_risk_reducing_only and not is_risk_dec:
                        continue

                    # Choose fair based on risk type
                    if is_risk_dec:
                        if not fair_dec_active[b]:
                            continue
                        chosen_fair = fair_b  # fair_price is the base
                        fair_tp = _FAIR_DEC
                        alpha_thr = alpha_thr_dec
                    else:
                        if not fair_inc_active[b]:
                            continue
                        chosen_fair = fair_b
                        fair_tp = _FAIR_INC
                        alpha_thr = alpha_thr_inc

                    # Executable alpha (bps)
                    if side == 1:  # BUY
                        # Opportunity: fair > ask
                        alpha_bps = (chosen_fair - ask0) / dv01_b
                    else:  # SELL
                        # Opportunity: fair < bid  => bid - fair > 0
                        alpha_bps = (bid0 - chosen_fair) / dv01_b

                    if alpha_bps <= alpha_thr:
                        continue

                    # Liquidity check: must have depth on required side
                    if side == 1:
                        if ask_sz[t, b, 0] <= 1e-15:
                            continue
                    else:
                        if bid_sz[t, b, 0] <= 1e-15:
                            continue

                    # Maturity filter for risk-increasing
                    if (not is_risk_dec and min_maturity_inc > 0.0
                            and maturity is not None):
                        mat_val = maturity[t, b] if maturity_2d else maturity[b]
                        if mat_val < min_maturity_inc:
                            continue

                    # Position limits: compute headroom in notional
                    if side == 1:  # BUY → position increases
                        headroom_notional = pos_limits_long[b] - pos[b]
                    else:  # SELL → position decreases
                        headroom_notional = pos[b] - pos_limits_short[b]

                    # Risk-decreasing: cap at current position so we don't
                    # overshoot past flat and flip the position.
                    if is_risk_dec:
                        headroom_notional = min(headroom_notional, abs(pos[b]))
                        headroom_notional = min(
                            headroom_notional, max_trade_notional_dec[b])
                    else:
                        headroom_notional = min(
                            headroom_notional, max_trade_notional_inc[b])

                    if headroom_notional < 1e-15:
                        continue

                    # Convert headroom to DV01
                    headroom_dv01 = headroom_notional * dv01_b

                    # Liquidity cap in DV01: walk the book to find max fillable
                    total_avail = 0.0
                    if side == 1:
                        book_px = ask_px[t, b]
                        book_sz = ask_sz[t, b]
                    else:
                        book_px = bid_px[t, b]
                        book_sz = bid_sz[t, b]
                    nl = min(max_levels, len(book_px))
                    for lev in range(nl):
                        total_avail += book_sz[lev] * haircut
                    dv01_liq = total_avail * dv01_b

                    # Store candidate
                    cand_inst[n_cand] = b
                    cand_side[n_cand] = side
                    cand_alpha[n_cand] = alpha_bps
                    cand_fair_type[n_cand] = fair_tp
                    cand_dv01_liq[n_cand] = dv01_liq
                    cand_pos_headroom[n_cand] = headroom_dv01
                    n_cand += 1

            # ==========================================================
            # Step 6: Rank by alpha and keep top K
            # ==========================================================
            if n_cand == 0:
                code = _NO_CANDIDATES if not in_cooldown else code
            else:
                # Trim to actual candidate count
                c_inst = cand_inst[:n_cand]
                c_side = cand_side[:n_cand]
                c_alpha = cand_alpha[:n_cand]
                c_fair_type = cand_fair_type[:n_cand]
                c_dv01_liq = cand_dv01_liq[:n_cand]
                c_pos_hr = cand_pos_headroom[:n_cand]

                if n_cand > top_k:
                    # Keep top_k by descending alpha
                    top_idx = np.argsort(-c_alpha)[:top_k]
                    c_inst = c_inst[top_idx]
                    c_side = c_side[top_idx]
                    c_alpha = c_alpha[top_idx]
                    c_fair_type = c_fair_type[top_idx]
                    c_dv01_liq = c_dv01_liq[top_idx]
                    c_pos_hr = c_pos_hr[top_idx]
                    n_cand = top_k

                # ==================================================
                # Step 7: Compute current bucket exposures for LP
                # ==================================================
                if n_issuers > 0 and issuer_bucket is not None:
                    current_issuer_dv01[:] = 0.0
                    for b in range(B):
                        iss = issuer_bucket[b]
                        if 0 <= iss < n_issuers:
                            current_issuer_dv01[iss] += pos[b] * dv01[t, b]

                if n_mat_buckets > 0 and maturity_bucket is not None:
                    current_mat_dv01[:] = 0.0
                    mat_bkt_t = maturity_bucket[t] if maturity_bucket_2d else maturity_bucket
                    for b in range(B):
                        bkt = mat_bkt_t[b]
                        if 0 <= bkt < n_mat_buckets:
                            current_mat_dv01[bkt] += pos[b] * dv01[t, b]

                # ==================================================
                # Step 8: Solve LP / greedy
                # ==================================================
                mat_bkt_lp = maturity_bucket[t] if maturity_bucket_2d else maturity_bucket
                dv01_alloc, used_greedy = _solve_lp(
                    c_alpha,
                    c_dv01_liq,
                    gross_dv01_cap,
                    c_pos_hr,
                    c_inst,
                    c_side,
                    issuer_bucket,
                    mat_bkt_lp,
                    issuer_dv01_caps,
                    mat_bucket_dv01_caps,
                    current_issuer_dv01,
                    current_mat_dv01,
                )

                # ==================================================
                # Step 9: Round and execute
                # ==================================================
                any_executed = False
                any_partial = False
                any_failed = False

                # Accumulate target hedge position across all instrument fills
                if has_hedge:
                    hedge_target = hedge_pos.copy()

                for k in range(n_cand):
                    raw_dv01 = dv01_alloc[k]
                    if raw_dv01 < 1e-15:
                        continue

                    b_idx = int(c_inst[k])
                    side_k = int(c_side[k])
                    dv01_b = dv01[t, b_idx]

                    if dv01_b < 1e-15:
                        continue

                    # Convert DV01 to notional quantity, then round size
                    raw_qty = raw_dv01 / dv01_b
                    qty_req = _round_qty_trade(raw_qty, min_qty_trade, qty_step[b_idx])
                    if qty_req < 1e-15:
                        continue

                    # Execute via book walking
                    if side_k == 1:  # BUY → lift ask
                        filled, vwap = _walk_book(
                            ask_px[t, b_idx], ask_sz[t, b_idx],
                            qty_req, max_levels, haircut,
                        )
                    else:  # SELL → hit bid
                        filled, vwap = _walk_book(
                            bid_px[t, b_idx], bid_sz[t, b_idx],
                            qty_req, max_levels, haircut,
                        )

                    # Check fill ratio
                    fill_ratio = filled / qty_req if qty_req > 1e-15 else 0.0

                    if filled < 1e-15:
                        # No fill at all
                        tr_code_k = _FILL_FAILED_LIQUIDITY
                        any_failed = True
                    elif min_fill_ratio > 0.0 and fill_ratio < min_fill_ratio:
                        # Below minimum fill ratio → skip
                        tr_code_k = _FILL_BELOW_MIN
                        any_failed = True
                        filled = 0.0  # treat as not filled
                    elif filled < qty_req - 1e-10:
                        tr_code_k = _FILL_PARTIAL
                        any_partial = True
                    else:
                        tr_code_k = _FILL_OK

                    # Record trade (even if failed, for diagnostics)
                    if n_trades_total < max_trades:
                        mid_b = mid_px[t, b_idx]
                        dv01_filled = filled * dv01_b
                        exec_cost = abs(vwap - mid_b) * filled if filled > 1e-15 else 0.0

                        tr_bar[n_trades_total] = t
                        tr_instrument[n_trades_total] = b_idx
                        tr_side[n_trades_total] = side_k
                        tr_qty_req[n_trades_total] = qty_req
                        tr_qty_fill[n_trades_total] = filled
                        tr_dv01_req[n_trades_total] = qty_req * dv01_b
                        tr_dv01_fill[n_trades_total] = dv01_filled
                        tr_alpha[n_trades_total] = c_alpha[k]
                        tr_fair_type[n_trades_total] = c_fair_type[k]
                        tr_vwap[n_trades_total] = vwap
                        tr_mid[n_trades_total] = mid_b
                        tr_cost[n_trades_total] = exec_cost
                        tr_code[n_trades_total] = tr_code_k
                        n_trades_total += 1

                    # Update pos/cash only if filled
                    if filled > 1e-15 and tr_code_k in (_FILL_OK, _FILL_PARTIAL):
                        signed_qty = side_k * filled
                        # BUY: cash -= qty * vwap, pos += qty
                        # SELL: cash += qty * vwap, pos -= qty
                        cash -= signed_qty * vwap
                        pos[b_idx] += signed_qty
                        cum_instrument_cash_mid -= signed_qty * mid_px[t, b_idx]
                        pos_exec_cost = abs(vwap - mid_px[t, b_idx]) * filled
                        bar_cost += pos_exec_cost
                        bar_position_cost += pos_exec_cost
                        any_executed = True
                        bar_n_trades += 1

                        # -- Accumulate hedge target --
                        if has_hedge:
                            for h in range(H):
                                hr = hedge_ratios[t, b_idx, h]
                                if abs(hr) < 1e-15:
                                    continue
                                hdv01 = hedge_dv01[t, h]
                                if hdv01 < 1e-15:
                                    continue
                                hedge_target[h] += signed_qty * dv01_b * hr / hdv01

                # ==================================================
                # Step 10: Execute net hedge for the bar
                # ==================================================
                if has_hedge and any_executed:
                    bar_hedge_cost = 0.0
                    for h in range(H):
                        hedge_remaining = hedge_target[h] - hedge_pos[h]
                        # Round hedge trade size using per-hedge qty_step
                        hedge_remaining = _round_qty_trade(
                            hedge_remaining, min_qty_trade, hedge_qty_step[h],
                        )
                        if abs(hedge_remaining) < 1e-15:
                            continue
                        if hedge_remaining > 1e-15:
                            h_filled, h_vwap = _walk_book(
                                hedge_ask_px[t, h], hedge_ask_sz[t, h],
                                hedge_remaining, max_levels, haircut,
                            )
                        else:
                            h_filled, h_vwap = _walk_book(
                                hedge_bid_px[t, h], hedge_bid_sz[t, h],
                                abs(hedge_remaining), max_levels, haircut,
                            )
                        if h_filled < 1e-15:
                            continue
                        h_signed = np.sign(hedge_remaining) * h_filled
                        cash -= h_signed * h_vwap
                        hedge_trade_cash += h_signed * h_vwap
                        cum_hedge_cash_mid -= h_signed * hedge_mid_px[t, h]
                        hedge_pos[h] += h_signed
                        h_exec_cost = abs(h_vwap - hedge_mid_px[t, h]) * h_filled
                        bar_cost += h_exec_cost
                        bar_hedge_cost += h_exec_cost
                        bar_hedge_cost_t += h_exec_cost
                        # Record in the last trade's hedge arrays
                        last_trade = n_trades_total - 1
                        if last_trade >= 0 and last_trade < max_trades:
                            tr_hedge_sizes[last_trade, h] = hedge_remaining
                            tr_hedge_fills[last_trade, h] = h_signed
                            tr_hedge_vwaps[last_trade, h] = h_vwap
                    if bar_hedge_cost > 0.0:
                        last_trade = n_trades_total - 1
                        if last_trade >= 0 and last_trade < max_trades:
                            tr_hedge_cost[last_trade] = bar_hedge_cost

                # Determine bar code
                if any_executed and not any_partial and not any_failed:
                    code = _EXEC_GREEDY if used_greedy else _EXEC_OK
                elif any_executed and (any_partial or any_failed):
                    code = _EXEC_PARTIAL
                elif not any_executed and any_failed:
                    code = _EXEC_NO_LIQUIDITY
                else:
                    # LP returned all zeros (infeasible or no allocation)
                    code = _LP_INFEASIBLE if not used_greedy else _LP_NO_CANDIDATES

                # Start cooldown if any trades were executed
                if any_executed and cooldown_bars > 0:
                    cooldown_remaining = cooldown_bars

        # ==============================================================
        # End-of-bar bookkeeping
        # ==============================================================
        # equity = cash + sum(pos[b] * mid[b]) + sum(hedge_pos[h] * hedge_mid[h])
        position_mtm_t = 0.0
        for b in range(B):
            position_mtm_t += pos[b] * mid_px[t, b]
        hedge_mtm_t_eq = 0.0
        if has_hedge:
            for h in range(H):
                hedge_mtm_t_eq += hedge_pos[h] * hedge_mid_px[t, h]
        equity_t = cash + position_mtm_t + hedge_mtm_t_eq

        pnl_t = equity_t - prev_equity
        gross_pnl_t = pnl_t + bar_cost
        prev_equity = equity_t

        # Hedge PnL: MTM change minus cash outflows for hedge trades
        hedge_pnl_t = 0.0
        if has_hedge:
            hedge_pnl_t = hedge_mtm_t_eq - prev_hedge_mtm - hedge_trade_cash
            prev_hedge_mtm = hedge_mtm_t_eq

        # Write output arrays
        for b in range(B):
            out_pos[t, b] = pos[b]
        if has_hedge:
            for h in range(H):
                out_hedge_pos[t, h] = hedge_pos[h]
        out_cash[t] = cash
        out_equity[t] = equity_t
        out_pnl[t] = pnl_t
        out_gross_pnl[t] = gross_pnl_t
        out_hedge_pnl[t] = hedge_pnl_t
        out_codes[t] = code
        out_n_trades_bar[t] = bar_n_trades
        out_cooldown[t] = cooldown_remaining
        out_instrument_position_mtm[t] = position_mtm_t
        out_hedge_position_mtm[t] = hedge_mtm_t_eq
        out_instrument_cash_mtm[t] = cum_instrument_cash_mid
        out_hedge_cash_mtm[t] = cum_hedge_cash_mid
        out_portfolio_mtm[t] = (cum_instrument_cash_mid + cum_hedge_cash_mid
                                + position_mtm_t + hedge_mtm_t_eq)
        out_instrument_cost[t] = bar_position_cost
        out_hedge_cost_bar[t] = bar_hedge_cost_t
        out_portfolio_cost[t] = bar_position_cost + bar_hedge_cost_t

    result = (
        # Per-bar arrays (9)
        out_pos,            # 0: (T, B)
        out_cash,           # 1: (T,)
        out_equity,         # 2: (T,)
        out_pnl,            # 3: (T,)
        out_gross_pnl,      # 4: (T,)
        out_codes,          # 5: (T,)
        out_n_trades_bar,   # 6: (T,)
        out_cooldown,       # 7: (T,)
        out_hedge_pos,      # 8: (T, H)
        # Per-trade arrays (17)
        tr_bar,             # 9:  (max_trades,)
        tr_instrument,            # 10: (max_trades,)
        tr_side,            # 11: (max_trades,)
        tr_qty_req,         # 12: (max_trades,)
        tr_qty_fill,        # 13: (max_trades,)
        tr_dv01_req,        # 14: (max_trades,)
        tr_dv01_fill,       # 15: (max_trades,)
        tr_alpha,           # 16: (max_trades,)
        tr_fair_type,       # 17: (max_trades,)
        tr_vwap,            # 18: (max_trades,)
        tr_mid,             # 19: (max_trades,)
        tr_cost,            # 20: (max_trades,)
        tr_code,            # 21: (max_trades,)
        tr_hedge_sizes,     # 22: (max_trades, H)
        tr_hedge_vwaps,     # 23: (max_trades, H)
        tr_hedge_fills,     # 24: (max_trades, H)
        tr_hedge_cost,      # 25: (max_trades,)
        # Per-bar hedge PnL
        out_hedge_pnl,      # 26: (T,)
        # Per-bar MTM and cost breakdown
        out_instrument_position_mtm,  # 27: (T,)
        out_hedge_position_mtm,       # 28: (T,)
        out_instrument_cash_mtm,      # 29: (T,) cumulative cash from instrument trades at mid
        out_hedge_cash_mtm,           # 30: (T,) cumulative cash from hedge trades at mid
        out_portfolio_mtm,            # 31: (T,) sum of 27-30
        out_instrument_cost,          # 32: (T,)
        out_hedge_cost_bar,           # 33: (T,)
        out_portfolio_cost,           # 34: (T,) instrument + hedge cost
        # Scalar
        n_trades_total,               # 35
    )

    if return_final_state:
        final_state = LoopState(
            pos=pos.copy(),
            hedge_pos=hedge_pos.copy(),
            cash=cash,
            cooldown_remaining=cooldown_remaining,
            prev_equity=prev_equity,
            prev_hedge_mtm=prev_hedge_mtm,
            cum_instrument_cash_mid=cum_instrument_cash_mid,
            cum_hedge_cash_mid=cum_hedge_cash_mid,
        )
        return result + (final_state,)

    return result
