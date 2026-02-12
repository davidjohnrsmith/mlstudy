

You are Claude. Please inspect and update the code in this repository folder:

  src/mlstudy/trading/backtest

Goal
Modify/extend the existing backtester so it supports a mean-reversion strategy workflow with:
- L2 order book execution (book-walk + VWAP)
- explicit state machine (FLAT/LONG/SHORT + quarantine)
- per-bar attempt/outcome tracking using standardized codes
- JIT-capable hot loop (Numba-friendly arrays/scalars)

IMPORTANT constraints
- Do NOT add post-analysis/reporting features; focus only on engine mechanics, state machine, execution simulation, and trackers.
- Avoid lookahead: decisions at bar t may only use signal inputs computed from information available up to t-1 (assume signals passed in are already lagged correctly). If you compute anything internally, lag it.
- Keep code modular: signals are inputs; do not build a general signal library.

Portfolio / hedge ratio conventions
- The trading package contains a reference instrument + other instruments.
- Hedge ratios are in yield space with:
    r_ref = +1 (reference instrument ratio is 1)
    sum_i r_i = 0
- The package is traded as a basket of legs with sizes derived from hedge ratios and DV01 (Option A: output leg sizes directly to execution).

Inputs the backtester must support
Market data per bar t:
- L2 book:
    bid_px[t, i, l], bid_sz[t, i, l]
    ask_px[t, i, l], ask_sz[t, i, l]
  for instruments i=0..N-1 and levels l=0..L-1 (top level l=0).
- Mid price per instrument mid_px[t, i] is given separately (which are not based the bid and offer here).
- DV01 per instrument dv01[t, i], with units par 1 price per 1bp yield

Signal inputs per bar t (already computed and lagged appropriately):
- zscore[t]  (from mid yields)
- expected_yield_pnl_bps[t]
    DEFINITION: expected yield change of the package residual, in yield bp (not price bp).
- package_yield_bps[t]
    DEFINITION: package_yield_bps(t), in yield bp.

Config inputs (must be supported, minimal set)
Entry:
- entry_z_threshold

Take profit (TP) thresholds (REQUIRED, use these exact names/semantics):
- take_profit_zscore_soft_threshold
- take_profit_yield_change_soft_threshold
- take_profit_yield_change_hard_threshold

Stop loss threshold (REQUIRED, use this exact name/semantics):
- stop_loss_yield_change_hard_threshold

Other exit / governance:
- max_holding_bars
- expected_yield_pnl_bps_multiplier (applied to expected_yield_pnl_bps)
- cost premia in yield bp:
    entry_cost_premium_yield_bps
    tp_cost_premium_yield_bps
    sl_cost_premium_yield_bps
- quarantine cooldowns (bars):
    tp_quarantine_bars
    sl_quarantine_bars
    time_quarantine_bars
- sizing risk budget:
    e.g. target_notional_ref
- market validity check scope:
    validate_scope = "REF_ONLY" or "ALL_LEGS"
- execution:
    max_levels_to_cross (for L2 walk)
    size_haircut (optional, default 1.0)

Standardized per-bar attempt/outcome codes (MANDATORY)
Every bar must record a single action/outcome code from:
- ENTRY_xxx
- EXIT_TP_xxx
- EXIT_SL_xxx
- EXIT_TIME_xxx (if you implement max holding as a separate exit category; otherwise map to EXIT_SL or a generic exit)
- NO_ACTION (optional; if no attempt occurs and you prefer a neutral code)

Suffix set (shared):
- OK               : trade executed and state changed as intended
- NO_SIGNAL        : no attempt because conditions not met (optional; can also use NO_ACTION)
- INVALID_BOOK     : market validity check failed
- NO_LIQUIDITY     : cannot fully fill required quantity via L2 walk
- TOO_WIDE         : fully fillable but execution cost > acceptable cost
- IN_COOLDOWN      : attempt blocked by quarantine
- NOT_FLAT         : entry blocked because already in position
- FORCED           : used for forced exits (SL/TIME) where no cost acceptance check is applied

Core mechanics to implement

(0) Main loop
- Iterate bars t in a for-loop.
- Initial state = FLAT.
- Maintain strategy state across bars.

(1) Read signals at t
- zscore[t], expected_yield_pnl_bps[t], package_yield_bps[t]

(2) Entry intent
- If abs(zscore) beyond entry_z_threshold => strategy wants to enter LONG or SHORT accordingly.
- sign convention: zscore > 0 -> cheap;  zscore  < 0 -> expensive

(3) Compute acceptable execution cost for entry (in PRICE DELTA)
- acceptable_yield_bps = multiplier * expected_yield_pnl_bps[t] - entry_cost_premium_yield_bps
- Convert yield-bps budget -> price delta using DV01:
  You must define and implement a consistent conversion based on your DV01 convention and the basket sizing.
  Example (if DV01 in price points per bp per unit):
    acceptable_price_delta = acceptable_yield_bps * basket_dv01_price_per_bp_per_unit
  where basket_dv01 is computed consistently for the intended basket size.
- Document clearly how basket_dv01 is computed (e.g., ref leg dv01 * size_ref, or sum(|dv01_i * size_i|), etc.). Choose one, implement, and keep consistent.

(4) Basket sizing for entry
- Compute leg sizes using hedge ratios + DV01:
  - ref ratio is 1; other ratios sum to -1 across others.
  - target_notional_ref; then size_i derived from yield-space hedge ratios + DV01 scaling so yield hedge holds.
- Output leg sizes directly in leg space.

(5) Market validity check
- Confirm bid/ask are valid:
    bid0 <= mid <= ask0
- Provide config option:
    validate_scope = REF_ONLY vs ALL_LEGS.
- If invalid -> set attempt outcome code and do not trade.

(6) Liquidity + execution pricing (L2 walk)
- For each leg:
  - If buying: walk ask levels from best to deeper until fully filled; compute VWAP.
  - If selling: walk bid levels similarly; compute VWAP.
- Basket entry requires ATOMIC full fill across all legs:
  - If any leg cannot be fully filled => ENTRY_NO_LIQUIDITY (no partial entry; remain FLAT).
- If full fill possible:
  - Compute execution cost:
      buy:  VWAP - mid
      sell: mid - VWAP
    For basket, define cost aggregation consistently (e.g., ref-leg cost only, or DV01-weighted basket cost). Choose one, implement, document.
  - Entry condition:
      basket_cost <= acceptable_price_delta
    If true => enter; else => ENTRY_TOO_WIDE.

(7) Entry state updates
- If entry succeeds:
  - state => LONG or SHORT
  - record trade tracker: timestamp/bar idx, intended sizes, filled sizes, VWAPs, costs, mid, reason, attempt code ENTRY_OK
- If fails:
  - remain FLAT
  - record attempt code:
      ENTRY_INVALID_BOOK / ENTRY_NO_LIQUIDITY / ENTRY_TOO_WIDE / ENTRY_IN_COOLDOWN

(8) Retry policy
- Skip retry within the same bar (no pending orders).
- Re-evaluate next bar t+1 using signals[t+1].

(9) PnL tracking
- Mark-to-mid each bar.
- - still update MTM PnL using mid_px even if invalid
- Track equity curve.
- Include slippage as execution-vs-mid at fills.
- Update pnl tracker and trades tracker EVERY bar.

(10) Take profit (TP) intent (UPDATED LOGIC — MUST MATCH EXACTLY)
Trigger TP if EITHER:
- calculate package_yield_delta_bps = package_yield_bps(t) - package_yield_delta_bps(entry_t) which is in trades tracker
(A) (zscore[t] > take_profit_zscore_soft_threshold) AND (package_yield_delta_bps[t] > take_profit_yield_change_soft_threshold)
OR
(B) (package_yield_delta_bps[t] > take_profit_yield_change_hard_threshold)

(11) Compute acceptable TP cost (in PRICE DELTA)
- acceptable_yield_bps = multiplier * expected_yield_pnl_bps[t] - tp_cost_premium_yield_bps
- Convert to acceptable_price_delta via DV01 using the same convention as entry.

(12) Exit sizing for TP
- Fully close current position (use current position sizes).

(13) Apply (5) validity + (6) liquidity/cost acceptance for TP
- TP exit requires atomic full fill.
- If full fill and cost ok => EXIT_TP_OK and close position.
- If invalid => EXIT_TP_INVALID_BOOK
- If not fillable => EXIT_TP_NO_LIQUIDITY
- If too wide => EXIT_TP_TOO_WIDE
- Optional: TP quarantine. If configured, after EXIT_TP_OK set quarantine timer tp_quarantine_bars and block entry.

(14) Stop-loss / max-holding forced exits
Stop-loss triggers if:
- package_yield_delta_bps[t] > stop_loss_yield_change_hard_threshold

Max-holding triggers if:
- holding duration >= max_holding_bars

Forced exit semantics:
- Execute by L2 walking until fully filled; DO NOT apply cost acceptance check.
- If book has zero depth, record EXIT_SL_NO_LIQUIDITY (or keep trying next bar until complete; choose one, document).
- Update attempt code:
    EXIT_SL_FORCED when forcing exit (and EXIT_SL_OK when fully flat, if you separate)
    EXIT_TIME_FORCED similarly for max-holding (or map to EXIT_SL_* if you prefer fewer categories).

(15) Quarantine after forced exits
- After SL or TIME exit, set quarantine timer (sl_quarantine_bars or time_quarantine_bars).
- During quarantine:
  - block entry attempts with ENTRY_IN_COOLDOWN (or NO_ACTION + separate cooldown state; choose one and be consistent).

(16) Return to FLAT after quarantine
- When quarantine timer reaches 0, allow entry again; state is FLAT.

(17) Always update trackers
- Every bar must produce:
  - action/outcome code
  - positions
  - pnl/equity
  - and if an attempt occurred, record attempt metadata even if no fill.

Deliverables
A) Audit current code:
   - list modules/classes under src/mlstudy/trading/backtest
   - map what exists vs missing vs what must change
B) Implement missing pieces with minimal churn:
   - strategy state machine + quarantine
   - L2 book-walk execution returning (filled_qty, vwap, fully_filled)
   - atomic basket fill logic
   - cost acceptance check vs acceptable_price_delta
   - trackers: per-bar pnl/equity arrays + per-trade records + per-bar attempt codes
C) Tests:
   - At least one end-to-end toy test generating synthetic yields/dv01 + 2-level L2 book.
   - Must cover: successful entry, successful TP, forced SL exit, and one failed entry (NO_LIQUIDITY or TOO_WIDE).
D) Ensure no lookahead:
   - signals used at t must be lagged (assume provided inputs are lagged); document this and ensure no internal leakage
E) Keep interfaces simple:
   - Inputs: L2 book + mid prices + dv01 + signals.
   - Sizing: hedge ratios + dv01 -> leg sizes directly to execution.
F) JIT support:
   - Implement a Numba-friendly jit_loop (arrays/scalars only; no dict/list/dataclass in the hot loop).
   - Provide a Python reference loop for debugging; outputs must match on the toy test.
   - If full JIT for trade-record list is difficult, keep trade records collected outside the jit hot loop but ensure per-bar arrays and attempt codes are produced by jit.

Please implement changes directly in code and show key diffs/new files.

