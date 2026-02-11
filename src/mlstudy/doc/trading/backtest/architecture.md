
## Responsibilities

### core/
- **engine.py**: orchestrates validation → packing → JIT loop → results.
- **types.py**: `BacktestInputs`, `BacktestConfig`, `OrderBookL2`, enums.
- **packer.py**: converts nested config to flat arrays/scalars for JIT; validates required fields.
- **jit_loop.py**: numba loop over time; pure array state transitions; calls JIT helpers for fills/mtm.
- **results.py**: result containers (positions, pnl, cash/equity, fills).

### data/
- **align.py**: calendar alignment, missing-bar handling.
- **validation.py**: shape checks and sanity checks (bid<ask, sizes>=0, (T,N,L) consistency).
- **roll.py**: optional roll / active-mask utilities.

### execution/ (L2 order book)
- **timing.py**: execution lag rules and mapping to which book snapshot is used.
- **fills.py**: book-walk VWAP fills, depth caps, size haircuts, partial-fill/reject reasons.
- **liquidity.py**: participation caps vs bar volume/ADV; pre-clip orders.
- **impact.py**: extra impact/noise beyond walked book.
- **rounding.py**: min trade size, lot rounding, netting vs close-then-open.
- **costs.py**: explicit fees/commissions, per-leg multipliers.

### lifecycle/
(Optional: used only if the engine manages entry/exit logic; can be bypassed if controls provide targets/orders.)
- **base.py**: PositionManager interface.
- **mean_reversion.py**: MR state machine.
- **momentum.py**: MOM state machine.
- **common_guards.py**: generic stops/time stops/cooldowns.

### risk/
- **limits.py**: gross/net position caps, DV01 caps, concentration limits.
- **derisk.py**: drawdown de-risk policy.
- **exposures.py**: DV01 and factor exposures (optional).

### accounting/
- **mtm.py**: mark price selection (MID/LAST/EXTERNAL).
- **ledger.py**: position/cash/equity updates; PnL accounting.
- **funding.py**: repo/borrow/coupon accrual (optional).

### scenarios/
- **presets.py**: FAST/REALISTIC/STRESS config presets.
- **sweeps.py**: parameter sweeps (haircut, depth, impact, lag, participation).

### tests/
- **test_execution.py**: book-walk correctness, partial fills, rejects, haircuts.
- **test_engine.py**: no lookahead, cash/pos invariants.
- lifecycle tests as needed.

## Notes
- Store **truth positions in legs (N)**; portfolios (M) are mapped to legs via weights `W[t,m,i]` if used.
- Execution uses L2 book snapshots; MTM typically uses mid = (best_bid+best_ask)/2 unless overridden.
EOF

i want to claude to look at codes in src/mlstudy/trading/backtest, and update any if needed to support my backtest functionality i want to have
(0) do a for loop all given market data with initial state = flat
(1) test signal is focusing on mean reverting now, in yield space, zscore (based on mid yield) to decide to signal wants to enter or exit, signal value = expected pnl in bps
(2) try enter if signal wants to enter (zscore> a threshold), and convert ( multiplier * expected pnl in bps - enter cost premium in bps) to acceptable cost in prices delta using dv01
(3) calculate the notional to enter or exit based on hedge ratios in yield space and dv01
(4) check bid offer market is valid, i,e, provided mid price in top book bid offer price
(5) check the liquidity: if signal is buying, check offer prices and sizes from top level to deeper level, until fully filled, calculate the weighted price to buy, and calculate the cost: weighted price - mid; enter position if weighted price - mid <= acceptable cost in prices delta;  like wise for selling
(6) if enters, change state to long or short and update trades tracker, change state to otherwise try_long_but_no_liquidity(if cannot be fully filled),try_long_but_bid_offer_wide (if cost > acceptable cost in prices delta) etc.
(7) signal repeat 3-6 until until enters
(8) tracking pnl 
(9) try taking profit if signal (zscore + signal package yield delta > a soft threshold) or signal package yield delta > a hard threshold; and convert ( multiplier * expected pnl in bps - take profit cost premium in bps) to acceptable cost in prices delta using dv01
(10) calculate notional to exit based on the position
(11) repeat (4)
(12) repeat (5) and the exit if condition ok
(13) change state to take_profit/take_profit_quarantine (if tuning for a gating period after taking profit) and update trades tracker, try_take_profit_but_no_liquidity, try_take_profit_but_bid_offer_wide
(13) try stop loss/max_holding if signal package yield delta > a hard threshold;; and convert ( multiplier * expected pnl in bps - stop loss cost premium in bps) to acceptable cost in prices delta using dv01
(14) repeat (5), but no condition check, force to exit
(15) change state to stop_loss/stop_loss_quarantine(if tuning for a gating period after stopping loss)/max_holding/max_holding_quarantine  and update trades tracker
(16) change state flat if quarantine period passed (if no quarantine tuning, then  quarantine period =0)
(17) always update pnl tracker and trades tracker at each step

can you review my backtester design and generate the claude task



You are Claude. Please inspect and update the code in this repository folder:

  src/mlstudy/trading/backtest

Goal: modify/extend the existing backtester so it supports the following mean-reversion strategy workflow with L2 book execution and explicit state machine.

IMPORTANT constraints:
- Do NOT add post-analysis/reporting features; focus only on the backtest engine mechanics, state machine, execution simulation, and trackers.
- Avoid lookahead: any signal used for decisions at bar t must be computed using information available up to t-1 unless explicitly lagged/executed at t+1.
- Keep code modular: signal computation can be included only insofar as needed for mean-reversion zscore; but do not build a general signal library.

the trading package contains a ref instrument + other instruments, the hedge ratio is assumed the ratio of the ref is 1, and sum of all ratios are 0 
Desired behavior (must implement):
(0) Iterate market data bars in a for-loop. Initial strategy state = FLAT. Maintain strategy state across bars.
(1) Mean-reversion signal in yield space, input signals have calculated :
    - zscore: calculated from mid yields.
    - Signal value = expected PnL in yield bps (yield-bps or price-bps; pick one convention and document it).
(2) Entry intention:
    - If zscore > entry threshold => wants to enter short/long accordingly.
    - Convert (multiplier * expected_pnl_yield_bps - entry_cost_premium_yield_bps) into an acceptable execution cost in par 1 PRICE DELTA using DV01.
(3) Sizing:
    - Compute trade notionals/sizes based on hedge ratios in yield space and DV01
(4) Market validity:
    - Confirm bid/ask are valid: provided mid price should lie within top-of-book bid/ask (bid <= mid <= ask). If invalid, mark attempt state
    - with option to check ref instrument or all instrument 
(5) Liquidity + execution pricing:
    - If buying, walk offer levels (L2) from best to deeper until fully filled. Compute VWAP.
    - Compute execution cost = (VWAP - mid) for buy, (mid - VWAP) for sell.
    - Entry condition: execution cost <= acceptable cost (price delta). If true, enter.
(6) If entry succeeds:
    - Set state to LONG or SHORT.
    - Update trade tracker (fills, VWAP, costs, reason, timestamp/bar index).
    - If fails: set state to TRY_LONG_NO_LIQUIDITY / TRY_LONG_TOO_WIDE / TRY_SHORT_NO_LIQUIDITY / TRY_SHORT_TOO_WIDE etc.
(7) Skip retry within the same bar. Re-evaluate next bar (t+1) using that bar’s signal. (No pending orders.)
(8) Track PnL each bar (mark-to-mid). Include slippage as difference between execution and mid at trade time. Track equity curve.
(9) Take profit logic:
    - Trigger TP if (zscore + package_yield_delta > take profit soft threshold) OR (package_yield_delta > take profit hard threshold).
    - Compute acceptable cost for TP: (multiplier * expected_pnl_bps - tp_cost_premium_bps) converted via DV01 to price delta.
(10) Exit sizing:
    - Determine exit notionals from current position (fully close).
(11) Apply market validity checks (same as 4).
(12) Apply liquidity execution check (same as 5) and exit only if cost <= acceptable cost.
(13) On TP success:
    - state=TAKE_PROFIT or TAKE_PROFIT_QUARANTINE (if configured)
    - record trade tracker attempt states if TP fails (NO_LIQUIDITY / TOO_WIDE)
(14) Stop-loss / max-holding:
    - Stop-loss trigger if package_yield_delta > stop  loss hard threshold (or other defined rule).
    - Convert (multiplier * expected_pnl_bps - stop_cost_premium_bps) to acceptable cost via DV01.
    - Forced exit: execute by walking the book until fully filled; DO NOT apply cost acceptance check (must exit).
(15) On stop/max-holding exit:
    - state=STOP_LOSS / MAX_HOLDING or *_QUARANTINE if configured
    - update trade tracker with exit reason.
(16) Quarantine:
    - When quarantine timer expires (0 means disabled), return to FLAT.
(17) Always update pnl tracker and trades tracker each bar, including attempt states even when no fill occurs.

What you must deliver:
A) A short audit of current code: list existing modules/classes under src/mlstudy/trading/backtest and map what already exists vs what is missing.
B) Implement missing pieces with minimal churn:
   - A strategy state machine (FLAT/LONG/SHORT + attempt states + quarantine).
   - L2 book-walk execution that returns (filled_qty, vwap_price, fully_filled bool).
   - Cost acceptance checks vs acceptable cost in price delta.
   - Trackers: per-bar PnL/equity arrays and per-trade records (entries/exits + attempt outcomes).
C) Add/Update tests:
   - At least one end-to-end toy test generating synthetic yields, dv01s, and 2-level bid/ask prices+sizes.
   - The test should cover: entry succeeds, TP succeeds, stop-loss forced exit, and a case where entry fails due to liquidity or too-wide spread.
D) Ensure no lookahead:
   - If you compute zscore/beta in the backtester, lag them or execute at next bar.
E) Keep interfaces simple:
   - Input market data should support L2 (bid/ask px+sz per level) plus yields and dv01s.
   - Sizing uses hedge ratios + dv01; output leg sizes to execution.
(F) implementation supports jit

Please implement these changes directly in code and show the key diffs / new files you added.


You are Claude. Please inspect and update the code in:

  src/mlstudy/trading/backtest

Goal: extend the backtester to support a mean-reversion strategy workflow with L2 book execution + explicit state machine + JIT-capable hot loop.

Constraints:
- No post-analysis/reporting; only engine mechanics, execution, state machine, trackers.
- No lookahead: decisions at bar t may only use signal inputs computed from <= t-1 (assume signals passed in are already lagged correctly).
- Signals are inputs (do not build a general signal library).

Portfolio conventions:
- There is a reference instrument (index 0). Hedge ratios r_i are in yield space with r_ref = +1 and sum_i r_i = 0.
- The package is traded as a basket of legs with sizes derived from hedge ratios and DV01.

Inputs (must be supported):
- L2 book: bid_px[t,i,l], bid_sz[t,i,l], ask_px[t,i,l], ask_sz[t,i,l] for levels l=0..L-1
- mid_px[t,i] (or compute from top of book; but validity check uses bid<=mid<=ask)
- dv01[t,i] with explicit units: (PRICE_POINTS per 1bp per 1 unit) OR (CURRENCY per 1bp per 1 unit + contract_multiplier). Choose one and document.
- signals per bar (already computed + lagged):
  - zscore[t]
  - expected_yield_pnl_bps[t]   # DEFINITION: expected yield change of the package residual in bp
  - package_yield_bps[t]  # DEFINITION: current yield value of the package

Sizing:
- Determine basket size by a scalar risk budget, e.g. target_dv01_ref (input config). Set ref leg size so dv01_ref * size_ref = target_dv01_ref, then compute other leg sizes from hedge ratios and dv01 scaling.

Execution / cost:
- Convert acceptable yield cost to acceptable PRICE cost:
    acceptable_yield_bps = multiplier * expected_yield_pnl_bps - cost_premium_yield_bps
    acceptable_price_delta = acceptable_yield_bps * package_dv01_price_per_bp
  Define package_dv01_price_per_bp explicitly (e.g., abs(sum(size_i * dv01_i)) per 1 unit of basket, or ref dv01, etc.) and keep consistent.
- L2 book-walk to get full-fill VWAP; require full fill for entry/TP. For STOP/MAX_HOLDING, force exit (book-walk until full fill; no cost check).

Market validity:
- Check mid within top-of-book: bid0 <= mid <= ask0. Provide option: validate ref-only or all legs.

State machine (must implement):
- FLAT, LONG, SHORT, plus attempt states:
  TRY_LONG_NO_LIQUIDITY, TRY_LONG_TOO_WIDE, TRY_LONG_INVALID_BOOK,
  TRY_SHORT_NO_LIQUIDITY, TRY_SHORT_TOO_WIDE, TRY_SHORT_INVALID_BOOK,
  TRY_TP_NO_LIQUIDITY, TRY_TP_TOO_WIDE, TRY_TP_INVALID_BOOK, etc.
- Quarantine states after TP/SL/TIME exits with configurable cooldown bars; during quarantine block entry.

Workflow:
(0) for t in bars: start with state=FLAT
(1) read signal inputs: zscore[t], expected_yield_pnl_bps[t], package_yield_delta_bps[t]
(2)-(7) Entry: if zscore beyond entry threshold => attempt entry ONCE this bar (skip retry). If cannot fully fill => NO_LIQUIDITY. If fully fill but cost>acceptable => TOO_WIDE. If invalid book => INVALID_BOOK. If success => enter LONG/SHORT and record trade.
(8) Update PnL each bar (mark-t






