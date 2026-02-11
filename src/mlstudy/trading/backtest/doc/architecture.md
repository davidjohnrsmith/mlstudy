
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
