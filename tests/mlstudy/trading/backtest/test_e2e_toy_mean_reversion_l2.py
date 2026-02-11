# tests/test_e2e_toy_regression_hedge_ratio_l2.py
#
# End-to-end toy test (Option A):
# - Generate synthetic yields for N=3 legs (daily bars)
# - Compute a DAILY (per-bar) regression hedge ratio in yield-change space:
#     Δy_front ~ beta * Δy_back
#   beta is constant within the day/bar but changes day-to-day (rolling OLS, lagged)
# - Build a mean-reversion signal on the hedged residual in yield level:
#     resid_t = y_front_t - beta_t * y_back_t
# - Convert hedge ratio + DV01 into executable leg sizes (contracts):
#     q_back = beta_t * (DV01_front / DV01_back) * q_front
#   with q_front set by a DV01 risk budget
# - Build top-2 L2 order book (bid/ask levels + sizes) from synthetic mid prices
# - Feed leg-level target_positions (T,N) into the backtester (mapping_mode="NONE")
#
# This test validates:
# - end-to-end wiring works (engine runs, produces fills/equity)
# - hedge ratio series varies across days
# - leg sizes are generated from yield-space hedge ratios + DV01
# - L2 execution + MTM behave sanely

from __future__ import annotations

import numpy as np

from mlstudy.trading.backtest.core.engine import BacktestEngine
from mlstudy.trading.backtest.core.types import (
    BacktestInputs,
    BacktestConfig,
    OrderBookL2,
    MarketState,
    Controls,
    PortfolioMapping,
    Availability,
)


def _ou_process(T: int, *, kappa: float, sigma: float, dt: float, x0: float = 0.0, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(T, dtype=np.float64)
    x[0] = x0
    for t in range(1, T):
        eps = rng.normal()
        x[t] = x[t - 1] + (-kappa * x[t - 1]) * dt + sigma * np.sqrt(dt) * eps
    return x


def _rw(T: int, *, sigma: float, dt: float, x0: float, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(T, dtype=np.float64)
    x[0] = x0
    for t in range(1, T):
        x[t] = x[t - 1] + sigma * np.sqrt(dt) * rng.normal()
    return x


def _rolling_beta_dy(
    dy_x: np.ndarray,
    dy_h: np.ndarray,
    *,
    T_yields: int,
    lookback: int,
    ridge: float = 1e-12,
) -> np.ndarray:
    """
    Rolling OLS beta for dy_x ~ beta * dy_h (no intercept).
    Outputs beta aligned to yield index t with shape (T_yields,).

    Alignment:
      dy arrays correspond to yield transitions: dy[k] = y[k+1]-y[k] for k=0..T-2.
      beta[t] is computed using dy window ending at transition (t-1), i.e. using dy[t-lookback : t].
      Therefore beta[0..lookback-1] are NaN; beta[t] valid for t in [lookback, T-1].
    """
    if dy_x.shape != dy_h.shape:
        raise ValueError(f"dy_x and dy_h must have same shape, got {dy_x.shape} vs {dy_h.shape}")
    if dy_x.shape[0] != T_yields - 1:
        raise ValueError(f"Expected dy length {T_yields-1}, got {dy_x.shape[0]}")

    beta = np.full(T_yields, np.nan, dtype=np.float64)

    # t indexes yields, dy slice is dy[t-lookback : t]
    # last valid t is T_yields-1, which maps to dy up to index T_yields-2 (exists)
    for t in range(lookback, T_yields):
        a = dy_h[t - lookback : t]
        b = dy_x[t - lookback : t]
        den = float(np.dot(a, a)) + float(ridge)
        num = float(np.dot(a, b))
        beta[t] = num / den

    return beta

def _zscore(x: np.ndarray, *, lookback: int) -> np.ndarray:
    T = x.shape[0]
    z = np.full(T, np.nan, dtype=np.float64)
    for t in range(lookback - 1, T):
        w = x[t - lookback + 1 : t + 1]
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=1)) if lookback > 1 else float(np.std(w))
        z[t] = 0.0 if sd <= 1e-12 else (x[t] - mu) / sd
    return z


def make_toy_inputs_regression_hedge_ratio_l2(
    *,
    T: int = 260,
    N: int = 3,
    L: int = 2,
    seed: int = 11,
) -> tuple[BacktestInputs, BacktestConfig, dict]:
    """
    Returns (inputs, cfg, debug) where debug contains toy hedge-ratio series etc.
    """
    assert N == 3, "This toy generator assumes N=3 legs."
    assert L == 2, "This toy generator assumes top-2 L2 levels."

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    datetimes_int = np.arange(20240101, 20240101 + T, dtype=np.int64)

    # --- yields ---
    # front and back are random walks with some co-movement
    common = _rw(T, sigma=0.003, dt=dt, x0=0.0, rng=rng)
    y_front = 0.020 + common + _rw(T, sigma=0.002, dt=dt, x0=0.0, rng=rng)
    y_back = 0.030 + 0.8 * common + _rw(T, sigma=0.002, dt=dt, x0=0.0, rng=rng)

    # belly is unused in the switch, but keep it realistic-ish
    mr_belly = _ou_process(T, kappa=4.0, sigma=0.008, dt=dt, x0=0.0, rng=rng)
    y_belly = 0.025 + 0.5 * (y_front + y_back) + mr_belly

    yields = np.stack([y_front, y_belly, y_back], axis=1)  # (T,N)

    # --- DV01 per unit (toy constants) ---
    dv01s = np.tile(np.array([0.050, 0.060, 0.080], dtype=np.float64), (T, 1))  # (T,N)

    # --- regression hedge ratio in yield-change space (daily, lagged) ---
    # dy in bp for stability
    dy_front_bp = np.diff(y_front) * 1e4  # (T-1,)
    dy_back_bp = np.diff(y_back) * 1e4    # (T-1,)

    beta = _rolling_beta_dy(
        dy_x=dy_front_bp,
        dy_h=dy_back_bp,
        T_yields=T,
        lookback=60,
        ridge=1e-6,
    ) # (T,) with NaNs early

    # clamp betas to avoid insane hedge sizes
    beta_clipped = beta.copy()
    ok = np.isfinite(beta_clipped)
    beta_clipped[ok] = np.clip(beta_clipped[ok], -3.0, +3.0)

    # --- mean-reversion signal on hedged residual (yield level) ---
    # resid_t = y_front_t - beta_t * y_back_t
    resid = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        if np.isfinite(beta_clipped[t]):
            resid[t] = y_front[t] - beta_clipped[t] * y_back[t]

    z = _zscore(resid, lookback=40)

    # --- translate to leg target sizes (Option A, leg-space controls) ---
    # Choose a DV01 risk budget R on the "front" leg, then derive hedge size for back:
    #   q_back = beta * (DV01_front / DV01_back) * q_front
    #
    # Direction:
    # - If resid z-score is high (front rich vs hedged back) => short resid:
    #     short front, long hedge back
    # - If resid z-score is low => long resid:
    #     long front, short hedge back
    entry = 1.25
    exit_ = 0.25
    R_dv01 = 50.0  # "currency per bp" DV01 budget on front leg per trade state

    target_leg = np.zeros((T, N), dtype=np.float64)
    state = 0  # -1 short resid, +1 long resid, 0 flat

    for t in range(T):
        if not np.isfinite(z[t]) or not np.isfinite(beta_clipped[t]):
            # if missing beta/z, go flat
            state = 0
            continue

        if state == 0:
            if z[t] >= entry:
                state = -1
            elif z[t] <= -entry:
                state = +1
        else:
            if abs(z[t]) <= exit_:
                state = 0

        if state == 0:
            continue

        dv01_front = float(dv01s[t, 0])
        dv01_back = float(dv01s[t, 2])
        b = float(beta_clipped[t])

        # Base size on front from DV01 budget
        q_front = (R_dv01 / dv01_front) * float(state)  # signed
        # Hedge back size from yield-space beta + DV01 ratio
        q_back = b * (dv01_front / dv01_back) * q_front

        # Put into legs: [front, belly, back]
        target_leg[t, 0] = q_front
        target_leg[t, 1] = 0.0
        target_leg[t, 2] = -q_back  # minus because hedge is opposite side of residual definition

    # --- convert yields -> mid prices (toy duration approximation) ---
    base_yield = np.array([0.020, 0.025, 0.030], dtype=np.float64)
    duration = np.array([6.0, 7.5, 9.0], dtype=np.float64)
    mid_px = 100.0 - (yields - base_yield) * (duration * 100.0)

    # --- build L2 top-2 book around mid ---
    tick = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    half_spread = 0.5 * 2 * tick  # 2 ticks wide spread

    bid0 = mid_px - half_spread
    ask0 = mid_px + half_spread
    bid1 = bid0 - tick
    ask1 = ask0 + tick

    bid_px = np.stack([bid0, bid1], axis=2)  # (T,N,L)
    ask_px = np.stack([ask0, ask1], axis=2)

    base_sz = rng.integers(low=60, high=140, size=(T, N)).astype(np.float64)
    bid_sz = np.stack([base_sz, 0.5 * base_sz], axis=2)
    ask_sz = np.stack([base_sz, 0.5 * base_sz], axis=2)
    traded_volume = (4.0 * base_sz).astype(np.float64)

    orderbook = OrderBookL2(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        traded_volume=traded_volume,
    )

    market = MarketState(
        yields=yields,
        dv01s=dv01s,
        mtm_px=None,  # use MID from book
    )

    # --- controls are leg-level (T,N) => mapping_mode NONE ---
    controls = Controls(target_positions=target_leg)

    availability = Availability(tradable_mask=np.ones((T, N), dtype=np.int8))

    inputs = BacktestInputs(
        datetimes_int=datetimes_int,
        orderbook=orderbook,
        market=market,
        mapping=PortfolioMapping(W=None, W_lag_bars=0),
        controls=controls,
        availability=availability,
    )

    # --- config: mapping disabled, L2 orderbook walk ---
    from dataclasses import replace

    cfg = BacktestConfig()
    cfg = replace(cfg,
        control=replace(cfg.control, control_mode="target_positions"),
        mapping=replace(cfg.mapping, mapping_mode="NONE"),
        execution=replace(cfg.execution,
            mode="ORDERBOOK_WALK",
            size_haircut=0.8,
            max_levels_to_cross=2,
            reject_if_insufficient_depth=False,
            participation_cap=None,
        ),
        partial_fills=replace(cfg.partial_fills, allow_partial_fills=True, carry_unfilled=False),
        rounding=replace(cfg.rounding, min_trade_size=0.0, round_to_size=1.0, netting="NET"),
        fees=replace(cfg.fees, fee_bps=0.2, fee_mode="PER_NOTIONAL", fee_per_unit=0.0),
        impact=replace(cfg.impact, mode="NONE", k_bps=0.0, noise_bps_std=0.0),
        accounting=replace(cfg.accounting, mtm_source="MID", pnl_method="PRICE_MTM"),
        portfolio=replace(cfg.portfolio, initial_capital=0.0, initial_positions=None, store_positions_as="LEGS"),
    )

    debug = {
        "beta": beta,
        "beta_clipped": beta_clipped,
        "resid": resid,
        "z": z,
        "target_leg": target_leg,
    }
    return inputs, cfg, debug


def test_end_to_end_toy_regression_hedge_ratio_l2():
    inputs, cfg, dbg = make_toy_inputs_regression_hedge_ratio_l2(T=260, seed=42)

    engine = BacktestEngine(enable_jit=True)
    res = engine.run(inputs, cfg)

    # ---- shapes ----
    T = inputs.datetimes_int.shape[0]
    N = inputs.orderbook.bid_px.shape[1]
    assert res.positions_leg.shape == (T, N)
    assert res.cash.shape == (T,)
    assert res.equity.shape == (T,)
    assert res.pnl.shape == (T,)

    # ---- hedge ratio "data": constant within a day/bar, changing daily ----
    beta = dbg["beta_clipped"]
    finite = np.isfinite(beta)
    assert finite.sum() > 0
    # it should vary over time (not constant)
    assert float(np.nanstd(beta)) > 0.0

    # ---- sanity: book top always bid <= ask ----
    bid0 = inputs.orderbook.bid_px[:, :, 0]
    ask0 = inputs.orderbook.ask_px[:, :, 0]
    assert np.all(bid0 <= ask0)

    # ---- there should be trading/fills ----
    assert res.fills is not None
    assert res.fills.shape[0] > 0
    assert int(np.sum(np.abs(res.fills["qty"]) > 0)) > 0

    # ---- equity should move (MTM + trading) ----
    assert np.all(np.isfinite(res.equity))
    assert float(np.std(res.equity)) > 0.0

    # ---- exposure check: belly should stay ~0 in targets (this is a switch demo) ----
    tgt = dbg["target_leg"]
    assert float(np.max(np.abs(tgt[:, 1]))) == 0.0

    # ---- DV01-consistency check (rough): hedge size should be related to beta and DV01 ratio ----
    # For days when we have a nonzero position, verify the ratio is finite and broadly consistent.
    dv01s = inputs.market.dv01s
    assert dv01s is not None

    idx = np.where((np.abs(tgt[:, 0]) > 0) & np.isfinite(beta))[0]
    if idx.size > 5:
        t0 = int(idx[idx.size // 2])
        q_front = tgt[t0, 0]
        q_back = -tgt[t0, 2]  # our hedge definition above
        b = beta[t0]
        dv01_front = float(dv01s[t0, 0])
        dv01_back = float(dv01s[t0, 2])

        # expected: q_back ≈ b*(dv01_front/dv01_back)*q_front
        rhs = b * (dv01_front / dv01_back) * q_front
        # allow slack due to rounding to 1.0
        assert np.isfinite(rhs)
        assert abs(q_back - rhs) <= 1.5  # rounding tolerance
