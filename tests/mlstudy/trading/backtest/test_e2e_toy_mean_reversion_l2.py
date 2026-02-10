# tests/test_e2e_toy_mean_reversion_l2.py
#
# End-to-end toy test:
# - generate synthetic yield data with a mean-reverting fly residual
# - convert yields -> mid prices
# - build top-2 L2 orderbook (bid/ask prices + sizes)
# - build a simple MR target_positions stream in portfolio space (M=1)
# - map portfolio -> 3 legs (N=3) via weights W[t, m, i]
# - run BacktestEngine and assert basic invariants

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
    """
    Simple OU: dx = -kappa*x*dt + sigma*sqrt(dt)*eps
    """
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


def _zscore(x: np.ndarray, *, lookback: int) -> np.ndarray:
    """
    Rolling z-score with simple expanding warmup (nan until enough obs).
    """
    T = x.shape[0]
    z = np.full(T, np.nan, dtype=np.float64)
    for t in range(lookback - 1, T):
        w = x[t - lookback + 1 : t + 1]
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=1)) if lookback > 1 else float(np.std(w))
        z[t] = 0.0 if sd <= 1e-12 else (x[t] - mu) / sd
    return z


def make_toy_inputs_mean_reversion_l2(
    *,
    T: int = 260,
    N: int = 3,
    L: int = 2,
    seed: int = 7,
) -> tuple[BacktestInputs, BacktestConfig]:
    """
    Create toy BacktestInputs + BacktestConfig:
    - N=3 legs
    - M=1 portfolio (a "fly")
    - L=2 book levels
    """
    assert N == 3, "This toy generator assumes a 3-leg fly."
    assert L == 2, "This toy generator assumes top-2 book levels."

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    # --- timeline (simple int date index) ---
    datetimes_int = np.arange(20240101, 20240101 + T, dtype=np.int64)

    # --- generate yields ---
    # y1, y3 drift slowly (random walk)
    y1 = _rw(T, sigma=0.004, dt=dt, x0=0.025, rng=rng)  # 2.5%
    y3 = _rw(T, sigma=0.004, dt=dt, x0=0.030, rng=rng)  # 3.0%

    # residual is mean-reverting OU
    resid = _ou_process(T, kappa=6.0, sigma=0.010, dt=dt, x0=0.0, rng=rng)

    # y2 is the belly: average of wings + residual
    y2 = 0.5 * (y1 + y3) + resid

    yields = np.stack([y1, y2, y3], axis=1)  # (T, N)

    # --- dv01s (toy constant-ish) ---
    # DV01 per unit position (in "currency per 1bp"). Just keep stable positive values.
    dv01s = np.tile(np.array([0.045, 0.060, 0.050], dtype=np.float64), (T, 1))  # (T, N)

    # --- convert yields to mid prices (toy duration approximation) ---
    # price ≈ 100 - duration*(yield - base_yield)*100
    base_yield = np.array([0.025, 0.028, 0.030], dtype=np.float64)
    duration = np.array([6.0, 7.5, 9.0], dtype=np.float64)
    mid_px = 100.0 - (yields - base_yield) * (duration * 100.0)  # (T, N)

    # --- build top-2 L2 orderbook ---
    # Level 0 = best, Level 1 = next
    # Use small tick-like increments; ensure bid < ask.
    tick = np.array([0.01, 0.01, 0.01], dtype=np.float64)  # per leg
    spread_ticks = 2  # best bid/ask around mid
    half_spread = 0.5 * spread_ticks * tick  # (N,)

    bid0 = mid_px - half_spread
    ask0 = mid_px + half_spread

    # Second level is one tick worse
    bid1 = bid0 - tick
    ask1 = ask0 + tick

    bid_px = np.stack([bid0, bid1], axis=2)  # (T, N, L)
    ask_px = np.stack([ask0, ask1], axis=2)  # (T, N, L)

    # Sizes: larger at top, smaller at level 1; always non-negative
    # Sizes are in "contracts" here.
    base_sz = rng.integers(low=50, high=120, size=(T, N)).astype(np.float64)
    bid_sz = np.stack([base_sz, 0.6 * base_sz], axis=2)
    ask_sz = np.stack([base_sz, 0.6 * base_sz], axis=2)

    # Optional volume proxy (helps if you later enable participation caps)
    traded_volume = (3.0 * base_sz).astype(np.float64)  # (T,N)

    orderbook = OrderBookL2(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        traded_volume=traded_volume,
    )

    market = MarketState(
        mtm_px=None,    # we'll use MID derived by engine/config
        yields=yields,
        dv01s=dv01s,
    )

    # --- portfolio mapping: one portfolio (fly) over 3 legs ---
    # weights: +1 * wing1, -2 * belly, +1 * wing3
    # Shape (T, M, N) where M=1
    W = np.zeros((T, 1, N), dtype=np.float64)
    W[:, 0, :] = np.array([+1.0, -2.0, +1.0], dtype=np.float64)

    mapping = PortfolioMapping(W=W, W_lag_bars=0)

    # --- mean reverting signal -> target portfolio positions (T, M) ---
    # Use z-score of residual (fly yield) = y2 - 0.5*(y1+y3)
    fly = yields[:, 1] - 0.5 * (yields[:, 0] + yields[:, 2])  # (T,)
    z = _zscore(fly, lookback=40)

    entry = 1.25
    exit_ = 0.25
    unit = 10.0  # portfolio units

    target = np.zeros((T, 1), dtype=np.float64)
    pos = 0.0
    for t in range(T):
        if not np.isfinite(z[t]):
            target[t, 0] = 0.0
            pos = 0.0
            continue

        # hysteresis: enter on entry, exit on exit band
        if pos == 0.0:
            if z[t] >= entry:
                pos = -unit
            elif z[t] <= -entry:
                pos = +unit
        else:
            if abs(z[t]) <= exit_:
                pos = 0.0

        target[t, 0] = pos

    controls = Controls(target_positions=target)

    # Tradable everywhere in this toy
    availability = Availability(
        tradable_mask=np.ones((T, N), dtype=np.int8),
    )

    inputs = BacktestInputs(
        datetimes_int=datetimes_int,
        orderbook=orderbook,
        market=market,
        mapping=mapping,
        controls=controls,
        availability=availability,
    )

    # --- config: realistic-ish L2 execution ---
    cfg = BacktestConfig()
    cfg = BacktestConfig(
        instrument=cfg.instrument,
        control=cfg.control,
        timing=cfg.timing,
        execution=cfg.execution,
        partial_fills=cfg.partial_fills,
        rounding=cfg.rounding,
        impact=cfg.impact,
        fees=cfg.fees,
        mapping=cfg.mapping,
        risk=cfg.risk,
        data=cfg.data,
        accounting=cfg.accounting,
        portfolio=cfg.portfolio,
    )

    # set meaningful knobs
    cfg = cfg  # (keep defaults then patch fields below)

    # Patch fields (dataclasses are frozen; we rebuild via replace in scenarios normally.
    # For test simplicity, just construct a new cfg with desired nested configs.)
    from dataclasses import replace

    cfg = replace(cfg,
        control=replace(cfg.control, control_mode="target_positions"),
        mapping=replace(cfg.mapping, mapping_mode="NOTIONAL_WEIGHTS"),
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

    return inputs, cfg


def test_end_to_end_toy_mean_reversion_l2():
    inputs, cfg = make_toy_inputs_mean_reversion_l2(T=260, seed=11)

    engine = BacktestEngine(enable_jit=True)  # uses the numpy placeholder jit_loop in this repo
    res = engine.run(inputs, cfg)

    # ---- shape invariants ----
    T = inputs.datetimes_int.shape[0]
    N = inputs.orderbook.bid_px.shape[1]
    assert res.positions_leg.shape == (T, N)
    assert res.cash.shape == (T,)
    assert res.equity.shape == (T,)
    assert res.pnl.shape == (T,)

    # ---- basic sanity ----
    assert np.all(np.isfinite(res.equity))
    assert np.all(np.isfinite(res.pnl))

    # there should be some trading / fills
    assert res.fills is not None
    assert res.fills.shape[0] > 0

    # MTM from mid prices should create changing equity (not all zeros)
    assert float(np.std(res.equity)) > 0.0

    # Check bid<ask always in the toy data (top of book)
    bid0 = inputs.orderbook.bid_px[:, :, 0]
    ask0 = inputs.orderbook.ask_px[:, :, 0]
    assert np.all(bid0 <= ask0)

    # Ensure we didn't exceed depth too often (some fills could be partial, but should exist)
    filled_nonzero = np.sum(np.abs(res.fills["qty"]) > 0)
    assert filled_nonzero > 0
