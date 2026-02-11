
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
from tests.mlstudy.trading.backtest.test_e2e_toy_mean_reversion_l2 import make_toy_inputs_regression_hedge_ratio_l2


def _available_qty_from_l2(
    *,
    side: int,
    bid_sz_levels: np.ndarray,  # (L,)
    ask_sz_levels: np.ndarray,  # (L,)
    max_levels: int,
    haircut: float,
) -> float:
    """
    Available qty you can *reasonably* take from the book on the given side,
    using sum(level sizes) * haircut across up to max_levels.
    """
    L = int(bid_sz_levels.shape[0])
    k = min(L, int(max_levels))
    if side > 0:  # BUY hits ask
        return float(np.sum(ask_sz_levels[:k]) * haircut)
    else:         # SELL hits bid
        return float(np.sum(bid_sz_levels[:k]) * haircut)


def build_targets_implementability_gated(
    *,
    z: np.ndarray,                 # (T,)
    beta: np.ndarray,              # (T,)
    dv01s: np.ndarray,             # (T,N)
    bid_sz: np.ndarray,            # (T,N,L)
    ask_sz: np.ndarray,            # (T,N,L)
    entry: float,
    exit_: float,
    R_dv01: float,
    # execution assumptions for "fillable" check
    size_haircut: float,
    max_levels_to_cross: int,
    require_full_basket: bool = True,
) -> np.ndarray:
    """
    Implementability-gated targets in leg space (T,N) for a 2-leg switch:
    - legs used: front=0 and back=2
    - belly=1 stays 0
    - only enter if BOTH legs are fillable for the *entry trade* under book depth assumption
    - exit: attempt to flatten; if require_full_basket and cannot fully exit, then hold (no flip)
    """
    T = int(z.shape[0])
    N = int(dv01s.shape[1])

    tgt = np.zeros((T, N), dtype=np.float64)
    pos = np.zeros(N, dtype=np.float64)

    state = 0  # -1 short resid, +1 long resid, 0 flat

    for t in range(T):
        # hold previous by default
        tgt[t] = pos

        if not np.isfinite(z[t]) or not np.isfinite(beta[t]):
            # if signal missing: try to exit (flatten) but only if feasible, else hold
            desired = -pos
            if np.allclose(desired, 0.0):
                state = 0
                continue

            # feasibility for exit trade
            ok_exit = True
            for leg in (0, 2):
                q = float(desired[leg])
                if q == 0:
                    continue
                side = 1 if q > 0 else -1
                avail = _available_qty_from_l2(
                    side=side,
                    bid_sz_levels=bid_sz[t, leg],
                    ask_sz_levels=ask_sz[t, leg],
                    max_levels=max_levels_to_cross,
                    haircut=size_haircut,
                )
                if abs(q) > avail + 1e-9:
                    ok_exit = False
                    break

            if ok_exit or not require_full_basket:
                pos[:] = 0.0
                tgt[t] = pos
                state = 0
            continue

        # signal-driven desired state
        desired_state = state
        if state == 0:
            if z[t] >= entry:
                desired_state = -1
            elif z[t] <= -entry:
                desired_state = +1
        else:
            if abs(z[t]) <= exit_:
                desired_state = 0

        # If no change, keep holding
        if desired_state == state:
            continue

        # Build desired target positions for the new state
        if desired_state == 0:
            desired_pos = np.zeros(N, dtype=np.float64)
        else:
            dv01_front = float(dv01s[t, 0])
            dv01_back = float(dv01s[t, 2])
            b = float(beta[t])

            q_front = (R_dv01 / dv01_front) * float(desired_state)
            q_back = b * (dv01_front / dv01_back) * q_front

            desired_pos = np.zeros(N, dtype=np.float64)
            desired_pos[0] = q_front
            desired_pos[1] = 0.0
            desired_pos[2] = -q_back

        # Trade needed
        trade = desired_pos - pos

        # Feasibility check for the basket trade
        ok = True
        for leg in (0, 2):
            q = float(trade[leg])
            if q == 0:
                continue
            side = 1 if q > 0 else -1
            avail = _available_qty_from_l2(
                side=side,
                bid_sz_levels=bid_sz[t, leg],
                ask_sz_levels=ask_sz[t, leg],
                max_levels=max_levels_to_cross,
                haircut=size_haircut,
            )
            if abs(q) > avail + 1e-9:
                ok = False
                break

        if ok or not require_full_basket:
            # accept transition
            pos = desired_pos
            tgt[t] = pos
            state = desired_state
        else:
            # reject transition; hold
            tgt[t] = pos

    return tgt


def test_end_to_end_toy_regression_hedge_ratio_l2_implementability_gated():
    """
    Compare two runs on the same toy market:
    - signal-driven targets (always "intend" to enter)
    - implementability-gated targets (only enter if book depth can support entry basket)
    We inject a liquidity drought on the back leg to force gating to matter.
    """
    inputs, cfg, dbg = make_toy_inputs_regression_hedge_ratio_l2(T=260, seed=42)

    # --- inject tradability stress: reduce depth on back leg (leg=2) on some days ---
    # This makes entries harder (especially when hedge size is large).
    ob = inputs.orderbook
    bid_sz = ob.bid_sz.copy()
    ask_sz = ob.ask_sz.copy()

    rng = np.random.default_rng(123)
    drought_days = rng.choice(np.arange(80, 240), size=40, replace=False)  # avoid warmup
    drought_days.sort()

    # crush sizes at both levels on leg 2
    bid_sz[drought_days, 2, :] *= 0.05
    ask_sz[drought_days, 2, :] *= 0.05

    inputs_stressed = BacktestInputs(
        datetimes_int=inputs.datetimes_int,
        orderbook=OrderBookL2(
            bid_px=ob.bid_px,
            bid_sz=bid_sz,
            ask_px=ob.ask_px,
            ask_sz=ask_sz,
            last_px=ob.last_px,
            last_sz=ob.last_sz,
            traded_volume=ob.traded_volume,
            adv=ob.adv,
        ),
        market=inputs.market,
        mapping=inputs.mapping,
        controls=inputs.controls,  # placeholder; we'll override below
        availability=inputs.availability,
    )

    # --- Run 1: SIGNAL-DRIVEN (the existing precomputed targets from dbg) ---
    inputs_signal = BacktestInputs(
        datetimes_int=inputs_stressed.datetimes_int,
        orderbook=inputs_stressed.orderbook,
        market=inputs_stressed.market,
        mapping=inputs_stressed.mapping,
        controls=Controls(target_positions=dbg["target_leg"]),  # same targets, ignores drought
        availability=inputs_stressed.availability,
    )
    engine = BacktestEngine(enable_jit=True)
    res_signal = engine.run(inputs_signal, cfg)

    # --- Run 2: IMPLEMENTABILITY-GATED targets ---
    tgt_gate = build_targets_implementability_gated(
        z=dbg["z"],
        beta=dbg["beta_clipped"],
        dv01s=inputs_stressed.market.dv01s,
        bid_sz=inputs_stressed.orderbook.bid_sz,
        ask_sz=inputs_stressed.orderbook.ask_sz,
        entry=1.25,
        exit_=0.25,
        R_dv01=50.0,
        size_haircut=cfg.execution.size_haircut,
        max_levels_to_cross=cfg.execution.max_levels_to_cross,
        require_full_basket=True,
    )
    inputs_gate = BacktestInputs(
        datetimes_int=inputs_stressed.datetimes_int,
        orderbook=inputs_stressed.orderbook,
        market=inputs_stressed.market,
        mapping=inputs_stressed.mapping,
        controls=Controls(target_positions=tgt_gate),
        availability=inputs_stressed.availability,
    )
    res_gate = engine.run(inputs_gate, cfg)

    # --- Assertions: gated strategy should trade less (fewer position changes / fewer filled qty) ---
    # 1) It should have fewer non-zero target days
    nz_signal = int(np.sum(np.abs(dbg["target_leg"][:, 0]) > 0))
    nz_gate = int(np.sum(np.abs(tgt_gate[:, 0]) > 0))
    assert nz_gate <= nz_signal

    # 2) It should have fewer filled trades (nonzero fill qty) under drought
    fills_signal = 0 if res_signal.fills is None else int(np.sum(np.abs(res_signal.fills["qty"]) > 0))
    fills_gate = 0 if res_gate.fills is None else int(np.sum(np.abs(res_gate.fills["qty"]) > 0))
    assert fills_gate <= fills_signal

    # 3) Both runs should be numerically sane
    assert np.all(np.isfinite(res_signal.equity))
    assert np.all(np.isfinite(res_gate.equity))
    assert float(np.std(res_signal.equity)) > 0.0
    assert float(np.std(res_gate.equity)) > 0.0
