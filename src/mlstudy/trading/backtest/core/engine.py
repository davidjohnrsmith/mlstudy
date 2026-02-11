"""
backtest/core/engine.py

High-level orchestration for the backtesting engine.

Design goals
- Engine is strategy-agnostic: it consumes controls (targets/orders) and market data.
- Execution realism is delegated (order book walk, haircuts, partial fills) to JIT loop helpers.
- This module does NOT do post-analysis (Sharpe/drawdown charts). It returns raw results.

Core flow
1) Validate inputs + config compatibility (shapes, required fields)
2) Derive/prepare any missing MTM series (e.g., mid from L2)
3) Pack config (nested dataclasses) into flat arrays/scalars for Numba JIT
4) Call jit_loop.run(...) to produce results
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from .types import BacktestInputs, BacktestConfig
from .packer import pack_config
from .results import BacktestResults
from ..data.validation import validate_inputs_basic


def _derive_mid_from_l2(orderbook) -> np.ndarray:
    """Derive mid price from best bid/ask in L2 book. Returns shape (T, N)."""
    bid0 = orderbook.bid_px[:, :, 0]
    ask0 = orderbook.ask_px[:, :, 0]
    return 0.5 * (bid0 + ask0)


def _get_mtm_px(inputs: BacktestInputs, cfg: BacktestConfig) -> np.ndarray:
    """
    Select/derive mark-to-market price series used for MTM PnL.
    Returns shape (T, N).
    """
    src = cfg.accounting.mtm_source
    if src == "EXTERNAL_MTM":
        if inputs.market.mtm_px is None:
            raise ValueError("accounting.mtm_source='EXTERNAL_MTM' but inputs.market.mtm_px is None")
        return inputs.market.mtm_px

    if src == "LAST":
        if inputs.orderbook.last_px is None:
            raise ValueError("accounting.mtm_source='LAST' but inputs.orderbook.last_px is None")
        return inputs.orderbook.last_px

    # default MID
    return _derive_mid_from_l2(inputs.orderbook)


def _resolve_controls(inputs: BacktestInputs, cfg: BacktestConfig) -> np.ndarray:
    """
    Return controls array with shape (T, M) in the configured control mode.

    - target_positions: desired holdings
    - orders: signed trade sizes
    """
    mode = cfg.control.control_mode
    if mode == "target_positions":
        if inputs.controls.target_positions is None:
            raise ValueError("control_mode='target_positions' but inputs.controls.target_positions is None")
        return inputs.controls.target_positions
    if mode == "orders":
        if inputs.controls.orders is None:
            raise ValueError("control_mode='orders' but inputs.controls.orders is None")
        return inputs.controls.orders
    raise ValueError(f"Unknown control mode: {mode}")


def _resolve_mapping_W(inputs: BacktestInputs, cfg: BacktestConfig) -> Optional[np.ndarray]:
    """
    Return mapping weights W with shape (T, M, N) if mapping is enabled; else None.
    """
    if cfg.mapping.mapping_mode == "NONE":
        return None
    if inputs.mapping.W is None:
        raise ValueError("mapping_mode != 'NONE' but inputs.mapping.W is None")
    return inputs.mapping.W


def _resolve_tradable_mask(inputs: BacktestInputs) -> Optional[np.ndarray]:
    """Prefer tradable_mask; fall back to active_mask; else None."""
    if inputs.availability.tradable_mask is not None:
        return inputs.availability.tradable_mask
    if inputs.availability.active_mask is not None:
        return inputs.availability.active_mask
    return None


class BacktestEngine:
    """
    High-level engine wrapper around the JIT loop.

    Typical usage:
        engine = BacktestEngine()
        results = engine.run(inputs, cfg)
    """

    def __init__(self, *, enable_jit: bool = True):
        self.enable_jit = enable_jit

    def run(self, inputs: BacktestInputs, cfg: BacktestConfig) -> BacktestResults:
        """
        Run the backtest and return raw results (positions, pnl, fills, etc.).
        """
        # ---- 1) Basic validation (shapes, sanity) ----
        validate_inputs_basic(inputs)

        T = int(inputs.datetimes_int.shape[0])
        bid_px = inputs.orderbook.bid_px
        N = int(bid_px.shape[1])

        # Ensure we have controls
        controls = _resolve_controls(inputs, cfg)
        if controls.shape[0] != T:
            raise ValueError(f"Controls T mismatch: controls.shape[0]={controls.shape[0]} vs T={T}")

        # Determine M
        M = int(controls.shape[1])

        # Mapping
        W = _resolve_mapping_W(inputs, cfg)
        if W is not None:
            if W.shape[0] != T or W.shape[1] != M or W.shape[2] != N:
                raise ValueError(
                    f"Mapping W shape must be (T,M,N)=({T},{M},{N}) but got {W.shape}"
                )

        # MTM price series (T,N)
        mtm_px = _get_mtm_px(inputs, cfg)
        if mtm_px.shape != (T, N):
            raise ValueError(f"mtm_px must be shape (T,N)=({T},{N}) but got {mtm_px.shape}")

        # Optional arrays
        yields = inputs.market.yields
        dv01s = inputs.market.dv01s
        if cfg.risk.max_gross_dv01 != np.inf or cfg.risk.max_net_dv01 != np.inf or cfg.accounting.pnl_method == "YIELD_DV01_APPROX":
            if dv01s is None:
                raise ValueError("DV01-based risk or DV01 PnL requested but inputs.market.dv01s is None")

        tradable_mask = _resolve_tradable_mask(inputs)

        # ---- 2) Pack config into JIT-friendly params ----
        packed = pack_config(cfg)

        # ---- 3) Call JIT loop or Python fallback ----
        if self.enable_jit:
            from . import jit_loop
            out = jit_loop.run(
                datetimes_int=inputs.datetimes_int,
                # L2 book
                bid_px=inputs.orderbook.bid_px,
                bid_sz=inputs.orderbook.bid_sz,
                ask_px=inputs.orderbook.ask_px,
                ask_sz=inputs.orderbook.ask_sz,
                # MTM
                mtm_px=mtm_px,
                # optional microstructure
                last_px=inputs.orderbook.last_px,
                traded_volume=inputs.orderbook.traded_volume,
                adv=inputs.orderbook.adv,
                tradable_mask=tradable_mask,
                # rates optional
                yields=yields,
                dv01s=dv01s,
                # controls + mapping
                controls=controls,
                W=W,
                # packed params
                params_float=packed.params_float,
                params_int=packed.params_int,
                flags=packed.flags,
                # init state (portfolio init lives in cfg)
                initial_capital=cfg.portfolio.initial_capital,
                initial_positions=cfg.portfolio.initial_positions,
            )
            return BacktestResults.from_jit(out, datetimes_int=inputs.datetimes_int)

        # Minimal fallback (non-JIT) for debugging only
        return self._run_python_debug(
            inputs=inputs,
            cfg=cfg,
            controls=controls,
            W=W,
            mtm_px=mtm_px,
            tradable_mask=tradable_mask,
        )

    # ---------------------------------------------------------------------
    # Debug fallback: intentionally minimal (not feature-complete).
    # Use only to sanity-check wiring before JIT.
    # ---------------------------------------------------------------------
    def _run_python_debug(
        self,
        *,
        inputs: BacktestInputs,
        cfg: BacktestConfig,
        controls: np.ndarray,
        W: Optional[np.ndarray],
        mtm_px: np.ndarray,
        tradable_mask: Optional[np.ndarray],
    ) -> BacktestResults:
        T = int(inputs.datetimes_int.shape[0])
        N = int(inputs.orderbook.bid_px.shape[1])
        M = int(controls.shape[1])

        # Truth positions in legs
        pos_leg = np.zeros((T, N), dtype=np.float64)
        cash = np.zeros(T, dtype=np.float64)
        equity = np.zeros(T, dtype=np.float64)
        pnl = np.zeros(T, dtype=np.float64)

        cash_prev = float(cfg.portfolio.initial_capital)
        pos_prev = np.zeros(N, dtype=np.float64)

        # Initialize positions if provided (interpret based on store_positions_as)
        if cfg.portfolio.initial_positions is not None:
            init = np.asarray(cfg.portfolio.initial_positions, dtype=np.float64)
            if cfg.portfolio.store_positions_as == "LEGS":
                if init.shape[0] != N:
                    raise ValueError("initial_positions expected shape (N,) when store_positions_as='LEGS'")
                pos_prev = init.copy()
            else:
                # portfolio-space: expand to legs if mapping enabled
                if init.shape[0] != M:
                    raise ValueError("initial_positions expected shape (M,) when store_positions_as='PORTFOLIO'")
                if cfg.mapping.mapping_mode == "NONE":
                    raise ValueError("store_positions_as='PORTFOLIO' requires mapping_mode != 'NONE'")
                pos_prev = init @ W[0]  # (M,) @ (M,N) -> (N,)

        # Mark-to-market convention: mid changes; no execution realism in debug mode.
        mtm0 = mtm_px[0]
        equity_prev = cash_prev + float(np.dot(pos_prev, mtm0) * cfg.instrument.contract_multiplier)

        for t in range(T):
            # tradability gate (very simple)
            if tradable_mask is not None and t < tradable_mask.shape[0]:
                if not np.all(tradable_mask[t].astype(bool)):
                    # skip trading; still MTM
                    target_leg = pos_prev
                else:
                    target_leg = self._compute_target_leg_positions(cfg, controls, W, t, pos_prev, N, M)
            else:
                target_leg = self._compute_target_leg_positions(cfg, controls, W, t, pos_prev, N, M)

            # Execute instantly at mtm price (debug)
            trade = target_leg - pos_prev
            exec_px = mtm_px[t]
            trade_value = float(np.dot(trade, exec_px) * cfg.instrument.contract_multiplier)

            # fees as bps of notional is ignored in debug; keep cash update simple
            cash_now = cash_prev - trade_value

            # MTM
            mtm_now = mtm_px[t]
            equity_now = cash_now + float(np.dot(target_leg, mtm_now) * cfg.instrument.contract_multiplier)

            # PnL vs previous equity
            pnl_t = equity_now - equity_prev

            pos_leg[t] = target_leg
            cash[t] = cash_now
            equity[t] = equity_now
            pnl[t] = pnl_t

            # roll forward
            pos_prev = target_leg
            cash_prev = cash_now
            equity_prev = equity_now

        return BacktestResults(
            datetimes_int=inputs.datetimes_int,
            positions_leg=pos_leg,
            cash=cash,
            equity=equity,
            pnl=pnl,
            fills=None,
            meta={"engine": "python_debug"},
        )

    def _compute_target_leg_positions(
        self,
        cfg: BacktestConfig,
        controls: np.ndarray,
        W: Optional[np.ndarray],
        t: int,
        pos_prev: np.ndarray,
        N: int,
        M: int,
    ) -> np.ndarray:
        """
        Convert controls into target leg holdings for this debug path.
        - In target_positions mode: controls[t] is desired portfolio or leg holdings
        - In orders mode: controls[t] is trade size (portfolio or legs)
        """
        mapping_mode = cfg.mapping.mapping_mode
        control_mode = cfg.control.control_mode

        if mapping_mode == "NONE":
            # controls operate directly on legs (M should equal N)
            if M != N:
                raise ValueError("mapping_mode='NONE' requires controls dimension M == N")
            if control_mode == "target_positions":
                return controls[t].astype(np.float64, copy=False)
            else:
                return pos_prev + controls[t].astype(np.float64, copy=False)

        # mapping enabled: controls are in portfolio-space (M), expand to legs using W[t]
        if W is None:
            raise ValueError("mapping enabled but W is None")
        Wt = W[t]  # (M, N)

        if control_mode == "target_positions":
            q = controls[t].astype(np.float64, copy=False)   # (M,)
        else:
            # orders: add to implied portfolio position (not tracked in this debug path)
            # For debug only, treat order as direct change in portfolio holdings from zero baseline.
            # Real implementation should track portfolio-space state or re-derive from legs.
            q = controls[t].astype(np.float64, copy=False)

        # portfolio units -> leg units
        # (M,) @ (M,N) -> (N,)
        return q @ Wt
