"""
backtest/data/validation.py

Input validation utilities:
- shape checks for L2 order book inputs (T,N,L)
- consistency checks for market arrays (T,N)
- controls vs mapping dimension checks (T,M) and W (T,M,N)
- basic numeric sanity (finite prices, non-negative sizes, bid<=ask)

These checks are intentionally strict. You can relax them if your venue has quirks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import BacktestInputs, BacktestConfig


def _require_ndarray(name: str, x: object) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(x)}")
    return x


def _check_same_shape(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {name_a}{a.shape} vs {name_b}{b.shape}")


def _check_finite(name: str, x: np.ndarray, *, allow_nan: bool = False) -> None:
    if allow_nan:
        bad = ~np.isfinite(x) & ~np.isnan(x)
    else:
        bad = ~np.isfinite(x)
    if np.any(bad):
        idx = np.argwhere(bad)
        raise ValueError(f"{name} has non-finite values at first bad index {tuple(idx[0])}")


def _check_nonneg(name: str, x: np.ndarray) -> None:
    if np.any(x < 0):
        idx = np.argwhere(x < 0)
        raise ValueError(f"{name} has negative values at first bad index {tuple(idx[0])}")


def _infer_TNL_from_book(inputs: BacktestInputs) -> tuple[int, int, int]:
    bid_px = _require_ndarray("orderbook.bid_px", inputs.orderbook.bid_px)
    if bid_px.ndim != 3:
        raise ValueError(f"orderbook.bid_px must be (T,N,L) but got ndim={bid_px.ndim}, shape={bid_px.shape}")
    T, N, L = bid_px.shape
    return int(T), int(N), int(L)


def validate_inputs_basic(inputs: BacktestInputs) -> None:
    """
    Basic validation that does not require BacktestConfig.
    """
    dt = _require_ndarray("datetimes_int", inputs.datetimes_int)
    if dt.ndim != 1:
        raise ValueError(f"datetimes_int must be (T,) but got shape {dt.shape}")
    if dt.size == 0:
        raise ValueError("datetimes_int is empty")

    T, N, L = _infer_TNL_from_book(inputs)

    # Book arrays
    bid_px = _require_ndarray("orderbook.bid_px", inputs.orderbook.bid_px)
    bid_sz = _require_ndarray("orderbook.bid_sz", inputs.orderbook.bid_sz)
    ask_px = _require_ndarray("orderbook.ask_px", inputs.orderbook.ask_px)
    ask_sz = _require_ndarray("orderbook.ask_sz", inputs.orderbook.ask_sz)

    for name, arr in [("bid_px", bid_px), ("bid_sz", bid_sz), ("ask_px", ask_px), ("ask_sz", ask_sz)]:
        if arr.shape != (T, N, L):
            raise ValueError(f"orderbook.{name} must be (T,N,L)=({T},{N},{L}) but got {arr.shape}")

    _check_finite("orderbook.bid_px", bid_px)
    _check_finite("orderbook.ask_px", ask_px)
    _check_nonneg("orderbook.bid_sz", bid_sz)
    _check_nonneg("orderbook.ask_sz", ask_sz)

    # Top-of-book sanity: bid0 <= ask0 (allow equality)
    bid0 = bid_px[:, :, 0]
    ask0 = ask_px[:, :, 0]
    if np.any(bid0 > ask0):
        idx = np.argwhere(bid0 > ask0)
        t, i = int(idx[0, 0]), int(idx[0, 1])
        raise ValueError(
            f"Top-of-book crossed at t={t}, inst={i}: bid0={bid0[t,i]} > ask0={ask0[t,i]}"
        )

    # Optional book extras
    for opt_name, opt_arr, shape in [
        ("orderbook.last_px", inputs.orderbook.last_px, (T, N)),
        ("orderbook.last_sz", inputs.orderbook.last_sz, (T, N)),
        ("orderbook.traded_volume", inputs.orderbook.traded_volume, (T, N)),
        ("orderbook.adv", inputs.orderbook.adv, (T, N)),
    ]:
        if opt_arr is not None:
            arr = _require_ndarray(opt_name, opt_arr)
            if arr.shape != shape:
                raise ValueError(f"{opt_name} must be shape {shape} but got {arr.shape}")
            if "px" in opt_name:
                _check_finite(opt_name, arr, allow_nan=True)
            else:
                _check_nonneg(opt_name, arr)

    # Optional market arrays
    for opt_name, opt_arr in [
        ("market.mtm_px", inputs.market.mtm_px),
        ("market.yields", inputs.market.yields),
        ("market.dv01s", inputs.market.dv01s),
        ("market.accrual_factors", inputs.market.accrual_factors),
    ]:
        if opt_arr is not None:
            arr = _require_ndarray(opt_name, opt_arr)
            if arr.shape != (T, N):
                raise ValueError(f"{opt_name} must be (T,N)=({T},{N}) but got {arr.shape}")
            _check_finite(opt_name, arr, allow_nan=True)

    for opt_name, opt_arr in [
        ("market.coupons", inputs.market.coupons),
    ]:
        if opt_arr is not None:
            arr = _require_ndarray(opt_name, opt_arr)
            if arr.ndim == 1:
                if arr.shape != (N,):
                    raise ValueError(f"{opt_name} must be (N,)={(N,)} but got {arr.shape}")
            elif arr.ndim == 2:
                if arr.shape != (T, N):
                    raise ValueError(f"{opt_name} must be (T,N)=({T},{N}) but got {arr.shape}")
            else:
                raise ValueError(f"{opt_name} must be (N,) or (T,N), got shape {arr.shape}")
            _check_finite(opt_name, arr, allow_nan=True)

    if inputs.market.repo_rates is not None:
        rr = _require_ndarray("market.repo_rates", inputs.market.repo_rates)
        if rr.ndim == 1 and rr.shape != (T,):
            raise ValueError(f"market.repo_rates must be (T,)={(T,)} or (T,N), got {rr.shape}")
        if rr.ndim == 2 and rr.shape != (T, N):
            raise ValueError(f"market.repo_rates must be (T,N)=({T},{N}), got {rr.shape}")
        if rr.ndim not in (1, 2):
            raise ValueError(f"market.repo_rates must be (T,) or (T,N), got {rr.shape}")
        _check_finite("market.repo_rates", rr, allow_nan=True)

    # Availability masks
    for opt_name, opt_arr in [
        ("availability.active_mask", inputs.availability.active_mask),
        ("availability.tradable_mask", inputs.availability.tradable_mask),
    ]:
        if opt_arr is not None:
            arr = _require_ndarray(opt_name, opt_arr)
            if arr.shape != (T, N):
                raise ValueError(f"{opt_name} must be (T,N)=({T},{N}) but got {arr.shape}")

    # Controls
    tp = inputs.controls.target_positions
    od = inputs.controls.orders
    if tp is not None and od is not None:
        raise ValueError("Provide only one of controls.target_positions or controls.orders, not both.")
    if tp is None and od is None:
        # allowed if lifecycle will generate controls externally; still warn here via exception to be strict
        raise ValueError("No controls provided: set controls.target_positions or controls.orders.")

    ctrl = tp if tp is not None else od
    ctrl = _require_ndarray("controls", ctrl)
    if ctrl.ndim != 2:
        raise ValueError(f"controls must be (T,M) but got shape {ctrl.shape}")
    if ctrl.shape[0] != T:
        raise ValueError(f"controls first dim must match T={T} but got {ctrl.shape[0]}")

    # Mapping W (optional)
    if inputs.mapping.W is not None:
        W = _require_ndarray("mapping.W", inputs.mapping.W)
        if W.ndim != 3:
            raise ValueError(f"mapping.W must be (T,M,N) but got shape {W.shape}")
        if W.shape[0] != T or W.shape[2] != N:
            raise ValueError(f"mapping.W must have shape (T,M,N)=({T},M,{N}), got {W.shape}")
        # M should match controls second dim
        if W.shape[1] != ctrl.shape[1]:
            raise ValueError(f"mapping.W M={W.shape[1]} does not match controls M={ctrl.shape[1]}")


def validate_config_compat(inputs: BacktestInputs, cfg: BacktestConfig) -> None:
    """
    Checks that inputs contain required fields given cfg choices.
    Call this from engine.py (after validate_inputs_basic).
    """
    # Control mode vs which controls array exists
    if cfg.control.control_mode == "target_positions" and inputs.controls.target_positions is None:
        raise ValueError("cfg.control_mode='target_positions' requires inputs.controls.target_positions")
    if cfg.control.control_mode == "orders" and inputs.controls.orders is None:
        raise ValueError("cfg.control_mode='orders' requires inputs.controls.orders")

    # Execution mode requirements
    if cfg.execution.mode == "LAST" and inputs.orderbook.last_px is None:
        raise ValueError("execution.mode='LAST' requires inputs.orderbook.last_px")

    # Participation cap needs volume source if enabled
    if cfg.execution.participation_cap is not None:
        if inputs.orderbook.traded_volume is None and inputs.orderbook.adv is None:
            raise ValueError("execution.participation_cap requires orderbook.traded_volume or orderbook.adv")

    # MTM source requirements
    if cfg.accounting.mtm_source == "EXTERNAL_MTM" and inputs.market.mtm_px is None:
        raise ValueError("accounting.mtm_source='EXTERNAL_MTM' requires market.mtm_px")
    if cfg.accounting.mtm_source == "LAST" and inputs.orderbook.last_px is None:
        raise ValueError("accounting.mtm_source='LAST' requires orderbook.last_px")

    # DV01 requirements
    dv01_needed = (
        np.isfinite(cfg.risk.max_gross_dv01)
        or np.isfinite(cfg.risk.max_net_dv01)
        or cfg.accounting.pnl_method == "YIELD_DV01_APPROX"
        or cfg.mapping.mapping_mode == "DV01_WEIGHTS"
    )
    if dv01_needed and inputs.market.dv01s is None:
        raise ValueError("DV01-related config enabled but inputs.market.dv01s is None")

    # Mapping requirements
    if cfg.mapping.mapping_mode != "NONE" and inputs.mapping.W is None:
        raise ValueError("mapping_mode != 'NONE' requires inputs.mapping.W")

    if cfg.mapping.mapping_mode == "NONE":
        # controls must be in leg space: M == N
        T, N, _ = _infer_TNL_from_book(inputs)
        ctrl = inputs.controls.target_positions if inputs.controls.target_positions is not None else inputs.controls.orders
        assert ctrl is not None
        M = int(ctrl.shape[1])
        if M != N:
            raise ValueError(f"mapping_mode='NONE' requires controls M==N, but got M={M}, N={N}")
