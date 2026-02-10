"""
backtest/data/align.py

Alignment helpers.

This module is intentionally lightweight and does not require pandas.
If you already align upstream using pandas, you might not need this file.

Primary use:
- Align a set of arrays to a target calendar of datetimes_int
- Fill missing bars if allowed (forward-fill prices, zero sizes/volumes, etc.)

Because order books are 3D (T,N,L), alignment needs careful fill rules.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from ..core.types import BacktestInputs, OrderBookL2, MarketState, Availability, Controls, PortfolioMapping


def _index_map(src_dt: np.ndarray, dst_dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (src_idx, dst_idx) such that src_dt[src_idx] == dst_dt[dst_idx].
    Assumes both are sorted ascending.
    """
    src_pos = {int(d): i for i, d in enumerate(src_dt)}
    src_idx = []
    dst_idx = []
    for j, d in enumerate(dst_dt):
        i = src_pos.get(int(d))
        if i is not None:
            src_idx.append(i)
            dst_idx.append(j)
    return np.asarray(src_idx, dtype=np.int64), np.asarray(dst_idx, dtype=np.int64)


def _alloc_like(shape, dtype) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def align_inputs_to_calendar(
    inputs: BacktestInputs,
    target_datetimes_int: np.ndarray,
    *,
    ffill_book_prices: bool = False,
    ffill_mtm_px: bool = True,
    fill_missing_sizes_with_zero: bool = True,
) -> BacktestInputs:
    """
    Align BacktestInputs to target_datetimes_int.

    Fill rules (defaults):
    - mtm_px: forward-fill (common for daily bars; if missing, carry last mid)
    - L2 book prices: by default do NOT ffill (books can go stale); set ffill_book_prices=True if you must
    - book sizes: fill missing with zeros (meaning no liquidity) by default
    - controls/mapping: missing bars are filled with last value? (we default to zeros; you can change upstream)
    """
    src_dt = inputs.datetimes_int
    dst_dt = target_datetimes_int
    if src_dt.ndim != 1 or dst_dt.ndim != 1:
        raise ValueError("datetimes_int must be 1D arrays")
    if dst_dt.size == 0:
        raise ValueError("target_datetimes_int is empty")

    # Infer shapes
    T_src = int(src_dt.shape[0])
    T_dst = int(dst_dt.shape[0])
    bid_px = inputs.orderbook.bid_px
    T, N, L = bid_px.shape
    if T != T_src:
        raise ValueError("inputs.datetimes_int length does not match orderbook T")

    # Map indices
    src_idx, dst_idx = _index_map(src_dt, dst_dt)

    # Allocate new arrays
    bid_px_new = _alloc_like((T_dst, N, L), bid_px.dtype)
    ask_px_new = _alloc_like((T_dst, N, L), inputs.orderbook.ask_px.dtype)
    bid_sz_new = _alloc_like((T_dst, N, L), inputs.orderbook.bid_sz.dtype)
    ask_sz_new = _alloc_like((T_dst, N, L), inputs.orderbook.ask_sz.dtype)

    # Copy matched
    bid_px_new[dst_idx] = inputs.orderbook.bid_px[src_idx]
    ask_px_new[dst_idx] = inputs.orderbook.ask_px[src_idx]
    bid_sz_new[dst_idx] = inputs.orderbook.bid_sz[src_idx]
    ask_sz_new[dst_idx] = inputs.orderbook.ask_sz[src_idx]

    # Fill missing sizes
    if fill_missing_sizes_with_zero:
        # already zero-filled
        pass

    # Optionally ffill book prices
    if ffill_book_prices:
        for t in range(1, T_dst):
            missing = np.all(bid_px_new[t] == 0.0) & np.all(ask_px_new[t] == 0.0)
            if missing:
                bid_px_new[t] = bid_px_new[t - 1]
                ask_px_new[t] = ask_px_new[t - 1]

    # Optional extras
    def _align_2d(opt: Optional[np.ndarray], *, ffill: bool) -> Optional[np.ndarray]:
        if opt is None:
            return None
        if opt.shape != (T_src, N):
            raise ValueError(f"Expected (T,N)=({T_src},{N}) but got {opt.shape}")
        out = _alloc_like((T_dst, N), opt.dtype)
        out[dst_idx] = opt[src_idx]
        if ffill:
            for t in range(1, T_dst):
                # ffill row if it is all zeros
                if np.all(out[t] == 0.0):
                    out[t] = out[t - 1]
        return out

    last_px_new = _align_2d(inputs.orderbook.last_px, ffill=False)
    last_sz_new = _align_2d(inputs.orderbook.last_sz, ffill=False)
    traded_volume_new = _align_2d(inputs.orderbook.traded_volume, ffill=False)
    adv_new = _align_2d(inputs.orderbook.adv, ffill=False)

    # Market mtm_px can be ffilled
    mtm_px_new = _align_2d(inputs.market.mtm_px, ffill=ffill_mtm_px)
    yields_new = _align_2d(inputs.market.yields, ffill=False)
    dv01s_new = _align_2d(inputs.market.dv01s, ffill=False)
    accrual_new = _align_2d(inputs.market.accrual_factors, ffill=False)

    # Coupons can be (N,) or (T,N)
    coupons_new = inputs.market.coupons
    if coupons_new is not None and isinstance(coupons_new, np.ndarray) and coupons_new.ndim == 2:
        if coupons_new.shape != (T_src, N):
            raise ValueError(f"market.coupons expected (T,N)=({T_src},{N}) but got {coupons_new.shape}")
        tmp = _alloc_like((T_dst, N), coupons_new.dtype)
        tmp[dst_idx] = coupons_new[src_idx]
        coupons_new = tmp

    # repo_rates can be (T,) or (T,N)
    repo_new = inputs.market.repo_rates
    if repo_new is not None:
        repo_new = np.asarray(repo_new)
        if repo_new.ndim == 1:
            if repo_new.shape != (T_src,):
                raise ValueError(f"market.repo_rates expected (T,) but got {repo_new.shape}")
            tmp = _alloc_like((T_dst,), repo_new.dtype)
            tmp[dst_idx] = repo_new[src_idx]
            # optionally ffill
            if ffill_mtm_px:
                for t in range(1, T_dst):
                    if tmp[t] == 0:
                        tmp[t] = tmp[t - 1]
            repo_new = tmp
        elif repo_new.ndim == 2:
            if repo_new.shape != (T_src, N):
                raise ValueError(f"market.repo_rates expected (T,N) but got {repo_new.shape}")
            tmp = _alloc_like((T_dst, N), repo_new.dtype)
            tmp[dst_idx] = repo_new[src_idx]
            repo_new = tmp
        else:
            raise ValueError("market.repo_rates must be (T,) or (T,N)")

    # Controls (T,M) aligned; default fill missing with zeros
    def _align_ctrl(opt: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if opt is None:
            return None
        opt = np.asarray(opt)
        if opt.ndim != 2 or opt.shape[0] != T_src:
            raise ValueError(f"controls expected (T,M) with T={T_src}, got {opt.shape}")
        M = int(opt.shape[1])
        out = _alloc_like((T_dst, M), opt.dtype)
        out[dst_idx] = opt[src_idx]
        return out

    tp_new = _align_ctrl(inputs.controls.target_positions)
    od_new = _align_ctrl(inputs.controls.orders)

    # Mapping W (T,M,N)
    W_new = inputs.mapping.W
    if W_new is not None:
        W_new = np.asarray(W_new)
        if W_new.ndim != 3 or W_new.shape[0] != T_src or W_new.shape[2] != N:
            raise ValueError(f"mapping.W expected (T,M,N)=({T_src},M,{N}), got {W_new.shape}")
        M = int(W_new.shape[1])
        out = _alloc_like((T_dst, M, N), W_new.dtype)
        out[dst_idx] = W_new[src_idx]
        W_new = out

    # Availability masks
    def _align_mask(opt: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if opt is None:
            return None
        opt = np.asarray(opt)
        if opt.shape != (T_src, N):
            raise ValueError(f"mask expected (T,N)=({T_src},{N}), got {opt.shape}")
        out = _alloc_like((T_dst, N), opt.dtype)
        out[dst_idx] = opt[src_idx]
        return out

    active_new = _align_mask(inputs.availability.active_mask)
    tradable_new = _align_mask(inputs.availability.tradable_mask)

    new_inputs = BacktestInputs(
        datetimes_int=dst_dt,
        orderbook=OrderBookL2(
            bid_px=bid_px_new,
            bid_sz=bid_sz_new,
            ask_px=ask_px_new,
            ask_sz=ask_sz_new,
            last_px=last_px_new,
            last_sz=last_sz_new,
            traded_volume=traded_volume_new,
            adv=adv_new,
        ),
        market=MarketState(
            mtm_px=mtm_px_new,
            yields=yields_new,
            dv01s=dv01s_new,
            coupons=coupons_new,
            accrual_factors=accrual_new,
            repo_rates=repo_new,
        ),
        mapping=PortfolioMapping(W=W_new, W_lag_bars=inputs.mapping.W_lag_bars),
        controls=Controls(target_positions=tp_new, orders=od_new),
        availability=Availability(active_mask=active_new, tradable_mask=tradable_new),
    )
    return new_inputs
