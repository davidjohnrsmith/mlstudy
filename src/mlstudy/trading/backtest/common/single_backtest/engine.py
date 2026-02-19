"""Reusable validation and conversion helpers for backtest engines."""

from __future__ import annotations

import numpy as np


def ensure_f64(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """Return *arr* as float64, copying only if the dtype differs."""
    if arr.dtype != np.float64:
        return arr.astype(np.float64)
    return arr


def validate_l2_shapes(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    label: str = "",
) -> tuple[int, int, int]:
    """Check that L2 book arrays are consistently shaped.

    Parameters
    ----------
    bid_px, bid_sz, ask_px, ask_sz : (T, N, L)
    mid_px : (T, N)
    label : optional prefix for error messages (e.g. "hedge_").

    Returns
    -------
    (T, N, L)
    """
    pfx = f"{label}" if label else ""
    if bid_px.ndim != 3:
        raise ValueError(f"{pfx}bid_px must be 3-D, got shape {bid_px.shape}")
    T, N, L = bid_px.shape
    for name, arr in [
        (f"{pfx}bid_sz", bid_sz),
        (f"{pfx}ask_px", ask_px),
        (f"{pfx}ask_sz", ask_sz),
    ]:
        if arr.shape != (T, N, L):
            raise ValueError(f"{name} shape {arr.shape} != expected {(T, N, L)}")
    if mid_px.shape != (T, N):
        raise ValueError(f"{pfx}mid_px shape {mid_px.shape} != expected {(T, N)}")
    return T, N, L
