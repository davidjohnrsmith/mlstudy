"""
backtest/execution/impact.py

Additional execution impact/noise on top of the book-derived fill price.

This is useful because:
- book snapshots may be optimistic (hidden liquidity, replenishment, queue)
- you want stress-testing knobs beyond pure book-walk VWAP

Two modes:
- POWER: add impact_bps = k_bps * (|qty| / liquidity)^alpha
- STOCHASTIC: add random bps noise with std noise_bps_std
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.types import ImpactMode, LiquiditySource


def apply_impact_and_noise(
    *,
    fill_px: float,
    filled_qty: float,
    side: int,
    impact_mode: int,
    k_bps: float,
    alpha: float,
    liquidity_source: int,
    bar_volume: Optional[float],
    adv: Optional[float],
    rng_state: Optional[np.random.Generator],
    noise_bps_std: float,
) -> Tuple[float, float]:
    """
    Returns (adjusted_fill_px, added_bps_total).

    Convention:
    - BUY => adverse move increases price
    - SELL => adverse move decreases price
    """
    px = float(fill_px)
    if px <= 0 or filled_qty == 0:
        return px, 0.0

    bps = 0.0

    if impact_mode == int(ImpactMode.POWER):
        liq = None
        if liquidity_source == int(LiquiditySource.BAR_VOLUME):
            liq = bar_volume
        else:
            liq = adv

        if liq is not None and liq > 0 and k_bps > 0:
            frac = abs(float(filled_qty)) / float(liq)
            bps += float(k_bps) * (frac ** float(alpha))

    if impact_mode == int(ImpactMode.STOCHASTIC):
        if noise_bps_std and noise_bps_std > 0:
            if rng_state is None:
                rng_state = np.random.default_rng(0)
            bps += float(rng_state.normal(0.0, float(noise_bps_std)))

    # apply adverse direction
    sign = +1.0 if side > 0 else -1.0
    px_adj = px * (1.0 + sign * bps / 1e4)
    return float(px_adj), float(bps)
