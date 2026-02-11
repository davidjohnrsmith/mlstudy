"""
backtest/risk/limits.py

Risk limits and clipping policies.

Important design choice:
- Risk checks should be applied in leg space (N), because execution, liquidity and MTM live there.
- If you control in portfolio space (M), map to legs first, then apply limits.

Supported limits (based on BacktestConfig.risk):
- max_abs_pos_per_leg
- max_gross_pos, max_net_pos
- max_gross_dv01, max_net_dv01 (requires dv01s)
- action_on_breach: CLIP | FLATTEN | REJECT

This module provides functions that are easy to port to Numba:
no dicts, no classes, just arrays and scalars.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .exposures import gross_net_pos, gross_net_dv01


def clip_positions_legs(
    target_leg: np.ndarray,
    *,
    max_abs_pos_per_leg: float,
) -> np.ndarray:
    """
    Clip per-leg target positions to [-max_abs, +max_abs].
    """
    if not np.isfinite(max_abs_pos_per_leg):
        return target_leg
    m = float(max_abs_pos_per_leg)
    return np.clip(target_leg, -m, +m)


def clip_trade_legs(
    trade_leg: np.ndarray,
    *,
    max_abs_trade_per_leg: Optional[float] = None,
) -> np.ndarray:
    """
    Optional: clip per-leg trade size.
    Often useful when you want to constrain turnover.
    """
    if max_abs_trade_per_leg is None or not np.isfinite(max_abs_trade_per_leg):
        return trade_leg
    m = float(max_abs_trade_per_leg)
    return np.clip(trade_leg, -m, +m)


def _apply_gross_net_clip_pos(
    target_leg: np.ndarray,
    *,
    max_gross: float,
    max_net: float,
) -> np.ndarray:
    """
    Soft clip: scale down positions uniformly to satisfy gross/net limits.
    This preserves relative weights.

    - gross constraint: sum(|pos|) <= max_gross
    - net constraint: |sum(pos)| <= max_net  (if net can be +/-)
    """
    pos = target_leg.astype(np.float64, copy=True)

    gross, net = gross_net_pos(pos)

    # gross scale
    if np.isfinite(max_gross) and max_gross > 0 and gross > max_gross:
        scale = max_gross / gross
        pos *= scale
        gross, net = gross_net_pos(pos)

    # net scale (only if net is too large; scale all positions uniformly)
    if np.isfinite(max_net) and max_net >= 0 and abs(net) > max_net:
        if abs(net) > 1e-12:
            scale = max_net / abs(net)
            pos *= scale

    return pos


def _apply_gross_net_clip_dv01(
    target_leg: np.ndarray,
    dv01_leg: np.ndarray,
    *,
    max_gross_dv01: float,
    max_net_dv01: float,
) -> np.ndarray:
    """
    Soft clip based on DV01 exposure. Scales positions uniformly.
    """
    pos = target_leg.astype(np.float64, copy=True)
    dv01 = np.asarray(dv01_leg, dtype=np.float64)

    gross, net = gross_net_dv01(pos, dv01)

    if np.isfinite(max_gross_dv01) and max_gross_dv01 > 0 and gross > max_gross_dv01:
        scale = max_gross_dv01 / gross
        pos *= scale
        gross, net = gross_net_dv01(pos, dv01)

    if np.isfinite(max_net_dv01) and max_net_dv01 >= 0 and abs(net) > max_net_dv01:
        if abs(net) > 1e-12:
            scale = max_net_dv01 / abs(net)
            pos *= scale

    return pos


def apply_limits_step(
    target_leg: np.ndarray,
    *,
    enabled: bool,
    action_on_breach: str,
    max_abs_pos_per_leg: float,
    max_gross_pos: float,
    max_net_pos: float,
    max_gross_dv01: float,
    max_net_dv01: float,
    dv01_leg: Optional[np.ndarray],
) -> Tuple[np.ndarray, bool]:
    """
    Apply all risk limits to a desired leg target.
    Returns (new_target_leg, breached).

    Behavior by action_on_breach:
    - "CLIP": clip/scale to fit
    - "FLATTEN": if any breach detected (pre-clip), set all to zero
    - "REJECT": if any breach detected (pre-clip), return original target and breached=True
      (caller can then not trade / keep previous position)

    Note: We interpret breach based on the *unclipped* target.
    """

    if not enabled:
        return target_leg, False

    action = action_on_breach.upper()
    pos = target_leg.astype(np.float64, copy=True)

    # First: per-leg clip (always safe)
    pos_clipped = clip_positions_legs(pos, max_abs_pos_per_leg=max_abs_pos_per_leg)

    # Check exposures on unclipped target (pos) to decide "breach"
    gross_pos, net_pos = gross_net_pos(pos)
    breach = False
    if np.isfinite(max_gross_pos) and max_gross_pos > 0 and gross_pos > max_gross_pos:
        breach = True
    if np.isfinite(max_net_pos) and max_net_pos >= 0 and abs(net_pos) > max_net_pos:
        breach = True

    if dv01_leg is not None:
        gross_dv01, net_dv01 = gross_net_dv01(pos, dv01_leg)
        if np.isfinite(max_gross_dv01) and max_gross_dv01 > 0 and gross_dv01 > max_gross_dv01:
            breach = True
        if np.isfinite(max_net_dv01) and max_net_dv01 >= 0 and abs(net_dv01) > max_net_dv01:
            breach = True
    else:
        # if dv01 constraints configured but dv01_leg missing, caller should have validated earlier
        pass

    if not breach:
        # still apply soft clipping for per-leg bounds
        return pos_clipped, False

    if action == "FLATTEN":
        return np.zeros_like(pos_clipped), True

    if action == "REJECT":
        return target_leg, True

    # default CLIP: scale down to satisfy gross/net (and DV01 if present)
    pos2 = _apply_gross_net_clip_pos(
        pos_clipped,
        max_gross=max_gross_pos,
        max_net=max_net_pos,
    )
    if dv01_leg is not None:
        pos2 = _apply_gross_net_clip_dv01(
            pos2,
            dv01_leg,
            max_gross_dv01=max_gross_dv01,
            max_net_dv01=max_net_dv01,
        )
    return pos2, True
