"""
backtest/risk/derisk.py

Drawdown-based de-risking overlays.

This is portfolio-level logic:
- If equity drawdown exceeds max_drawdown, scale down or flatten.

The engine should call this after MTM update (or before next trading decision),
depending on your convention.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


def apply_drawdown_derisk(
    equity_t: float,
    peak_equity: float,
    *,
    max_drawdown: float,
    mode: Literal["FLATTEN", "SCALE"] = "FLATTEN",
) -> Tuple[float, bool]:
    """
    Returns (risk_multiplier, triggered).
    - risk_multiplier multiplies future targets (1.0 means no change)
    """
    if max_drawdown is None or max_drawdown <= 0:
        return 1.0, False

    peak = max(float(peak_equity), 1e-12)
    dd = (peak - float(equity_t)) / peak

    if dd <= max_drawdown:
        return 1.0, False

    if mode == "SCALE":
        # scale down linearly to zero at 2x max_drawdown (simple heuristic)
        # dd=max_drawdown => scale=1, dd=2*max_drawdown => scale=0
        scale = 1.0 - (dd - max_drawdown) / max_drawdown
        scale = float(np.clip(scale, 0.0, 1.0))
        return scale, True

    # default flatten
    return 0.0, True
