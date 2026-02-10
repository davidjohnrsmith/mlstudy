"""
backtest/risk/exposures.py

Exposure calculations (strategy-agnostic).
These are used for:
- reporting (gross/net)
- constraints (max gross/net DV01, etc.)

All functions operate in leg space (N).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def gross_net_pos(pos_leg: np.ndarray) -> Tuple[float, float]:
    """
    Gross and net exposure in "position units" (contracts/notional/shares).
    """
    pos = np.asarray(pos_leg, dtype=np.float64)
    gross = float(np.sum(np.abs(pos)))
    net = float(np.sum(pos))
    return gross, net


def gross_net_dv01(pos_leg: np.ndarray, dv01_leg: np.ndarray) -> Tuple[float, float]:
    """
    Gross and net DV01 exposure.

    dv01_leg is (N,) DV01 per unit position for each leg at time t.
    """
    pos = np.asarray(pos_leg, dtype=np.float64)
    dv01 = np.asarray(dv01_leg, dtype=np.float64)
    exp = pos * dv01
    gross = float(np.sum(np.abs(exp)))
    net = float(np.sum(exp))
    return gross, net


def leg_dv01_exposure(pos_leg: np.ndarray, dv01_leg: np.ndarray) -> np.ndarray:
    """
    Per-leg DV01 exposure array (N,).
    """
    return np.asarray(pos_leg, dtype=np.float64) * np.asarray(dv01_leg, dtype=np.float64)
