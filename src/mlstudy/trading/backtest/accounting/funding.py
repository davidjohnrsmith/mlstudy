"""
backtest/accounting/funding.py

Optional funding/carry accruals.

These functions are intentionally simple:
- cash interest on cash balance
- coupon accrual on long positions (and negative accrual on shorts if you choose)
- repo funding cost for holding positions (very simplified)

In practice, exact bond funding/carry requires more detailed instrument modeling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def accrue_cash_interest(
    *,
    cash: float,
    cash_rate: float,
    dt_years: float,
) -> float:
    """
    Simple cash interest accrual: cash * rate * dt.
    Returns cash delta (can be negative if cash is negative).
    """
    return float(cash) * float(cash_rate) * float(dt_years)


def accrue_coupon(
    *,
    pos_leg: np.ndarray,
    coupons: np.ndarray,
    accrual_factors: np.ndarray,
    contract_multiplier: float,
) -> float:
    """
    Coupon accrual (simplified):
      accrual_cash = sum(pos_i * coupon_i * accrual_factor_i) * contract_multiplier

    coupons: (N,) annual coupon rate (as decimal) or similar scalar
    accrual_factors: (N,) fraction of year for this bar (e.g., 1/365)
    """
    pos = np.asarray(pos_leg, dtype=np.float64)
    cpn = np.asarray(coupons, dtype=np.float64)
    af = np.asarray(accrual_factors, dtype=np.float64)
    return float(np.dot(pos, cpn * af) * float(contract_multiplier))


def apply_repo_funding(
    *,
    pos_leg: np.ndarray,
    mtm_px: np.ndarray,
    repo_rate: float,
    borrow_spread_bps: float,
    dt_years: float,
    contract_multiplier: float,
) -> float:
    """
    Very simplified repo funding model.

    - Treat position notional as pos * mtm_px * contract_multiplier.
    - Long positions pay repo on notional (negative cash).
    - Short positions pay borrow spread (additional) on absolute notional (negative cash).

    Returns cash delta (negative is cost).
    """
    pos = np.asarray(pos_leg, dtype=np.float64)
    px = np.asarray(mtm_px, dtype=np.float64)
    notional = pos * px * float(contract_multiplier)

    repo_cost = np.sum(np.maximum(notional, 0.0)) * float(repo_rate) * float(dt_years)
    borrow_spread = (float(borrow_spread_bps) / 1e4)
    short_cost = np.sum(np.maximum(-notional, 0.0)) * (float(repo_rate) + borrow_spread) * float(dt_years)

    return -float(repo_cost + short_cost)
