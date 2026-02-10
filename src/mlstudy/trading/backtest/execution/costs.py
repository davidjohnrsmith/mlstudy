"""
backtest/execution/costs.py

Explicit costs / fees (commissions, exchange fees) separate from execution price.

We model cash delta from fees so the accounting layer can do:
cash -= fee_cash_delta
"""

from __future__ import annotations

from typing import Literal

from ..core.types import RejectReason


def compute_fees_cash_delta(
    *,
    filled_qty: float,
    fill_px: float,
    contract_multiplier: float,
    fee_bps: float,
    fee_mode: Literal["PER_NOTIONAL", "PER_SIZE_UNIT"],
    fee_per_unit: float,
) -> float:
    """
    Return positive cash cost (to subtract from cash).
    """
    qty = abs(float(filled_qty))
    if qty <= 0 or fill_px <= 0:
        return 0.0

    fee_cash = 0.0

    if fee_bps and fee_bps != 0.0:
        # notional approximated as qty * px * multiplier
        notional = qty * float(fill_px) * float(contract_multiplier)
        fee_cash += (float(fee_bps) / 1e4) * notional

    if fee_mode == "PER_SIZE_UNIT" and fee_per_unit and fee_per_unit != 0.0:
        fee_cash += qty * float(fee_per_unit)

    return float(fee_cash)
