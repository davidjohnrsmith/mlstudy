"""
backtest/accounting/ledger.py

State update helpers:
- apply fills to positions and cash
- mark-to-market equity
- compute step PnL

Conventions
- Truth positions are stored in leg space (N).
- Cash and equity are in currency units.
- contract_multiplier converts (price * size) to currency notionals (for futures).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LedgerState:
    """
    Minimal portfolio state.
    """
    pos_leg: np.ndarray     # (N,)
    cash: float
    equity: float
    peak_equity: float


def init_state(
    *,
    N: int,
    initial_capital: float,
    initial_pos_leg: Optional[np.ndarray] = None,
) -> LedgerState:
    pos = np.zeros(N, dtype=np.float64) if initial_pos_leg is None else np.asarray(initial_pos_leg, dtype=np.float64).copy()
    cash = float(initial_capital)
    equity = float(initial_capital)  # will be overwritten by first MTM
    return LedgerState(pos_leg=pos, cash=cash, equity=equity, peak_equity=equity)


def mark_to_market_equity(
    *,
    pos_leg: np.ndarray,
    mtm_px: np.ndarray,
    contract_multiplier: float,
    cash: float,
) -> float:
    """
    Equity = cash + sum(pos * mtm_px * contract_multiplier)
    """
    return float(cash + np.dot(np.asarray(pos_leg, dtype=np.float64), np.asarray(mtm_px, dtype=np.float64)) * float(contract_multiplier))


def apply_fill_to_state(
    *,
    state: LedgerState,
    leg_idx: int,
    filled_qty: float,      # signed
    fill_px: float,
    contract_multiplier: float,
    fee_cash: float,
) -> None:
    """
    Apply a single leg fill to positions and cash.
    - Buy (filled_qty>0) consumes cash.
    - Sell (filled_qty<0) adds cash.
    fee_cash is a positive cash cost to subtract.
    """
    q = float(filled_qty)
    px = float(fill_px)
    mult = float(contract_multiplier)

    # Cash delta: pay q*px*mult for buys, receive for sells
    state.cash -= q * px * mult
    state.cash -= float(fee_cash)

    # Position update
    state.pos_leg[int(leg_idx)] += q


def compute_step_pnl(equity_now: float, equity_prev: float) -> float:
    """
    Step PnL defined as change in equity.
    """
    return float(equity_now - equity_prev)


def update_peak_equity(state: LedgerState) -> None:
    state.peak_equity = float(max(state.peak_equity, state.equity))
