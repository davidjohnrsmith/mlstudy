"""
backtest/accounting/mtm.py

Mark-to-market price selection and derivation.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


def derive_mid_mtm(bid_px_l0: np.ndarray, ask_px_l0: np.ndarray) -> np.ndarray:
    """
    Derive mid prices from best bid/ask.
    Inputs: (T,N) best bid and best ask.
    Returns: (T,N) mid.
    """
    return 0.5 * (bid_px_l0 + ask_px_l0)


def select_mtm_px(
    *,
    mtm_source: Literal["MID", "LAST", "EXTERNAL_MTM"],
    bid_px_l0: np.ndarray,
    ask_px_l0: np.ndarray,
    last_px: Optional[np.ndarray],
    external_mtm: Optional[np.ndarray],
) -> np.ndarray:
    """
    Choose the MTM series used for price MTM PnL.

    - MID: use (bid0+ask0)/2
    - LAST: use last trade price (requires last_px)
    - EXTERNAL_MTM: use external_mtm (requires it)

    Returns: (T,N)
    """
    if mtm_source == "EXTERNAL_MTM":
        if external_mtm is None:
            raise ValueError("mtm_source='EXTERNAL_MTM' requires external_mtm")
        return external_mtm

    if mtm_source == "LAST":
        if last_px is None:
            raise ValueError("mtm_source='LAST' requires last_px")
        return last_px

    # default MID
    return derive_mid_mtm(bid_px_l0, ask_px_l0)
