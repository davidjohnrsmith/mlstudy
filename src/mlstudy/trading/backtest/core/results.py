"""
backtest/core/results.py

Raw result containers returned by the engine.
No post-analysis; consumers can compute metrics elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BacktestResults:
    datetimes_int: np.ndarray                 # (T,)
    positions_leg: np.ndarray                 # (T, N)
    cash: np.ndarray                          # (T,)
    equity: np.ndarray                        # (T,)
    pnl: np.ndarray                           # (T,)

    # Optional: structured fills array or None
    fills: Optional[np.ndarray] = None

    # Optional: metadata (config snapshot, version, etc.)
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_jit(out: Dict[str, Any], *, datetimes_int: np.ndarray) -> "BacktestResults":
        """
        Convert jit_loop output dict into BacktestResults.
        """
        return BacktestResults(
            datetimes_int=datetimes_int,
            positions_leg=out["positions_leg"],
            cash=out["cash"],
            equity=out["equity"],
            pnl=out["pnl"],
            fills=out.get("fills", None),
            meta=out.get("meta", None),
        )
