from __future__ import annotations

from enum import Enum


class PortfolioParameter(Enum):
    """LP portfolio backtest parameters."""

    # LP constraint params
    GROSS_DV01_CAP = ("gross_dv01_cap", +1)
    TOP_K = ("top_k", +1)

    # signal gating — higher z thresholds → more selective
    Z_INC = ("z_inc", +1)
    Z_DEC = ("z_dec", +1)

    # p-value thresholds — lower is stricter
    P_INC = ("p_inc", -1)
    P_DEC = ("p_dec", -1)

    # alpha thresholds — higher → more selective
    ALPHA_THR_INC = ("alpha_thr_inc", +1)
    ALPHA_THR_DEC = ("alpha_thr_dec", +1)

    # execution
    MAX_LEVELS = ("max_levels", -1)
    HAIRCUT = ("haircut", -1)

    # cooldown
    COOLDOWN_BARS = ("cooldown_bars", +1)

    # capital
    INITIAL_CAPITAL = ("initial_capital", +1)

    def __init__(self, param_name: str, direction: int):
        self.key = param_name
        self.direction = direction

    @classmethod
    def from_key(cls, name: str) -> "PortfolioParameter":
        for p in cls:
            if p.key == name:
                return p
        raise ValueError(f"Unknown portfolio parameter {name!r}")
