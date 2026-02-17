from __future__ import annotations

from enum import Enum


class Parameter(Enum):
    TARGET_NOTIONAL_REF = ("target_notional_ref", +1)
    ENTRY_Z_THRESHOLD = ("entry_z_threshold", +1)
    EXPECTED_YIELD_PNL_BPS_MULTIPLIER = ("expected_yield_pnl_bps_multiplier", +1)

    ENTRY_COST_PREMIUM_YIELD_BPS = ("entry_cost_premium_yield_bps", +1)
    TP_COST_PREMIUM_YIELD_BPS = ("tp_cost_premium_yield_bps", +1)
    SL_COST_PREMIUM_YIELD_BPS = ("sl_cost_premium_yield_bps", +1)

    TP_QUARANTINE_BARS = ("tp_quarantine_bars", +1)
    SL_QUARANTINE_BARS = ("sl_quarantine_bars", +1)
    TIME_QUARANTINE_BARS = ("time_quarantine_bars", +1)

    INITIAL_CAPITAL = ("initial_capital", +1)

    TAKE_PROFIT_ZSCORE_SOFT_THRESHOLD = ("take_profit_zscore_soft_threshold", -1)
    TAKE_PROFIT_YIELD_CHANGE_SOFT_THRESHOLD = ("take_profit_yield_change_soft_threshold", -1)
    TAKE_PROFIT_YIELD_CHANGE_HARD_THRESHOLD = ("take_profit_yield_change_hard_threshold", -1)

    STOP_LOSS_YIELD_CHANGE_HARD_THRESHOLD = ("stop_loss_yield_change_hard_threshold", -1)
    MAX_HOLDING_BARS = ("max_holding_bars", -1)
    MAX_LEVELS_TO_CROSS = ("max_levels_to_cross", -1)
    SIZE_HAIRCUT = ("size_haircut", -1)

    def __init__(self, param_name: str, direction: int):
        self.key = param_name
        self.direction = direction

    @classmethod
    def from_key(cls, name: str) -> "Parameter":
        for p in cls:
            if p.key == name:
                return p
        raise ValueError(f"Unknown parameter {name!r}")
