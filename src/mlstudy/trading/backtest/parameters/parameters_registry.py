from __future__ import annotations


class ParameterPreferenceRegistry:
    """Direction registry for MRBacktestConfig numeric fields.

    +1 means higher is preferred, -1 means lower is preferred.
    """

    _DIRECTIONS: dict[str, int] = {
        # higher-is-preferred (+1)
        "target_notional_ref": +1,
        "entry_z_threshold": +1,
        "expected_yield_pnl_bps_multiplier": +1,
        "entry_cost_premium_yield_bps": +1,
        "tp_cost_premium_yield_bps": +1,
        "sl_cost_premium_yield_bps": +1,
        "tp_quarantine_bars": +1,
        "sl_quarantine_bars": +1,
        "time_quarantine_bars": +1,
        "initial_capital": +1,
        # lower-is-preferred (-1)
        "take_profit_zscore_soft_threshold": -1,
        "take_profit_yield_change_soft_threshold": -1,
        "take_profit_yield_change_hard_threshold": -1,
        "stop_loss_yield_change_hard_threshold": -1,
        "max_holding_bars": -1,
        "max_levels_to_cross": -1,
        "size_haircut": -1,
    }

    @classmethod
    def direction(cls, name: str) -> int:
        try:
            return cls._DIRECTIONS[name]
        except KeyError:
            raise ValueError(
                f"Unknown parameter {name!r}; choose from {sorted(cls._DIRECTIONS)}"
            )
