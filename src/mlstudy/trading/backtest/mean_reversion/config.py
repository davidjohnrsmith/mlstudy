from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MRBacktestConfig:
    """All scalar knobs for a single mean-reversion backtest run.

    Threshold sign conventions
    -------------------------
    All TP / SL thresholds are applied to **direction-adjusted** values
    inside the loop.  See :mod:`.loop` module docstring for details.

    DV01 / sizing
    -------------
    ``target_notional_ref`` is the par-notional size of the reference leg.
    Leg sizes for other instruments are derived so that yield-space hedge
    ratios are maintained::

        size_i = target_notional_ref * dv01[t, ref] * hedge_ratios[t, i] / dv01[t, i]

    Basket execution cost (always >= 0) is compared against::

        acceptable_cost = acceptable_yield_bps * target_notional_ref * dv01[t, ref]
    """

    # -- sizing / reference ------------------------------------------------
    target_notional_ref: float = 100.0
    ref_leg_idx: int = 0

    # -- entry -------------------------------------------------------------
    entry_z_threshold: float = 2.0

    # -- take-profit -------------------------------------------------------
    take_profit_zscore_soft_threshold: float = 0.5
    take_profit_yield_change_soft_threshold: float = 1.0  # yield bps
    take_profit_yield_change_hard_threshold: float = 3.0  # yield bps

    # -- stop-loss ---------------------------------------------------------
    stop_loss_yield_change_hard_threshold: float = 5.0  # yield bps

    # -- max holding -------------------------------------------------------
    max_holding_bars: int = 0  # 0 = disabled

    # -- cost premia -------------------------------------------------------
    expected_yield_pnl_bps_multiplier: float = 1.0
    entry_cost_premium_yield_bps: float = 0.0
    tp_cost_premium_yield_bps: float = 0.0
    sl_cost_premium_yield_bps: float = 0.0  # unused for forced

    # -- quarantine --------------------------------------------------------
    tp_quarantine_bars: int = 0
    sl_quarantine_bars: int = 0
    time_quarantine_bars: int = 0

    # -- execution ---------------------------------------------------------
    max_levels_to_cross: int = 5
    size_haircut: float = 1.0

    # -- market validity ---------------------------------------------------
    validate_scope: str = "REF_ONLY"  # "REF_ONLY" | "ALL_LEGS"

    # -- initial state -----------------------------------------------------
    initial_capital: float = 0.0

    # -- JIT ---------------------------------------------------------------
    use_jit: bool = False
