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
    target_notional_ref: float
    ref_leg_idx: int

    # -- entry -------------------------------------------------------------
    entry_z_threshold: float

    # -- take-profit -------------------------------------------------------
    take_profit_zscore_soft_threshold: float
    take_profit_yield_change_soft_threshold: float
    take_profit_yield_change_hard_threshold: float

    # -- stop-loss ---------------------------------------------------------
    stop_loss_yield_change_hard_threshold: float

    # -- max holding -------------------------------------------------------
    max_holding_bars: int

    # -- cost premia -------------------------------------------------------
    expected_yield_pnl_bps_multiplier: float
    entry_cost_premium_yield_bps: float
    tp_cost_premium_yield_bps: float
    sl_cost_premium_yield_bps: float

    # -- quarantine --------------------------------------------------------
    tp_quarantine_bars: int
    sl_quarantine_bars: int
    time_quarantine_bars: int

    # -- execution ---------------------------------------------------------
    max_levels_to_cross: int
    size_haircut: float

    # -- market validity ---------------------------------------------------
    validate_scope: str

    # -- initial state -----------------------------------------------------
    initial_capital: float

    # -- metrics -----------------------------------------------------------
    close_time: str   # e.g. "16:00:00"; use "none" to disable close filtering

    # -- JIT ---------------------------------------------------------------
    use_jit: bool
