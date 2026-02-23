"""Configuration dataclass for the LP portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """All scalar knobs for a single LP portfolio backtest run.

    Signal gating
    -------------
    Two signal gates (inc / dec) control which fair price is used.
    Risk-increasing trades require ``|zscore| > z_inc`` AND ``adf_p < p_inc``.
    Risk-decreasing trades require ``|zscore| > z_dec`` AND ``adf_p < p_dec``.

    Alpha thresholds
    ----------------
    ``alpha_thr_inc`` / ``alpha_thr_dec`` set the minimum executable alpha
    (in bps) for risk-increasing / risk-decreasing candidates.

    Execution
    ---------
    Trades are sized by the LP solver in DV01 space, converted to notional,
    rounded via ``qty_step`` and filtered by ``min_qty_trade``.
    """

    # -- LP constraint params -------------------------------------------------
    gross_dv01_cap: float
    top_k: int

    # -- Signal gating thresholds ---------------------------------------------
    z_inc: float
    p_inc: float
    z_dec: float
    p_dec: float

    # -- Alpha thresholds (bps) -----------------------------------------------
    alpha_thr_inc: float
    alpha_thr_dec: float

    # -- Execution params -----------------------------------------------------
    max_levels: int
    haircut: float
    qty_step: float
    min_qty_trade: float
    min_fill_ratio: float

    # -- Cooldown -------------------------------------------------------------
    cooldown_bars: int
    cooldown_mode: int

    # -- Maturity filter (0.0 to disable) -------------------------------------
    min_maturity_inc: float

    # -- Capital --------------------------------------------------------------
    initial_capital: float

    # -- Metrics ------------------------------------------------------------
    close_time: str   # e.g. "16:00:00"; use "none" to disable close filtering
