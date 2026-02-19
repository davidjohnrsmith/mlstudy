"""Configuration dataclass for the LP portfolio backtest."""

from __future__ import annotations

from dataclasses import dataclass

from mlstudy.trading.backtest.portfolio.single_backtest.state import CooldownMode


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
    gross_dv01_cap: float = 100.0
    top_k: int = 10

    # -- Signal gating thresholds ---------------------------------------------
    z_inc: float = 2.0
    p_inc: float = 0.05
    z_dec: float = 1.0
    p_dec: float = 0.10

    # -- Alpha thresholds (bps) -----------------------------------------------
    alpha_thr_inc: float = 1.0
    alpha_thr_dec: float = 0.5

    # -- Execution params -----------------------------------------------------
    max_levels: int = 3
    haircut: float = 1.0
    qty_step: float = 0.0
    min_qty_trade: float = 0.0
    min_fill_ratio: float = 0.0

    # -- Cooldown -------------------------------------------------------------
    cooldown_bars: int = 0
    cooldown_mode: int = int(CooldownMode.BLOCK_ALL)

    # -- Maturity filter (0.0 to disable) -------------------------------------
    min_maturity_inc: float = 0.0

    # -- Capital --------------------------------------------------------------
    initial_capital: float = 1_000_000.0
