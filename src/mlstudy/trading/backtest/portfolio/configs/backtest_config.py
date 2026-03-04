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
    rounded via per-instrument ``qty_step`` (from meta) and filtered by
    per-instrument ``min_qty_trade`` (from meta / hedge_meta).
    """

    # -- Solver mode (False = LP with greedy fallback, True = greedy only) ----
    use_greedy: bool

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
    min_fill_ratio: float

    # -- Cooldown -------------------------------------------------------------
    cooldown_bars: int
    cooldown_mode: int

    # -- Maturity filter (0.0 to disable) -------------------------------------
    min_maturity_inc: float

    # -- Capital --------------------------------------------------------------
    initial_capital: float

    # -- Metrics ------------------------------------------------------------
    close_time: str  # e.g. "16:00:00"; use "none" to disable close filtering

    # -- Maturity bucket bins (empty tuple to disable) -------------------------
    maturity_bucket_bins: tuple[float, ...] = ()

    # -- Bucket DV01 caps (empty dict/tuple to disable) -----------------------
    # issuer_dv01_caps: mapping from issuer name → max absolute DV01
    #   e.g. {"Japan": 50.0, "US": 100.0}
    issuer_dv01_caps: dict[str, float] = None
    # mat_bucket_dv01_caps: per-bucket cap, indexed by digitize bucket index
    #   length must equal len(maturity_bucket_bins) + 1
    #   e.g. maturity_bucket_bins=[2,5,7,10] → 5 buckets → 5 caps
    mat_bucket_dv01_caps: tuple[float, ...] = ()

    def __post_init__(self):
        if self.issuer_dv01_caps is None:
            object.__setattr__(self, "issuer_dv01_caps", {})
