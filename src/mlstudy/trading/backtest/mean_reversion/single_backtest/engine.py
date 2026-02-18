"""
High-level entry point for the mean-reversion backtester.

Validates inputs, selects the loop implementation, and wraps raw
arrays into :class:`MRBacktestResults`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..configs.backtest_config import MRBacktestConfig
from .loop import HAS_NUMBA, mr_loop_jit, _mr_loop_jit_impl
from .results import MRBacktestResults
from .state import ValidateScope


def _validate(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    dv01: np.ndarray,
    zscore: np.ndarray,
    expected_yield_pnl_bps: np.ndarray,
    package_yield_bps: np.ndarray,
    hedge_ratios: np.ndarray,
    cfg: MRBacktestConfig,
) -> None:
    """Basic shape and value checks (fail fast)."""
    T, N, L = bid_px.shape
    if bid_sz.shape != (T, N, L):
        raise ValueError(f"bid_sz shape {bid_sz.shape} != expected {(T, N, L)}")
    if ask_px.shape != (T, N, L):
        raise ValueError(f"ask_px shape {ask_px.shape} != expected {(T, N, L)}")
    if ask_sz.shape != (T, N, L):
        raise ValueError(f"ask_sz shape {ask_sz.shape} != expected {(T, N, L)}")
    if mid_px.shape != (T, N):
        raise ValueError(f"mid_px shape {mid_px.shape} != expected {(T, N)}")
    if dv01.shape != (T, N):
        raise ValueError(f"dv01 shape {dv01.shape} != expected {(T, N)}")
    if zscore.shape != (T,):
        raise ValueError(f"zscore shape {zscore.shape} != expected ({T},)")
    if expected_yield_pnl_bps.shape != (T,):
        raise ValueError(f"expected_yield_pnl_bps shape != ({T},)")
    if package_yield_bps.shape != (T,):
        raise ValueError(f"package_yield_bps shape != ({T},)")
    if hedge_ratios.shape != (T, N):
        raise ValueError(f"hedge_ratios shape {hedge_ratios.shape} != expected {(T, N)}")
    if not (0 <= cfg.ref_leg_idx < N):
        raise ValueError(f"ref_leg_idx {cfg.ref_leg_idx} out of range [0, {N})")

    hr_sums = np.sum(hedge_ratios, axis=1)
    bad_sum_mask = np.abs(hr_sums) > 1e-8
    if np.any(bad_sum_mask):
        first_bad = int(np.argmax(bad_sum_mask))
        raise ValueError(
            f"hedge_ratios must sum to 0 per row (row {first_bad} sums to "
            f"{hr_sums[first_bad]:.6g}).  "
            "Ensure the package is DV01-neutral in yield space."
        )

    ref_vals = hedge_ratios[:, cfg.ref_leg_idx]
    bad_ref_mask = np.abs(ref_vals - 1.0) > 1e-8
    if np.any(bad_ref_mask):
        first_bad = int(np.argmax(bad_ref_mask))
        raise ValueError(
            f"hedge_ratios[:, ref_leg_idx={cfg.ref_leg_idx}] must be 1.0 "
            f"(row {first_bad} has {ref_vals[first_bad]:.6g})"
        )


def run_backtest(
    *,
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    dv01: np.ndarray,
    zscore: np.ndarray,
    expected_yield_pnl_bps: np.ndarray,
    package_yield_bps: np.ndarray,
    hedge_ratios: np.ndarray,
    cfg: Optional[MRBacktestConfig] = None,
    datetimes: Optional[np.ndarray] = None,
) -> MRBacktestResults:
    """Run a mean-reversion backtest.

    Parameters
    ----------
    bid_px, bid_sz, ask_px, ask_sz : (T, N, L)
        L2 order book.  Level 0 = best.
    mid_px : (T, N)
        Independent mid prices (not derived from book).
    dv01 : (T, N)
        DV01 per instrument: price change per 1 bp yield, per 1 unit par.
    zscore : (T,)
        Pre-computed, lagged z-score signal.
    expected_yield_pnl_bps : (T,)
        Expected yield PnL magnitude in bps (non-negative).
    package_yield_bps : (T,)
        Package yield level in bps.
    hedge_ratios : (T, N)
        Yield-space hedge ratios per bar.  ``hedge_ratios[:, ref] == 1``,
        ``sum(axis=1) == 0``.
    cfg : MRBacktestConfig, optional
        If *None*, defaults are used.

    Returns
    -------
    MRBacktestResults
    """

    _validate(
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        mid_px,
        dv01,
        zscore,
        expected_yield_pnl_bps,
        package_yield_bps,
        hedge_ratios,
        cfg,
    )

    scope = ValidateScope.ALL_LEGS if cfg.validate_scope == "ALL_LEGS" else ValidateScope.REF_ONLY

    loop_fn = mr_loop_jit if (cfg.use_jit and HAS_NUMBA) else _mr_loop_jit_impl

    raw = loop_fn(
        bid_px.astype(np.float64, copy=False),
        bid_sz.astype(np.float64, copy=False),
        ask_px.astype(np.float64, copy=False),
        ask_sz.astype(np.float64, copy=False),
        mid_px.astype(np.float64, copy=False),
        dv01.astype(np.float64, copy=False),
        zscore.astype(np.float64, copy=False),
        expected_yield_pnl_bps.astype(np.float64, copy=False),
        package_yield_bps.astype(np.float64, copy=False),
        hedge_ratios.astype(np.float64, copy=False),
        int(cfg.ref_leg_idx),
        float(cfg.target_notional_ref),
        float(cfg.entry_z_threshold),
        float(cfg.take_profit_zscore_soft_threshold),
        float(cfg.take_profit_yield_change_soft_threshold),
        float(cfg.take_profit_yield_change_hard_threshold),
        float(cfg.stop_loss_yield_change_hard_threshold),
        int(cfg.max_holding_bars),
        float(cfg.expected_yield_pnl_bps_multiplier),
        float(cfg.entry_cost_premium_yield_bps),
        float(cfg.tp_cost_premium_yield_bps),
        float(cfg.sl_cost_premium_yield_bps),
        int(cfg.tp_quarantine_bars),
        int(cfg.sl_quarantine_bars),
        int(cfg.time_quarantine_bars),
        int(cfg.max_levels_to_cross),
        float(cfg.size_haircut),
        int(scope),
        float(cfg.initial_capital),
        int(ValidateScope.REF_ONLY),
    )
    return MRBacktestResults.from_loop_output(
        raw,
        datetimes=datetimes,
        mid_px=mid_px,
        package_yield_bps=package_yield_bps,
        zscore=zscore,
    )
