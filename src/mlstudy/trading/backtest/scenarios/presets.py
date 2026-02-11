"""
backtest/scenarios/presets.py

Named scenario presets for BacktestConfig.

These presets are meant to be:
- small and readable
- easy to modify
- used as baselines for sweeps

They assume the L2 execution config in core/types.py:
- execution.mode: MID / TOP_OF_BOOK / ORDERBOOK_WALK / LAST
- execution.size_haircut, max_levels_to_cross, participation_cap
- impact model for additional slippage stress
"""

from __future__ import annotations

from dataclasses import replace

from ..core.types import BacktestConfig, ExecutionModelConfig, ImpactModelConfig, FeesConfig


def preset(name: str, base: BacktestConfig | None = None) -> BacktestConfig:
    """
    Return a named preset config.

    name:
      - "FAST": optimistic, quick debugging
      - "REALISTIC": book-walk + haircut + participation cap + small fees
      - "STRESS": harsh execution assumptions + higher fees/impact + execution lag
    """
    base_cfg = BacktestConfig() if base is None else base

    key = name.upper().strip()
    if key == "FAST":
        return FAST(base_cfg)
    if key == "REALISTIC":
        return REALISTIC(base_cfg)
    if key == "STRESS":
        return STRESS(base_cfg)
    raise ValueError(f"Unknown preset name: {name}")


def FAST(base: BacktestConfig | None = None) -> BacktestConfig:
    """
    Optimistic baseline:
    - execute at MID (no spread)
    - no impact
    - no fees
    """
    cfg = BacktestConfig() if base is None else base
    cfg = replace(
        cfg,
        execution=replace(cfg.execution, mode="MID", size_haircut=1.0, max_levels_to_cross=1, participation_cap=None),
        impact=replace(cfg.impact, mode="NONE", k_bps=0.0, noise_bps_std=0.0),
        fees=replace(cfg.fees, fee_bps=0.0, fee_per_unit=0.0),
        timing=replace(cfg.timing, exec_lag_bars=0),
        partial_fills=replace(cfg.partial_fills, allow_partial_fills=True, carry_unfilled=False),
    )
    return cfg


def REALISTIC(base: BacktestConfig | None = None) -> BacktestConfig:
    """
    Reasonable execution realism:
    - ORDERBOOK_WALK with a size haircut (you won't get all displayed size)
    - participation cap if volume/ADV is available
    - small explicit fees
    """
    cfg = BacktestConfig() if base is None else base
    cfg = replace(
        cfg,
        execution=replace(
            cfg.execution,
            mode="ORDERBOOK_WALK",
            size_haircut=0.6,
            max_levels_to_cross=5,
            reject_if_insufficient_depth=False,
            participation_cap=0.10,
        ),
        impact=replace(cfg.impact, mode="POWER", k_bps=5.0, alpha=0.5, liquidity_source="BAR_VOLUME"),
        fees=replace(cfg.fees, fee_bps=0.5, fee_mode="PER_NOTIONAL", fee_per_unit=0.0),
        timing=replace(cfg.timing, exec_lag_bars=0),
        partial_fills=replace(cfg.partial_fills, allow_partial_fills=True, carry_unfilled=True),
    )
    return cfg


def STRESS(base: BacktestConfig | None = None) -> BacktestConfig:
    """
    Harsh execution stress test:
    - deeper spreads/less liquidity simulated by strong haircut and limited depth
    - participation cap tight
    - lag execution by 1 bar
    - stronger impact and higher fees
    """
    cfg = BacktestConfig() if base is None else base
    cfg = replace(
        cfg,
        execution=replace(
            cfg.execution,
            mode="ORDERBOOK_WALK",
            size_haircut=0.3,
            max_levels_to_cross=2,
            reject_if_insufficient_depth=True,
            participation_cap=0.05,
        ),
        impact=replace(cfg.impact, mode="POWER", k_bps=15.0, alpha=0.7, liquidity_source="ADV"),
        fees=replace(cfg.fees, fee_bps=2.0, fee_mode="PER_NOTIONAL", fee_per_unit=0.0),
        timing=replace(cfg.timing, exec_lag_bars=1),
        partial_fills=replace(cfg.partial_fills, allow_partial_fills=False, carry_unfilled=False),
    )
    return cfg
