"""
backtest/core/packer.py

Pack BacktestConfig (nested dataclasses) into small flat arrays for a JIT loop.

Rationale
- Keep jit_loop signature stable even as configs evolve
- Convert strings/enums -> int codes
- Bundle floats/ints/flags into dense vectors

This module should contain *no big arrays* (those are in BacktestInputs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import (
    BacktestConfig,
    ControlMode,
    ExecMode,
    ImpactMode,
    LiquiditySource,
    MappingMode,
    MissingBookAction,
    MissingMtmAction,
    MtmSource,
    NettingMode,
    PnLMethod,
)


@dataclass(frozen=True)
class PackedConfig:
    """
    Flat config to be consumed by jit_loop.run(...).
    Keep params_* ordering stable.
    """
    params_float: np.ndarray  # float64 vector
    params_int: np.ndarray    # int64 vector
    flags: np.ndarray         # int8 vector (0/1)


def _code_control_mode(s: str) -> int:
    return int(ControlMode.TARGET_POSITIONS if s == "target_positions" else ControlMode.ORDERS)


def _code_exec_mode(s: str) -> int:
    m = {
        "MID": ExecMode.MID,
        "TOP_OF_BOOK": ExecMode.TOP_OF_BOOK,
        "ORDERBOOK_WALK": ExecMode.ORDERBOOK_WALK,
        "LAST": ExecMode.LAST,
    }
    if s not in m:
        raise ValueError(f"Unknown execution.mode: {s}")
    return int(m[s])


def _code_netting(s: str) -> int:
    return int(NettingMode.NET if s == "NET" else NettingMode.CLOSE_THEN_OPEN)


def _code_impact_mode(s: str) -> int:
    m = {"NONE": ImpactMode.NONE, "POWER": ImpactMode.POWER, "STOCHASTIC": ImpactMode.STOCHASTIC}
    if s not in m:
        raise ValueError(f"Unknown impact.mode: {s}")
    return int(m[s])


def _code_liquidity_source(s: str) -> int:
    return int(LiquiditySource.BAR_VOLUME if s == "BAR_VOLUME" else LiquiditySource.ADV)


def _code_mapping_mode(s: str) -> int:
    m = {"NONE": MappingMode.NONE, "NOTIONAL_WEIGHTS": MappingMode.NOTIONAL_WEIGHTS, "DV01_WEIGHTS": MappingMode.DV01_WEIGHTS}
    if s not in m:
        raise ValueError(f"Unknown mapping.mapping_mode: {s}")
    return int(m[s])


def _code_missing_book_action(s: str) -> int:
    m = {"SKIP_TRADING": MissingBookAction.SKIP_TRADING, "FLATTEN": MissingBookAction.FLATTEN, "RAISE": MissingBookAction.RAISE}
    if s not in m:
        raise ValueError(f"Unknown data.missing_book_action: {s}")
    return int(m[s])


def _code_missing_mtm_action(s: str) -> int:
    m = {"DERIVE_FROM_BOOK": MissingMtmAction.DERIVE_FROM_BOOK, "FILL_FORWARD": MissingMtmAction.FILL_FORWARD, "RAISE": MissingMtmAction.RAISE}
    if s not in m:
        raise ValueError(f"Unknown data.missing_mtm_action: {s}")
    return int(m[s])


def _code_mtm_source(s: str) -> int:
    m = {"MID": MtmSource.MID, "LAST": MtmSource.LAST, "EXTERNAL_MTM": MtmSource.EXTERNAL_MTM}
    if s not in m:
        raise ValueError(f"Unknown accounting.mtm_source: {s}")
    return int(m[s])


def _code_pnl_method(s: str) -> int:
    return int(PnLMethod.PRICE_MTM if s == "PRICE_MTM" else PnLMethod.YIELD_DV01_APPROX)


def pack_config(cfg: BacktestConfig) -> PackedConfig:
    """
    Convert nested config into:
    - params_float: numeric float parameters
    - params_int: numeric int parameters
    - flags: booleans as int8
    """

    # ---- floats (stable ordering) ----
    # NOTE: keep this ordering stable across versions.
    f = []
    f.append(float(cfg.instrument.contract_multiplier))
    f.append(float(cfg.instrument.lot_size))
    f.append(float(cfg.execution.size_haircut))
    f.append(float(cfg.fees.fee_bps))
    f.append(float(cfg.fees.fee_per_unit))
    f.append(float(cfg.impact.k_bps))
    f.append(float(cfg.impact.alpha))
    f.append(float(cfg.impact.noise_bps_std))
    f.append(float(cfg.risk.max_abs_pos_per_leg))
    f.append(float(cfg.risk.max_gross_pos))
    f.append(float(cfg.risk.max_net_pos))
    f.append(float(cfg.risk.max_gross_dv01))
    f.append(float(cfg.risk.max_net_dv01))
    f.append(float(cfg.rounding.min_trade_size))
    f.append(float(cfg.rounding.round_to_size))
    # participation_cap can be None -> -1 sentinel
    f.append(float(cfg.execution.participation_cap) if cfg.execution.participation_cap is not None else -1.0)
    # drawdown can be None -> -1 sentinel
    f.append(float(cfg.risk.max_drawdown) if cfg.risk.max_drawdown is not None else -1.0)
    # cash rate
    f.append(float(cfg.accounting.cash_rate))
    # borrow spread
    f.append(float(cfg.accounting.borrow_spread_bps))

    params_float = np.asarray(f, dtype=np.float64)

    # ---- ints (stable ordering) ----
    i = []
    i.append(int(_code_control_mode(cfg.control.control_mode)))
    i.append(int(_code_exec_mode(cfg.execution.mode)))
    i.append(int(cfg.timing.exec_lag_bars))
    i.append(int(cfg.execution.max_levels_to_cross))
    i.append(int(_code_netting(cfg.rounding.netting)))
    i.append(int(_code_impact_mode(cfg.impact.mode)))
    i.append(int(_code_liquidity_source(cfg.impact.liquidity_source)))
    i.append(int(_code_mapping_mode(cfg.mapping.mapping_mode)))
    i.append(int(cfg.mapping.weights_lag_bars))
    i.append(int(_code_missing_book_action(cfg.data.missing_book_action)))
    i.append(int(_code_missing_mtm_action(cfg.data.missing_mtm_action)))
    i.append(int(_code_mtm_source(cfg.accounting.mtm_source)))
    i.append(int(_code_pnl_method(cfg.accounting.pnl_method)))
    # stale threshold
    i.append(int(cfg.data.stale_bars_threshold))
    params_int = np.asarray(i, dtype=np.int64)

    # ---- flags (stable ordering) ----
    fl = []
    fl.append(1 if cfg.partial_fills.allow_partial_fills else 0)
    fl.append(1 if cfg.partial_fills.carry_unfilled else 0)
    fl.append(1 if cfg.execution.reject_if_insufficient_depth else 0)
    fl.append(1 if cfg.mapping.enforce_dv01_neutral else 0)
    fl.append(1 if cfg.risk.enabled else 0)
    fl.append(1 if cfg.data.require_all_legs_present else 0)
    fl.append(1 if cfg.accounting.cash_interest_enabled else 0)
    fl.append(1 if cfg.accounting.funding_enabled else 0)
    fl.append(1 if cfg.accounting.include_coupon_accrual else 0)
    flags = np.asarray(fl, dtype=np.int8)

    return PackedConfig(params_float=params_float, params_int=params_int, flags=flags)
