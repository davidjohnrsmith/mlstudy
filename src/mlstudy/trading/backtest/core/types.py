"""
backtest/core/types.py

Central type definitions for the backtester:
- BacktestInputs: large, time-varying arrays (L2 order book + controls + mapping)
- BacktestConfig: small knobs (execution model, impact, fees, risk, data policy, accounting)
- Enums: side, execution mode, reject reasons, etc.

Notes
- Underlying instruments ("legs") dimension is N.
- Traded objects ("portfolios": switch/fly/custom baskets) dimension is M.
- Order book depth dimension is L.
- Controls are always in portfolio space (M). If you trade legs directly, set M=N and mapping_mode="NONE".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Optional

import numpy as np


# =============================================================================
# Enums (int-based for easy JIT packing)
# =============================================================================

class Side(IntEnum):
    SELL = -1
    BUY = 1


class ControlMode(IntEnum):
    TARGET_POSITIONS = 0
    ORDERS = 1


class ExecMode(IntEnum):
    MID = 0
    TOP_OF_BOOK = 1
    ORDERBOOK_WALK = 2
    LAST = 3


class RejectReason(IntEnum):
    NONE = 0
    NOT_TRADABLE = 1
    INSUFFICIENT_DEPTH = 2
    PARTICIPATION_CAP = 3
    MISSING_BOOK = 4
    MISSING_MTM = 5
    RISK_REJECT = 6
    MIN_TRADE = 7


class MappingMode(IntEnum):
    NONE = 0
    NOTIONAL_WEIGHTS = 1
    DV01_WEIGHTS = 2


class NettingMode(IntEnum):
    NET = 0
    CLOSE_THEN_OPEN = 1


class MissingBookAction(IntEnum):
    SKIP_TRADING = 0
    FLATTEN = 1
    RAISE = 2


class MissingMtmAction(IntEnum):
    DERIVE_FROM_BOOK = 0
    FILL_FORWARD = 1
    RAISE = 2


class MtmSource(IntEnum):
    MID = 0
    LAST = 1
    EXTERNAL_MTM = 2


class PnLMethod(IntEnum):
    PRICE_MTM = 0
    YIELD_DV01_APPROX = 1


class LiquiditySource(IntEnum):
    BAR_VOLUME = 0
    ADV = 1


class ImpactMode(IntEnum):
    NONE = 0
    POWER = 1
    STOCHASTIC = 2


# =============================================================================
# BacktestInputs: large arrays
# =============================================================================

@dataclass(frozen=True)
class OrderBookL2:
    """
    Level-2 order book snapshots.
    Shapes: (T, N, L)
    - Level 0 is best bid/ask.
    - Sizes are in "size units" (see InstrumentSpec in config).
    """
    bid_px: np.ndarray  # (T, N, L)
    bid_sz: np.ndarray  # (T, N, L)
    ask_px: np.ndarray  # (T, N, L)
    ask_sz: np.ndarray  # (T, N, L)

    # Optional trade prints / aggregates
    last_px: Optional[np.ndarray] = None       # (T, N)
    last_sz: Optional[np.ndarray] = None       # (T, N)

    # Optional volume/liquidity proxies
    traded_volume: Optional[np.ndarray] = None # (T, N) in same size units
    adv: Optional[np.ndarray] = None           # (T, N) in same size units


@dataclass(frozen=True)
class MarketState:
    """
    Optional market state used for rates risk and/or alternative MTM.
    If you run pure price-space execution + MTM, you can omit yields/dv01s.
    """
    mtm_px: Optional[np.ndarray] = None        # (T, N) external mark price
    yields: Optional[np.ndarray] = None        # (T, N)
    dv01s: Optional[np.ndarray] = None         # (T, N)

    # Carry/funding optional inputs
    coupons: Optional[np.ndarray] = None       # (N,) or (T, N)
    accrual_factors: Optional[np.ndarray] = None  # (T, N)
    repo_rates: Optional[np.ndarray] = None    # (T,) or (T, N)


@dataclass(frozen=True)
class PortfolioMapping:
    """
    Mapping from traded portfolios (M) to underlying legs (N).
    W[t, m, i] is weight of leg i in portfolio m at time t.

    If mapping_mode="NONE", W can be None and controls must be in leg space (M=N).
    """
    W: Optional[np.ndarray] = None             # (T, M, N)
    W_lag_bars: int = 0                        # avoid lookahead


@dataclass(frozen=True)
class Controls:
    """
    Strategy output / instruction stream.
    Provide exactly one based on BacktestConfig.control.control_mode:
      - target_positions: desired holdings per bar
      - orders: signed trade sizes per bar
    Shapes: (T, M)
    """
    target_positions: Optional[np.ndarray] = None  # (T, M)
    orders: Optional[np.ndarray] = None            # (T, M)


@dataclass(frozen=True)
class Availability:
    """
    Optional masks to block trading.
    Shapes: (T, N)
    """
    active_mask: Optional[np.ndarray] = None       # 1/0 instrument is live
    tradable_mask: Optional[np.ndarray] = None     # 1/0 tradable (halt/missing book)


@dataclass(frozen=True)
class BacktestInputs:
    """
    Large arrays only. Reusable across many BacktestConfig scenarios.
    """
    datetimes_int: np.ndarray                      # (T,)

    orderbook: OrderBookL2
    market: MarketState = MarketState()
    mapping: PortfolioMapping = PortfolioMapping()
    controls: Controls = Controls()
    availability: Availability = Availability()


# =============================================================================
# BacktestConfig: small knobs
# =============================================================================

@dataclass(frozen=True)
class InstrumentSpec:
    """
    Interprets sizes and converts price * size into currency PnL if needed.
    - size_unit defines the unit of book sizes and control quantities.
    - contract_multiplier scales (price change * size) into currency (futures).
    """
    size_unit: Literal["NOTIONAL", "CONTRACTS", "SHARES", "BOND_FACE"] = "CONTRACTS"
    contract_multiplier: float = 1.0
    lot_size: float = 1.0
    tick_size: Optional[float] = None


@dataclass(frozen=True)
class ControlModeConfig:
    control_mode: Literal["target_positions", "orders"] = "target_positions"


@dataclass(frozen=True)
class ExecutionTimingConfig:
    """
    Execution lag relative to bar index:
    - exec_lag_bars=0 uses book at bar t
    - exec_lag_bars=1 uses book at bar t+1
    """
    exec_lag_bars: int = 0


@dataclass(frozen=True)
class ExecutionModelConfig:
    """
    How to convert an order into a fill using the L2 book.
    """
    mode: Literal["MID", "TOP_OF_BOOK", "ORDERBOOK_WALK", "LAST"] = "ORDERBOOK_WALK"

    size_haircut: float = 1.0              # fraction of displayed depth usable
    max_levels_to_cross: int = 999
    reject_if_insufficient_depth: bool = False

    participation_cap: Optional[float] = None  # <= cap * volume/ADV (if available)


@dataclass(frozen=True)
class PartialFillPolicyConfig:
    allow_partial_fills: bool = True
    carry_unfilled: bool = False


@dataclass(frozen=True)
class RoundingConfig:
    min_trade_size: float = 0.0
    round_to_size: float = 0.0
    netting: Literal["NET", "CLOSE_THEN_OPEN"] = "NET"


@dataclass(frozen=True)
class ImpactModelConfig:
    """
    Extra adverse movement beyond book-walk (optional).
    """
    mode: Literal["NONE", "POWER", "STOCHASTIC"] = "NONE"

    # POWER: impact_bps = k_bps * (|trade| / liquidity)^alpha
    k_bps: float = 0.0
    alpha: float = 0.5
    liquidity_source: Literal["BAR_VOLUME", "ADV"] = "BAR_VOLUME"

    # STOCHASTIC: add random bps noise
    rng_seed: int = 0
    noise_bps_std: float = 0.0


@dataclass(frozen=True)
class FeesConfig:
    """
    Explicit fees/commissions separate from execution price.
    """
    fee_bps: float = 0.0
    fee_mode: Literal["PER_NOTIONAL", "PER_SIZE_UNIT"] = "PER_NOTIONAL"
    fee_per_unit: float = 0.0


@dataclass(frozen=True)
class MappingConfig:
    mapping_mode: Literal["NONE", "NOTIONAL_WEIGHTS", "DV01_WEIGHTS"] = "NONE"
    enforce_dv01_neutral: bool = False
    weights_lag_bars: int = 0


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = True

    # Position limits in "size units" (or notional) for legs
    max_abs_pos_per_leg: float = np.inf
    max_gross_pos: float = np.inf
    max_net_pos: float = np.inf

    # Rates-specific limits (requires dv01s)
    max_gross_dv01: float = np.inf
    max_net_dv01: float = np.inf

    max_drawdown: Optional[float] = None
    action_on_breach: Literal["CLIP", "FLATTEN", "REJECT"] = "CLIP"


@dataclass(frozen=True)
class DataPolicyConfig:
    missing_book_action: Literal["SKIP_TRADING", "FLATTEN", "RAISE"] = "SKIP_TRADING"
    missing_mtm_action: Literal["DERIVE_FROM_BOOK", "FILL_FORWARD", "RAISE"] = "DERIVE_FROM_BOOK"
    stale_bars_threshold: int = 0
    require_all_legs_present: bool = True


@dataclass(frozen=True)
class AccountingConfig:
    mtm_source: Literal["MID", "LAST", "EXTERNAL_MTM"] = "MID"
    pnl_method: Literal["PRICE_MTM", "YIELD_DV01_APPROX"] = "PRICE_MTM"

    cash_interest_enabled: bool = False
    cash_rate: float = 0.0

    funding_enabled: bool = False
    borrow_spread_bps: float = 0.0
    include_coupon_accrual: bool = False


@dataclass(frozen=True)
class PortfolioInitConfig:
    initial_capital: float = 0.0
    initial_positions: Optional[np.ndarray] = None  # (M,) or (N,) depending on store_positions_as
    store_positions_as: Literal["PORTFOLIO", "LEGS"] = "LEGS"


@dataclass(frozen=True)
class BacktestConfig:
    """
    Small knobs only. Safe to serialize aside from initial_positions ndarray.
    """
    instrument: InstrumentSpec = InstrumentSpec()
    control: ControlModeConfig = ControlModeConfig()

    timing: ExecutionTimingConfig = ExecutionTimingConfig()
    execution: ExecutionModelConfig = ExecutionModelConfig()
    partial_fills: PartialFillPolicyConfig = PartialFillPolicyConfig()
    rounding: RoundingConfig = RoundingConfig()

    impact: ImpactModelConfig = ImpactModelConfig()
    fees: FeesConfig = FeesConfig()

    mapping: MappingConfig = MappingConfig()
    risk: RiskConfig = RiskConfig()
    data: DataPolicyConfig = DataPolicyConfig()
    accounting: AccountingConfig = AccountingConfig()
    portfolio: PortfolioInitConfig = PortfolioInitConfig()
