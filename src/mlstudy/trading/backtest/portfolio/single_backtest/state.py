"""Action codes and constants for the LP portfolio backtest loop."""

from enum import IntEnum


class PortfolioActionCode(IntEnum):
    """Bar-level outcome codes."""

    # -- No action --
    NO_ACTION = 0
    NO_CANDIDATES = 1
    SKIP_COOLDOWN = 2
    SKIP_COOLDOWN_RISK_ONLY = 3     # cooldown, but risk-reducing trades allowed

    # -- Execution outcomes --
    EXEC_OK = 100                   # LP solved, trades executed
    EXEC_PARTIAL = 101              # some candidates filled, some not
    EXEC_NO_LIQUIDITY = 102         # all candidates failed liquidity
    EXEC_GREEDY = 110               # greedy fallback used (no scipy)

    # -- LP / optimisation --
    LP_INFEASIBLE = 200             # LP returned infeasible
    LP_NO_CANDIDATES = 201          # all candidates filtered before LP

    # -- Market / data issues --
    INVALID_BOOK = 300              # bid > ask or non-positive prices
    INVALID_DV01 = 301              # zero / missing dv01


class TradeCode(IntEnum):
    """Per-trade fill outcome codes."""

    FILL_OK = 0
    FILL_PARTIAL = 1
    FILL_FAILED_LIQUIDITY = 2
    FILL_FAILED_BOOK = 3
    FILL_BELOW_MIN = 4              # filled < min_fill_ratio * requested


class CooldownMode(IntEnum):
    """What is allowed during cooldown."""

    BLOCK_ALL = 0       # no trades during cooldown
    RISK_REDUCING = 1   # only risk-reducing trades allowed


class FairType(IntEnum):
    """Which fair price was used for a trade."""

    DEC = 0     # risk-decreasing fair
    INC = 1     # risk-increasing fair
