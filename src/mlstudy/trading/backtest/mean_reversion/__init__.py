"""Mean-reversion backtester with L2 book execution and explicit state machine."""

from .engine import MRBacktestConfig, run_backtest
from .results import MRBacktestResults
from .types import (
    CODE_NAMES,
    ENTRY_IN_COOLDOWN,
    ENTRY_INVALID_BOOK,
    ENTRY_NO_LIQUIDITY,
    ENTRY_NO_SIGNAL,
    ENTRY_NOT_FLAT,
    ENTRY_OK,
    ENTRY_TOO_WIDE,
    EXIT_SL_FORCED,
    EXIT_SL_NO_LIQUIDITY,
    EXIT_SL_OK,
    EXIT_TIME_FORCED,
    EXIT_TIME_NO_LIQUIDITY,
    EXIT_TIME_OK,
    EXIT_TP_INVALID_BOOK,
    EXIT_TP_NO_LIQUIDITY,
    EXIT_TP_OK,
    EXIT_TP_TOO_WIDE,
    NO_ACTION,
    STATE_FLAT,
    STATE_LONG,
    STATE_SHORT,
    TRADE_ENTRY,
    TRADE_EXIT_SL,
    TRADE_EXIT_TIME,
    TRADE_EXIT_TP,
    VALIDATE_ALL_LEGS,
    VALIDATE_REF_ONLY,
)

__all__ = [
    "MRBacktestConfig",
    "MRBacktestResults",
    "run_backtest",
    # codes
    "NO_ACTION",
    "ENTRY_OK",
    "ENTRY_NO_SIGNAL",
    "ENTRY_INVALID_BOOK",
    "ENTRY_NO_LIQUIDITY",
    "ENTRY_TOO_WIDE",
    "ENTRY_IN_COOLDOWN",
    "ENTRY_NOT_FLAT",
    "EXIT_TP_OK",
    "EXIT_TP_INVALID_BOOK",
    "EXIT_TP_NO_LIQUIDITY",
    "EXIT_TP_TOO_WIDE",
    "EXIT_SL_OK",
    "EXIT_SL_NO_LIQUIDITY",
    "EXIT_SL_FORCED",
    "EXIT_TIME_OK",
    "EXIT_TIME_NO_LIQUIDITY",
    "EXIT_TIME_FORCED",
    # state
    "STATE_FLAT",
    "STATE_LONG",
    "STATE_SHORT",
    # trade types
    "TRADE_ENTRY",
    "TRADE_EXIT_TP",
    "TRADE_EXIT_SL",
    "TRADE_EXIT_TIME",
    # validate scope
    "VALIDATE_REF_ONLY",
    "VALIDATE_ALL_LEGS",
    # helpers
    "CODE_NAMES",
]
