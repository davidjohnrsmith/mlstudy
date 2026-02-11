"""
Attempt/outcome codes and constants for the mean-reversion backtester.

Layout: category * 100 + suffix
  Category: 0=NO_ACTION, 1=ENTRY, 2=EXIT_TP, 3=EXIT_SL, 4=EXIT_TIME
  Suffix:   0=OK, 1=NO_SIGNAL, 2=INVALID_BOOK, 3=NO_LIQUIDITY,
            4=TOO_WIDE, 5=IN_COOLDOWN, 6=NOT_FLAT, 7=FORCED

All values are plain ints so they work inside Numba ``@njit`` functions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Attempt / outcome codes
# ---------------------------------------------------------------------------

NO_ACTION: int = 0

ENTRY_OK: int = 100
ENTRY_NO_SIGNAL: int = 101
ENTRY_INVALID_BOOK: int = 102
ENTRY_NO_LIQUIDITY: int = 103
ENTRY_TOO_WIDE: int = 104
ENTRY_IN_COOLDOWN: int = 105
ENTRY_NOT_FLAT: int = 106

EXIT_TP_OK: int = 200
EXIT_TP_INVALID_BOOK: int = 202
EXIT_TP_NO_LIQUIDITY: int = 203
EXIT_TP_TOO_WIDE: int = 204

EXIT_SL_OK: int = 300
EXIT_SL_NO_LIQUIDITY: int = 303
EXIT_SL_FORCED: int = 307

EXIT_TIME_OK: int = 400
EXIT_TIME_NO_LIQUIDITY: int = 403
EXIT_TIME_FORCED: int = 407

# ---------------------------------------------------------------------------
# Validate-scope constants
# ---------------------------------------------------------------------------

VALIDATE_REF_ONLY: int = 0
VALIDATE_ALL_LEGS: int = 1

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_FLAT: int = 0
STATE_LONG: int = 1
STATE_SHORT: int = -1

# ---------------------------------------------------------------------------
# Trade-record type codes (used in the ``tr_type`` output array)
# ---------------------------------------------------------------------------

TRADE_ENTRY: int = 0
TRADE_EXIT_TP: int = 1
TRADE_EXIT_SL: int = 2
TRADE_EXIT_TIME: int = 3

# ---------------------------------------------------------------------------
# Name lookup (debugging only – not used in hot loop)
# ---------------------------------------------------------------------------

CODE_NAMES: dict[int, str] = {
    NO_ACTION: "NO_ACTION",
    ENTRY_OK: "ENTRY_OK",
    ENTRY_NO_SIGNAL: "ENTRY_NO_SIGNAL",
    ENTRY_INVALID_BOOK: "ENTRY_INVALID_BOOK",
    ENTRY_NO_LIQUIDITY: "ENTRY_NO_LIQUIDITY",
    ENTRY_TOO_WIDE: "ENTRY_TOO_WIDE",
    ENTRY_IN_COOLDOWN: "ENTRY_IN_COOLDOWN",
    ENTRY_NOT_FLAT: "ENTRY_NOT_FLAT",
    EXIT_TP_OK: "EXIT_TP_OK",
    EXIT_TP_INVALID_BOOK: "EXIT_TP_INVALID_BOOK",
    EXIT_TP_NO_LIQUIDITY: "EXIT_TP_NO_LIQUIDITY",
    EXIT_TP_TOO_WIDE: "EXIT_TP_TOO_WIDE",
    EXIT_SL_OK: "EXIT_SL_OK",
    EXIT_SL_NO_LIQUIDITY: "EXIT_SL_NO_LIQUIDITY",
    EXIT_SL_FORCED: "EXIT_SL_FORCED",
    EXIT_TIME_OK: "EXIT_TIME_OK",
    EXIT_TIME_NO_LIQUIDITY: "EXIT_TIME_NO_LIQUIDITY",
    EXIT_TIME_FORCED: "EXIT_TIME_FORCED",
}
