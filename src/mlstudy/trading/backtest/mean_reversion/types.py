"""
Attempt/outcome codes and constants for the mean-reversion backtester.

Layout: category * 100 + suffix
  Category: 0=NO_ACTION, 1=ENTRY, 2=EXIT_TP, 3=EXIT_SL, 4=EXIT_TIME
  Suffix:   0=OK, 1=NO_SIGNAL, 2=INVALID_BOOK, 3=NO_LIQUIDITY,
            4=TOO_WIDE, 5=IN_COOLDOWN, 6=NOT_FLAT, 7=FORCED

All values are ``IntEnum`` members (and therefore plain ``int`` subclasses)
so they work in arithmetic, comparisons, and as numpy dtype values.

.. note::

   The JIT-compiled loop in :mod:`.loop` duplicates these values as bare
   ``int`` literals (prefixed with ``_``) because Numba ``@njit`` cannot
   resolve ``IntEnum`` members at compile time.  A test asserts parity.
"""

from __future__ import annotations

from enum import IntEnum


# ---------------------------------------------------------------------------
# Attempt / outcome codes
# ---------------------------------------------------------------------------


class ActionCode(IntEnum):
    """Outcome code written to ``out_codes`` on every bar.

    Encoding: ``category * 100 + suffix``.

    ====  ============  ====================================================
    Cat   Prefix        Meaning
    ====  ============  ====================================================
    0     NO_ACTION     No entry/exit attempted (flat & quiet, or holding).
    1     ENTRY_*       Entry attempted.
    2     EXIT_TP_*     Take-profit exit attempted.
    3     EXIT_SL_*     Stop-loss exit attempted.
    4     EXIT_TIME_*   Time-limit exit attempted.
    ====  ============  ====================================================
    """

    NO_ACTION = 0

    ENTRY_OK = 100
    ENTRY_NO_SIGNAL = 101
    ENTRY_INVALID_BOOK = 102
    ENTRY_NO_LIQUIDITY = 103
    ENTRY_TOO_WIDE = 104
    ENTRY_IN_COOLDOWN = 105
    ENTRY_NOT_FLAT = 106

    EXIT_TP_OK = 200
    EXIT_TP_INVALID_BOOK = 202
    EXIT_TP_NO_LIQUIDITY = 203
    EXIT_TP_TOO_WIDE = 204

    EXIT_SL_OK = 300
    EXIT_SL_NO_LIQUIDITY = 303
    EXIT_SL_FORCED = 307

    EXIT_TIME_OK = 400
    EXIT_TIME_NO_LIQUIDITY = 403
    EXIT_TIME_FORCED = 407


# ---------------------------------------------------------------------------
# Validate-scope constants
# ---------------------------------------------------------------------------


class ValidateScope(IntEnum):
    """Which legs to check for ``bid <= mid <= ask`` before trading."""

    REF_ONLY = 0
    ALL_LEGS = 1


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------


class State(IntEnum):
    """Position state of the backtester at the end of a bar."""

    FLAT = 0
    LONG = 1
    SHORT = -1


# ---------------------------------------------------------------------------
# Trade-record type codes (used in the ``tr_type`` output array)
# ---------------------------------------------------------------------------


class TradeType(IntEnum):
    """Type tag stored in the ``tr_type`` per-trade output array."""

    ENTRY = 0
    EXIT_TP = 1
    EXIT_SL = 2
    EXIT_TIME = 3


# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases
#
# Every consumer currently does ``from .types import ENTRY_OK, ...`` so we
# re-export every member as a module-level name.  Since IntEnum members ARE
# ints, existing ``==`` / ``!=`` / dict-key / numpy comparisons keep working.
# ---------------------------------------------------------------------------

# ActionCode
NO_ACTION: int = ActionCode.NO_ACTION

ENTRY_OK: int = ActionCode.ENTRY_OK
ENTRY_NO_SIGNAL: int = ActionCode.ENTRY_NO_SIGNAL
ENTRY_INVALID_BOOK: int = ActionCode.ENTRY_INVALID_BOOK
ENTRY_NO_LIQUIDITY: int = ActionCode.ENTRY_NO_LIQUIDITY
ENTRY_TOO_WIDE: int = ActionCode.ENTRY_TOO_WIDE
ENTRY_IN_COOLDOWN: int = ActionCode.ENTRY_IN_COOLDOWN
ENTRY_NOT_FLAT: int = ActionCode.ENTRY_NOT_FLAT

EXIT_TP_OK: int = ActionCode.EXIT_TP_OK
EXIT_TP_INVALID_BOOK: int = ActionCode.EXIT_TP_INVALID_BOOK
EXIT_TP_NO_LIQUIDITY: int = ActionCode.EXIT_TP_NO_LIQUIDITY
EXIT_TP_TOO_WIDE: int = ActionCode.EXIT_TP_TOO_WIDE

EXIT_SL_OK: int = ActionCode.EXIT_SL_OK
EXIT_SL_NO_LIQUIDITY: int = ActionCode.EXIT_SL_NO_LIQUIDITY
EXIT_SL_FORCED: int = ActionCode.EXIT_SL_FORCED

EXIT_TIME_OK: int = ActionCode.EXIT_TIME_OK
EXIT_TIME_NO_LIQUIDITY: int = ActionCode.EXIT_TIME_NO_LIQUIDITY
EXIT_TIME_FORCED: int = ActionCode.EXIT_TIME_FORCED

# ValidateScope
VALIDATE_REF_ONLY: int = ValidateScope.REF_ONLY
VALIDATE_ALL_LEGS: int = ValidateScope.ALL_LEGS

# State
STATE_FLAT: int = State.FLAT
STATE_LONG: int = State.LONG
STATE_SHORT: int = State.SHORT

# TradeType
TRADE_ENTRY: int = TradeType.ENTRY
TRADE_EXIT_TP: int = TradeType.EXIT_TP
TRADE_EXIT_SL: int = TradeType.EXIT_SL
TRADE_EXIT_TIME: int = TradeType.EXIT_TIME


# ---------------------------------------------------------------------------
# Name lookup (debugging only – not used in hot loop)
# ---------------------------------------------------------------------------

CODE_NAMES: dict[int, str] = {member.value: member.name for member in ActionCode}
