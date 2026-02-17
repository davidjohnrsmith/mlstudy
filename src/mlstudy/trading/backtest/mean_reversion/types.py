"""
Attempt/outcome codes and constants for the mean-reversion backtester.

The canonical IntEnum definitions for the loop's action codes live in
:mod:`.state`.  This module re-exports ``ActionCode`` from there and
provides backward-compatible module-level aliases.

``State`` and ``TradeType`` are defined locally with short member names
(``FLAT`` vs ``STATE_FLAT``) for backward compatibility; their integer
values match :mod:`.state` exactly.

.. note::

   The JIT-compiled loop in :mod:`.loop` duplicates these values as bare
   ``int`` literals (prefixed with ``_``) because Numba ``@njit`` cannot
   resolve ``IntEnum`` members at compile time.  A test asserts parity.
"""

from __future__ import annotations

from enum import IntEnum

from .single_backtest.state import ActionCode, State as _LoopState


# ---------------------------------------------------------------------------
# Validate-scope constants
# ---------------------------------------------------------------------------


class ValidateScope(IntEnum):
    """Which legs to check for ``bid <= mid <= ask`` before trading."""

    REF_ONLY = 0
    ALL_LEGS = 1


# ---------------------------------------------------------------------------
# State constants  (short member names for backward compat)
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
# Values match what the loop in .loop actually emits (state.py ActionCode).
# ---------------------------------------------------------------------------

# ActionCode
NO_ACTION: int = ActionCode.NO_ACTION
NO_ACTION_NO_SIGNAL: int = ActionCode.NO_ACTION_NO_SIGNAL
NO_ACTION_HOLD: int = ActionCode.NO_ACTION_HOLD

ENTRY_OK: int = ActionCode.ENTRY_OK
ENTRY_INVALID_BOOK: int = ActionCode.ENTRY_FAILED_INVALID_BOOK
ENTRY_NO_LIQUIDITY: int = ActionCode.ENTRY_FAILED_NO_LIQUIDITY
ENTRY_TOO_WIDE: int = ActionCode.ENTRY_FAILED_TOO_WIDE
ENTRY_IN_COOLDOWN: int = ActionCode.ENTRY_FAILED_IN_COOLDOWN
# Deprecated aliases — mapped to closest new equivalent
ENTRY_NO_SIGNAL: int = ActionCode.NO_ACTION_NO_SIGNAL  # was 101, now 1
ENTRY_NOT_FLAT: int = 106  # no longer emitted by the loop

EXIT_TP_OK: int = ActionCode.EXIT_TP_OK
EXIT_TP_INVALID_BOOK: int = ActionCode.EXIT_FAILED_TP_INVALID_BOOK
EXIT_TP_NO_LIQUIDITY: int = ActionCode.EXIT_FAILED_TP_NO_LIQUIDITY
EXIT_TP_TOO_WIDE: int = ActionCode.EXIT_FAILED_TP_TOO_WIDE

EXIT_SL_OK: int = ActionCode.EXIT_SL_OK
EXIT_SL_NO_LIQUIDITY: int = ActionCode.EXIT_FAILED_SL_NO_LIQUIDITY
EXIT_SL_FORCED: int = ActionCode.EXIT_SL_OK  # was 307; SL is always forced

EXIT_TIME_OK: int = ActionCode.EXIT_TIME_OK
EXIT_TIME_NO_LIQUIDITY: int = ActionCode.EXIT_FAILED_TIME_NO_LIQUIDITY
EXIT_TIME_FORCED: int = ActionCode.EXIT_TIME_OK  # was 407; time-exit is always forced

# ValidateScope
VALIDATE_REF_ONLY: int = ValidateScope.REF_ONLY
VALIDATE_ALL_LEGS: int = ValidateScope.ALL_LEGS

# State
STATE_FLAT: int = State.FLAT
STATE_LONG: int = State.LONG
STATE_SHORT: int = State.SHORT

# Cooldown states (from .state — used in the loop's output)
STATE_TP_COOLDOWN: int = int(_LoopState.STATE_TP_COOLDOWN)
STATE_SL_COOLDOWN: int = int(_LoopState.STATE_SL_COOLDOWN)
STATE_TIME_COOLDOWN: int = int(_LoopState.STATE_TIME_COOLDOWN)

# TradeType
TRADE_ENTRY: int = TradeType.ENTRY
TRADE_EXIT_TP: int = TradeType.EXIT_TP
TRADE_EXIT_SL: int = TradeType.EXIT_SL
TRADE_EXIT_TIME: int = TradeType.EXIT_TIME


# ---------------------------------------------------------------------------
# Name lookup (debugging only – not used in hot loop)
# ---------------------------------------------------------------------------

CODE_NAMES: dict[int, str] = {member.value: member.name for member in ActionCode}
