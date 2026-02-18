from enum import IntEnum


class ActionCode(IntEnum):
    # -- ActionCode --
    NO_ACTION = 0
    NO_ACTION_NO_SIGNAL = 1
    NO_ACTION_HOLD = 2

    ENTRY_OK = 100
    ENTRY_FAILED_INVALID_DV01 = 101
    ENTRY_FAILED_INVALID_BOOK = 102
    ENTRY_FAILED_TOO_WIDE = 103
    ENTRY_FAILED_NO_LIQUIDITY = 104
    ENTRY_FAILED_IN_COOLDOWN = 105

    EXIT_TP_OK = 200
    EXIT_TP_OK_WITH_COOLDOWN = 210
    EXIT_FAILED_TP_INVALID_BOOK = 202
    EXIT_FAILED_TP_TOO_WIDE = 203
    EXIT_FAILED_TP_NO_LIQUIDITY = 204

    EXIT_SL_OK = 300
    EXIT_SL_OK_WITH_COOLDOWN = 310
    EXIT_FAILED_SL_INVALID_BOOK = 302
    EXIT_FAILED_SL_NO_LIQUIDITY = 304

    EXIT_TIME_OK = 400
    EXIT_TIME_OK_WITH_COOLDOWN = 410
    EXIT_FAILED_TIME_INVALID_BOOK = 402
    EXIT_FAILED_TIME_NO_LIQUIDITY = 404


class State(IntEnum):
    # -- State --
    STATE_FLAT = 0
    STATE_LONG = 1
    STATE_SHORT = -1
    STATE_TP_COOLDOWN = 2
    STATE_SL_COOLDOWN = 3
    STATE_TIME_COOLDOWN = 4


class TradeType(IntEnum):
    # -- TradeType --
    TRADE_ENTRY = 0
    TRADE_EXIT_TP = 1
    TRADE_EXIT_SL = 2
    TRADE_EXIT_TIME = 3


NO_POSITION_STATES = (
    State.STATE_FLAT.value,
    State.STATE_TP_COOLDOWN.value,
    State.STATE_SL_COOLDOWN.value,
    State.STATE_TIME_COOLDOWN.value,
)


# ---------------------------------------------------------------------------
# Validate-scope constants
# ---------------------------------------------------------------------------


class ValidateScope(IntEnum):
    """Which legs to check for ``bid <= mid <= ask`` before trading."""

    REF_ONLY = 0
    ALL_LEGS = 1


# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases
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
STATE_FLAT: int = State.STATE_FLAT
STATE_LONG: int = State.STATE_LONG
STATE_SHORT: int = State.STATE_SHORT

# Cooldown states
STATE_TP_COOLDOWN: int = int(State.STATE_TP_COOLDOWN)
STATE_SL_COOLDOWN: int = int(State.STATE_SL_COOLDOWN)
STATE_TIME_COOLDOWN: int = int(State.STATE_TIME_COOLDOWN)

# TradeType
TRADE_ENTRY: int = TradeType.TRADE_ENTRY
TRADE_EXIT_TP: int = TradeType.TRADE_EXIT_TP
TRADE_EXIT_SL: int = TradeType.TRADE_EXIT_SL
TRADE_EXIT_TIME: int = TradeType.TRADE_EXIT_TIME

# ---------------------------------------------------------------------------
# Name lookup (debugging only – not used in hot loop)
# ---------------------------------------------------------------------------

CODE_NAMES = {member.value: member.name for member in ActionCode}
