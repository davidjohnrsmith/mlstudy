# """Mean-reversion backtester with L2 book execution and explicit state machine."""
#
# from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
# from mlstudy.trading.backtest.mean_reversion.configs.sweep_config import SweepConfig, load_sweep_config
# from mlstudy.trading.backtest.mean_reversion.single_backtest.engine import run_backtest
# from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults
# from mlstudy.trading.backtest.mean_reversion.sweep.sweep_rank import RankingPlan
# from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import FullScenario, SweepRunData, \
#     load_sweep_run
# from mlstudy.trading.backtest.mean_reversion.sweep.sweep_runner import SweepRunResult, run_sweep_from_config
# from . import analysis, sweep
# from .configs.utils import load_config_map, load_sweep_config_by_name
# from mlstudy.trading.backtest.mean_reversion.data.data_loader import BacktestDataLoader, MarketData
# from .sweep.sweep_types import SweepSummary, MetricsOnlyResult
#
# try:
#     from . import plots
# except ImportError:
#     plots = None  # type: ignore[assignment]
# from .types import (
#     # IntEnum classes
#     ActionCode,
#     State,
#     TradeType,
#     ValidateScope,
#     # backwards-compatible aliases
#     CODE_NAMES,
#     ENTRY_IN_COOLDOWN,
#     ENTRY_INVALID_BOOK,
#     ENTRY_NO_LIQUIDITY,
#     ENTRY_NO_SIGNAL,
#     ENTRY_NOT_FLAT,
#     ENTRY_OK,
#     ENTRY_TOO_WIDE,
#     EXIT_SL_FORCED,
#     EXIT_SL_NO_LIQUIDITY,
#     EXIT_SL_OK,
#     EXIT_TIME_FORCED,
#     EXIT_TIME_NO_LIQUIDITY,
#     EXIT_TIME_OK,
#     EXIT_TP_INVALID_BOOK,
#     EXIT_TP_NO_LIQUIDITY,
#     EXIT_TP_OK,
#     EXIT_TP_TOO_WIDE,
#     NO_ACTION,
#     NO_ACTION_HOLD,
#     NO_ACTION_NO_SIGNAL,
#     STATE_FLAT,
#     STATE_LONG,
#     STATE_SHORT,
#     STATE_SL_COOLDOWN,
#     STATE_TIME_COOLDOWN,
#     STATE_TP_COOLDOWN,
#     TRADE_ENTRY,
#     TRADE_EXIT_SL,
#     TRADE_EXIT_TIME,
#     TRADE_EXIT_TP,
#     VALIDATE_ALL_LEGS,
#     VALIDATE_REF_ONLY,
# )
#
# __all__ = [
#     "MRBacktestResults",
#     "run_backtest",
#     # codes
#     "NO_ACTION",
#     "ENTRY_OK",
#     "ENTRY_NO_SIGNAL",
#     "ENTRY_INVALID_BOOK",
#     "ENTRY_NO_LIQUIDITY",
#     "ENTRY_TOO_WIDE",
#     "ENTRY_IN_COOLDOWN",
#     "ENTRY_NOT_FLAT",
#     "EXIT_TP_OK",
#     "EXIT_TP_INVALID_BOOK",
#     "EXIT_TP_NO_LIQUIDITY",
#     "EXIT_TP_TOO_WIDE",
#     "EXIT_SL_OK",
#     "EXIT_SL_NO_LIQUIDITY",
#     "EXIT_SL_FORCED",
#     "EXIT_TIME_OK",
#     "EXIT_TIME_NO_LIQUIDITY",
#     "EXIT_TIME_FORCED",
#     # state
#     "STATE_FLAT",
#     "STATE_LONG",
#     "STATE_SHORT",
#     "STATE_TP_COOLDOWN",
#     "STATE_SL_COOLDOWN",
#     "STATE_TIME_COOLDOWN",
#     # new action codes
#     "NO_ACTION_NO_SIGNAL",
#     "NO_ACTION_HOLD",
#     # trade types
#     "TRADE_ENTRY",
#     "TRADE_EXIT_TP",
#     "TRADE_EXIT_SL",
#     "TRADE_EXIT_TIME",
#     # validate scope
#     "VALIDATE_REF_ONLY",
#     "VALIDATE_ALL_LEGS",
#     # IntEnum classes
#     "ActionCode",
#     "State",
#     "TradeType",
#     "ValidateScope",
#     # helpers
#     "CODE_NAMES",
#     # sweep dataclasses
#     "MetricsOnlyResult",
#     "SweepSummary",
#     "RankingPlan",
#     # sweep config
#     "SweepConfig",
#     "load_sweep_config",
#     # sweep runner
#     "SweepRunResult",
#     "run_sweep_from_config",
#     # sweep results reader
#     "SweepRunData",
#     "FullScenario",
#     "load_sweep_run",
#     # data loader
#     "BacktestDataLoader",
#     "MarketData",
#     # analysis, plots & sweep
#     "analysis",
#     "plots",
#     "sweep",
# ]
