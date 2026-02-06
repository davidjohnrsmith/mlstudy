# """Portfolio interface for multi-strategy trading.
#
# Provides standard types for strategy signals and portfolio-level aggregation,
# plus adapters to convert existing strategy outputs to the standard format.
# """
#
# from __future__ import annotations
#
# from mlstudy.trading.portfolio.aggregate import (
#     AggregationResult,
#     SizingRules,
#     aggregate_batches,
#     aggregate_signal_batch,
#     compute_position_changes,
#     compute_turnover,
#     create_sizing_rules_with_weighting,
#     signals_to_bond_targets,
# )
# from mlstudy.trading.portfolio.backtest import (
#     PortfolioBacktestConfig,
#     PortfolioBacktestResult,
#     RebalanceMode,
#     RebalanceRule,
#     Trade,
#     compute_costs_summary,
#     compute_pnl_from_prices,
#     run_portfolio_backtest,
#     run_portfolio_backtest_from_targets,
#     simulate_rebalance,
# )
#
# # __all__ = [
# #     # Types
# #     "LegWeight",
# #     "StrategySignal",
# #     "StrategySignalBatch",
# #     "PortfolioTarget",
# #     "signals_to_dataframe",
# #     # Aggregation
# #     "SizingRules",
# #     "AggregationResult",
# #     "signals_to_bond_targets",
# #     "aggregate_signal_batch",
# #     "aggregate_batches",
# #     "compute_turnover",
# #     "compute_position_changes",
# #     "create_sizing_rules_with_weighting",
# #     # Backtest
# #     "PortfolioBacktestConfig",
# #     "PortfolioBacktestResult",
# #     "RebalanceMode",
# #     "RebalanceRule",
# #     "Trade",
# #     "simulate_rebalance",
# #     "compute_pnl_from_prices",
# #     "run_portfolio_backtest",
# #     "run_portfolio_backtest_from_targets",
# #     "compute_costs_summary",
# #     # Weighting
# #     "WeightingConfig",
# #     "WeightingMethod",
# #     "WeightingResult",
# #     "compute_equal_weights",
# #     "compute_inverse_vol_weights",
# #     "compute_strategy_weights",
# #     "apply_weight_caps",
# #     "create_weighting_function",
# #     "compute_rolling_weights",
# #     "validate_weights",
# #     # Adapters
# #     "fly_name_to_strategy_id",
# #     "create_fly_legs",
# #     "signal_df_to_strategy_signals",
# #     "pnl_df_to_strategy_signals",
# #     "intraday_result_to_signals",
# #     "batch_signals_by_timestamp",
# #     "filter_signals_by_strategy",
# #     "filter_signals_by_direction",
# #     "get_active_signals",
# # ]
