"""Strategy evaluation and backtesting.

Provides:
- Fly backtest engine with yield-based signals and price-based PnL
- Intraday backtest engine with session-aware execution
- Performance metrics (Sharpe, drawdown, etc.)
- Position sizing utilities (fixed notional, DV01 target)
- Results visualization and reporting
"""
# ruff: noqa: I001
# Import order matters for circular dependency resolution

from __future__ import annotations

# engine must be imported first (other modules depend on BacktestConfig)
from mlstudy.trading.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    SizingMode,
    backtest_fly,
    backtest_fly_from_panel,
)

# metrics must be imported before intraday (fly_universe imports compute_metrics)
from mlstudy.trading.backtest.metrics import (
    BacktestMetrics,
    compute_avg_holding_period,
    compute_hit_rate,
    compute_max_drawdown,
    compute_metrics,
    compute_n_trades,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_tail_stats,
    compute_turnover,
    compute_win_loss_stats,
)
from mlstudy.trading.backtest.intraday import (
    IntradayBacktestConfig,
    IntradayBacktestResult,
    aggregate_to_daily,
    backtest_fly_intraday,
)

__all__ = [
    # Engine
    "BacktestConfig",
    "BacktestResult",
    "SizingMode",
    "backtest_fly",
    "backtest_fly_from_panel",
    # Intraday
    "IntradayBacktestConfig",
    "IntradayBacktestResult",
    "backtest_fly_intraday",
    "aggregate_to_daily",
    # Metrics
    "BacktestMetrics",
    "compute_metrics",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_max_drawdown",
    "compute_turnover",
    "compute_avg_holding_period",
    "compute_hit_rate",
    "compute_profit_factor",
    "compute_win_loss_stats",
    "compute_tail_stats",
    "compute_n_trades",
    # Report
    "RegimeData",
    "generate_report",
    "print_metrics_summary",
    "save_backtest_plots",
    # Regime diagnostics
    "RegimeDiagnostics",
    "compute_regime_diagnostics",
    "plot_fly_with_regime",
    "plot_zscore_positions_stops",
    "save_regime_plots",
    "print_regime_summary",
]
