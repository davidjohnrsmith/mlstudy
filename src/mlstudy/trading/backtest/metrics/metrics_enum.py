from __future__ import annotations

from enum import Enum

from mlstudy.trading.backtest.metrics.metrics import MetricCategory


class Metric(Enum):
    TOTAL_PNL = ("total_pnl", MetricCategory.EQUITY, +1)
    MEAN_DAILY_RETURN = ("mean_daily_return", MetricCategory.EQUITY, +1)
    STD_DAILY_RETURN = ("std_daily_return", MetricCategory.EQUITY, -1)
    SHARPE_RATIO = ("sharpe_ratio", MetricCategory.EQUITY, +1)
    SORTINO_RATIO = ("sortino_ratio", MetricCategory.EQUITY, +1)
    MAX_DRAWDOWN = ("max_drawdown", MetricCategory.EQUITY, +1)
    MAX_DRAWDOWN_DURATION = ("max_drawdown_duration", MetricCategory.EQUITY, -1)
    CALMAR_RATIO = ("calmar_ratio", MetricCategory.EQUITY, +1)
    SKEWNESS = ("skewness", MetricCategory.EQUITY, +1)
    KURTOSIS = ("kurtosis", MetricCategory.EQUITY, -1)
    VAR_95 = ("var_95", MetricCategory.EQUITY, +1)
    CVAR_95 = ("cvar_95", MetricCategory.EQUITY, +1)
    TURNOVER_ANNUAL = ("turnover_annual", MetricCategory.TRADE, -1)
    AVG_HOLDING_PERIOD = ("avg_holding_period", MetricCategory.TRADE, +1)
    HIT_RATE = ("hit_rate", MetricCategory.TRADE, +1)
    PROFIT_FACTOR = ("profit_factor", MetricCategory.TRADE, +1)
    AVG_WIN = ("avg_win", MetricCategory.TRADE, +1)
    AVG_LOSS = ("avg_loss", MetricCategory.TRADE, +1)
    WIN_LOSS_RATIO = ("win_loss_ratio", MetricCategory.TRADE, +1)
    N_TRADES = ("n_trades", MetricCategory.TRADE, +1)
    PCT_TIME_IN_MARKET = ("pct_time_in_market", MetricCategory.TRADE, +1)

    def __init__(self, metric_name: str, category: MetricCategory, direction: int):
        self.key = metric_name
        self.category = category
        self.direction = direction

    @classmethod
    def from_key(cls, name: str) -> Metric:
        for m in cls:
            if m.key == name:
                return m
        raise ValueError(f"Unknown metric {name!r}")
