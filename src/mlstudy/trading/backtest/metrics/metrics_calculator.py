from __future__ import annotations

import pandas as pd

from mlstudy.trading.backtest.metrics.trades_utils import (
    compute_turnover,
    compute_avg_holding_period,
    compute_n_trades,
)
from mlstudy.trading.backtest.metrics.equity_utils import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_tail_stats,
    compute_hit_rate,
    compute_profit_factor,
    compute_win_loss_stats,
)


class MetricsCalculator:
    """Compute backtest metrics from a pnl DataFrame.

    Provides three interfaces: equity-only, trades-only, or both.
    """

    def __init__(
        self,
        pnl_df: pd.DataFrame,
        return_col: str = "net_return",
        position_col: str = "position",
        traded_notional_col: str = "traded_notional",
        gross_notional_col: str = "gross_notional",
        cumulative_col: str = "cumulative_pnl",
    ):
        self._returns = pnl_df[return_col].dropna()
        self._position = pnl_df[position_col]
        self._cumulative = pnl_df[cumulative_col].dropna()
        self._pnl_df = pnl_df
        self._traded_notional_col = traded_notional_col
        self._gross_notional_col = gross_notional_col

    def _compute_equity_fields(self) -> dict:
        returns = self._returns
        cumulative = self._cumulative

        total_pnl = returns.sum()
        mean_ret = returns.mean()
        std_ret = returns.std()

        sharpe = compute_sharpe_ratio(returns)
        sortino = compute_sortino_ratio(returns)

        max_dd, max_dd_duration = compute_max_drawdown(cumulative)
        annual_return = mean_ret * 252
        calmar = (annual_return / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

        skew, kurt, var_95, cvar_95 = compute_tail_stats(returns)

        return {
            "total_pnl": total_pnl,
            "mean_daily_return": mean_ret,
            "std_daily_return": std_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "calmar_ratio": calmar,
            "skewness": skew,
            "kurtosis": kurt,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def _compute_trade_fields(self) -> dict:
        returns = self._returns
        position = self._position
        pnl_df = self._pnl_df

        if (
            self._traded_notional_col in pnl_df.columns
            and self._gross_notional_col in pnl_df.columns
        ):
            turnover = compute_turnover(
                pnl_df[self._traded_notional_col],
                pnl_df[self._gross_notional_col],
            )
        else:
            turnover = 0.0

        avg_hold = compute_avg_holding_period(position)
        hit_rate = compute_hit_rate(returns, position)
        profit_factor = compute_profit_factor(returns)
        avg_win, avg_loss, win_loss = compute_win_loss_stats(returns)
        n_trades = compute_n_trades(position)
        pct_in_market = (position != 0).mean()

        return {
            "turnover_annual": turnover,
            "avg_holding_period": avg_hold,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss,
            "n_trades": n_trades,
            "pct_time_in_market": pct_in_market,
        }

    def compute_equity(self) -> BacktestMetrics:
        """Compute equity-curve metrics only; trade fields set to zero."""
        equity = self._compute_equity_fields()
        return BacktestMetrics(
            **equity,
            turnover_annual=0.0,
            avg_holding_period=0.0,
            hit_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            n_trades=0,
            pct_time_in_market=0.0,
        )

    def compute_trades(self) -> BacktestMetrics:
        """Compute trade metrics only; equity fields set to zero."""
        trade = self._compute_trade_fields()
        return BacktestMetrics(
            total_pnl=0.0,
            mean_daily_return=0.0,
            std_daily_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
            **trade,
        )

    def compute_all(self) -> BacktestMetrics:
        """Compute all metrics (equivalent to ``compute_metrics()``)."""
        fields = {**self._compute_equity_fields(), **self._compute_trade_fields()}
        return BacktestMetrics(**fields)
