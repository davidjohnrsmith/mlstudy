from __future__ import annotations

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics
from mlstudy.trading.backtest.metrics.trades_utils import (
    compute_turnover,
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
    """Compute backtest metrics from bar_df and trade_df.

    Parameters
    ----------
    bar_df : pd.DataFrame
        Per-bar DataFrame from ``MRBacktestResults.bar_df``.
        Required columns: ``pnl``, ``cumulative_pnl``, ``state``.
        Optional: ``equity``, position columns (``position`` or
        ``position_0..N-1``), ``holding``.
    trade_df : pd.DataFrame or None
        Round-trip trade DataFrame from ``MRBacktestResults.trade_df``.
        Expected columns: ``pnl``, ``holding_bars``, ``total_cost``.
        If *None*, trade-level metrics are zeroed out.
    annualization_factor : int
        Bars per year for annualizing ratios (default 252).
    """

    def __init__(
        self,
        bar_df: pd.DataFrame,
        trade_df: pd.DataFrame | None = None,
        *,
        annualization_factor: int = 252,
    ):
        self._bar_df = bar_df
        self._trade_df = trade_df if trade_df is not None else pd.DataFrame()
        self.annualization_factor = annualization_factor

        # Derive returns (bar-over-bar PnL) as a Series
        self._pnl = pd.Series(bar_df["pnl"].values, dtype=np.float64)
        self._cumulative_pnl = pd.Series(bar_df["cumulative_pnl"].values, dtype=np.float64)

        # Position: scalar state indicator for hit_rate / pct_in_market
        self._state = pd.Series(bar_df["state"].values, dtype=np.float64)

    def _compute_equity_fields(self) -> dict[str, float | int]:
        """Equity-curve metrics from bar_df."""
        returns = self._pnl
        cumulative = self._cumulative_pnl

        total_pnl = float(returns.sum())
        mean_ret = float(returns.mean())
        std_ret = float(returns.std())

        sharpe = float(compute_sharpe_ratio(returns, annualization_factor=self.annualization_factor))
        sortino = float(compute_sortino_ratio(returns, annualization_factor=self.annualization_factor))

        max_dd, max_dd_duration = compute_max_drawdown(cumulative)
        annual_return = mean_ret * self.annualization_factor
        calmar = float(abs(annual_return / max_dd)) if abs(max_dd) > 1e-10 else 0.0

        skew, kurt, var_95, cvar_95 = compute_tail_stats(returns)

        return {
            "total_pnl": total_pnl,
            "mean_daily_return": mean_ret,
            "std_daily_return": std_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": float(max_dd),
            "max_drawdown_duration": int(max_dd_duration),
            "calmar_ratio": calmar,
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
        }

    def _compute_trade_fields(self) -> dict[str, float | int]:
        """Trade-level metrics from trade_df and bar_df state."""
        trade_df = self._trade_df
        state = self._state
        returns = self._pnl

        # --- From trade_df (round-trips) ---
        if trade_df.empty:
            n_trades = 0
            avg_hold = 0.0
            hit_rate = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            win_loss = 0.0
        else:
            n_trades = len(trade_df)
            avg_hold = float(trade_df["holding_bars"].mean())

            trade_pnl = trade_df["pnl"]
            winners = trade_pnl[trade_pnl > 0]
            losers = trade_pnl[trade_pnl < 0]

            hit_rate = float((trade_pnl > 0).mean()) if n_trades > 0 else 0.0

            gross_profit = float(winners.sum())
            gross_loss = float(losers.abs().sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else (
                np.inf if gross_profit > 0 else 0.0
            )

            avg_win = float(winners.mean()) if len(winners) > 0 else 0.0
            avg_loss = float(losers.mean()) if len(losers) > 0 else 0.0
            win_loss = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-10 else (
                np.inf if avg_win > 0 else 0.0
            )

        # --- From bar_df ---
        pct_in_market = float((state != 0).mean()) if len(state) > 0 else 0.0

        # Turnover: derive from position columns in bar_df
        bar_df = self._bar_df
        pos_cols = [c for c in bar_df.columns if c.startswith("position")]
        if pos_cols:
            positions = bar_df[pos_cols].values  # (T, N) or (T, 1)
            pos_delta = np.abs(np.diff(positions, axis=0, prepend=0.0))
            traded_notional = pd.Series(np.sum(pos_delta, axis=1), dtype=np.float64)
            gross_notional = pd.Series(
                np.sum(np.abs(positions), axis=1), dtype=np.float64
            )
            turnover = compute_turnover(traded_notional, gross_notional, self.annualization_factor)
        else:
            turnover = 0.0

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
        """Compute all metrics (equity + trade)."""
        fields = {**self._compute_equity_fields(), **self._compute_trade_fields()}
        return BacktestMetrics(**fields)
