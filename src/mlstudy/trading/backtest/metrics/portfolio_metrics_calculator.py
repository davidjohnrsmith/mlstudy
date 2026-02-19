"""Portfolio-specific metrics calculator.

Extends :class:`MetricsCalculator` with trade-level metrics computed from
the portfolio's per-fill ``trade_df``.  Key additions:

* FIFO-based round-trip matching per instrument to estimate average
  holding time, hit rate, and profit factor from individual fills.
* Execution-cost breakdown (instrument vs. hedge).
* Per-instrument trade counts and DV01 statistics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics
from mlstudy.trading.backtest.metrics.metrics_calculator import MetricsCalculator
from mlstudy.trading.backtest.metrics.trades_utils import compute_turnover


@dataclass
class FIFORoundTrip:
    """A single FIFO-matched round-trip."""

    instrument: int
    qty: float
    entry_bar: int
    exit_bar: int
    entry_vwap: float
    exit_vwap: float
    side: int  # +1 long, -1 short

    @property
    def holding_bars(self) -> int:
        return self.exit_bar - self.entry_bar

    @property
    def pnl(self) -> float:
        return self.side * (self.exit_vwap - self.entry_vwap) * self.qty


def fifo_match(trade_df: pd.DataFrame) -> list[FIFORoundTrip]:
    """Match fills into round-trips using FIFO per instrument.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Portfolio per-fill trade DataFrame with columns:
        ``instrument``, ``side``, ``qty_fill``, ``bar``, ``vwap``.

    Returns
    -------
    list[FIFORoundTrip]
        Matched round-trips.  Open fills at the end are discarded.
    """
    if trade_df.empty:
        return []

    round_trips: list[FIFORoundTrip] = []

    for inst in trade_df["instrument"].unique():
        inst_df = trade_df[trade_df["instrument"] == inst].sort_values("bar")

        # FIFO queue: each entry is (remaining_qty, bar, vwap, side)
        queue: deque[list] = deque()

        for _, row in inst_df.iterrows():
            side = int(row["side"])
            qty = float(row["qty_fill"])
            bar = int(row["bar"])
            vwap = float(row["vwap"])

            if qty <= 1e-15:
                continue

            # If queue is empty or same direction, push
            if not queue or queue[0][3] == side:
                queue.append([qty, bar, vwap, side])
                continue

            # Opposite direction → match FIFO
            remaining = qty
            while remaining > 1e-15 and queue:
                front = queue[0]
                match_qty = min(remaining, front[0])

                round_trips.append(
                    FIFORoundTrip(
                        instrument=int(inst),
                        qty=match_qty,
                        entry_bar=front[1],
                        exit_bar=bar,
                        entry_vwap=front[2],
                        exit_vwap=vwap,
                        side=front[3],
                    )
                )

                front[0] -= match_qty
                remaining -= match_qty
                if front[0] <= 1e-15:
                    queue.popleft()

            # Leftover goes into queue as new position
            if remaining > 1e-15:
                queue.append([remaining, bar, vwap, side])

    return round_trips


class PortfolioMetricsCalculator(MetricsCalculator):
    """Metrics calculator for portfolio backtests.

    Overrides :meth:`_compute_trade_fields` to work with the per-fill
    ``trade_df`` format produced by the portfolio engine.  Uses FIFO
    matching to derive round-trip statistics.

    Parameters
    ----------
    bar_df : pd.DataFrame
        Per-bar DataFrame (must contain ``equity``, ``state`` columns).
    trade_df : pd.DataFrame or None
        Per-fill trade DataFrame from ``PortfolioBacktestResults.trade_df``.
        Expected columns: ``bar``, ``instrument``, ``side``, ``qty_fill``,
        ``vwap``, ``mid``, ``cost``, ``hedge_cost``.
    annualization_factor : int or None
        Bars per year for annualizing ratios.
    """

    def _compute_trade_fields(self) -> dict[str, float | int]:
        """Compute trade-level metrics from per-fill trade_df via FIFO matching."""
        trade_df = self._trade_df
        state = self._state
        bar_df = self._bar_df

        if trade_df.empty:
            return {
                "turnover_annual": 0.0,
                "avg_holding_period": 0.0,
                "hit_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0,
                "n_trades": 0,
                "pct_time_in_market": float((state != 0).mean()) if len(state) > 0 else 0.0,
            }

        # --- FIFO round-trip matching ---
        round_trips = fifo_match(trade_df)

        if round_trips:
            rt_pnls = np.array([rt.pnl for rt in round_trips])
            rt_holds = np.array([rt.holding_bars for rt in round_trips])
            rt_qtys = np.array([rt.qty for rt in round_trips])

            # Quantity-weighted average holding period
            total_qty = rt_qtys.sum()
            avg_hold = float(np.sum(rt_holds * rt_qtys) / total_qty) if total_qty > 1e-15 else 0.0

            # Hit rate: fraction of round-trips that are profitable
            hit_rate = float((rt_pnls > 0).mean())

            # Profit factor
            gross_profit = float(rt_pnls[rt_pnls > 0].sum())
            gross_loss = float(np.abs(rt_pnls[rt_pnls < 0]).sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else (
                np.inf if gross_profit > 0 else 0.0
            )

            # Win / loss stats
            winners = rt_pnls[rt_pnls > 0]
            losers = rt_pnls[rt_pnls < 0]
            avg_win = float(winners.mean()) if len(winners) > 0 else 0.0
            avg_loss = float(losers.mean()) if len(losers) > 0 else 0.0
            win_loss = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-10 else (
                np.inf if avg_win > 0 else 0.0
            )
        else:
            avg_hold = 0.0
            hit_rate = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            win_loss = 0.0

        # n_trades = number of fills
        n_trades = len(trade_df)

        # pct_time_in_market from bar_df state
        pct_in_market = float((state != 0).mean()) if len(state) > 0 else 0.0

        # Turnover from position columns
        pos_cols = [c for c in bar_df.columns if c.startswith("position")]
        if pos_cols:
            positions = bar_df[pos_cols].values
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
