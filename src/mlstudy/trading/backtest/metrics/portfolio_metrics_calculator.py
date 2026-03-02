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
    entry_datetime: object = None  # np.datetime64 or None
    exit_datetime: object = None   # np.datetime64 or None
    entry_dv01: float = 0.0       # instrument DV01 at entry

    @property
    def holding_bars(self) -> int:
        return self.exit_bar - self.entry_bar

    @property
    def holding_days(self) -> float | None:
        """Calendar days between entry and exit, or None if datetimes unavailable."""
        if self.entry_datetime is None or self.exit_datetime is None:
            return None
        delta = np.datetime64(self.exit_datetime) - np.datetime64(self.entry_datetime)
        return float(delta / np.timedelta64(1, "D"))

    @property
    def unhedged_pnl(self) -> float:
        return self.side * (self.exit_vwap - self.entry_vwap) * self.qty

    theoretical_hedged_pnl: float = 0.0
    net_hedged_pnl: float = 0.0


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

    has_dt = "datetime" in trade_df.columns
    has_dv01 = "dv01_fill" in trade_df.columns
    round_trips: list[FIFORoundTrip] = []

    for inst in trade_df["instrument"].unique():
        inst_df = trade_df[trade_df["instrument"] == inst].sort_values("bar")

        # FIFO queue: each entry is [remaining_qty, bar, vwap, side, datetime, dv01_fill]
        queue: deque[list] = deque()

        for _, row in inst_df.iterrows():
            side = int(row["side"])
            qty = float(row["qty_fill"])
            bar = int(row["bar"])
            vwap = float(row["vwap"])
            dt = row["datetime"] if has_dt else None
            dv01_fill = float(row["dv01_fill"]) if has_dv01 else 0.0

            if qty <= 1e-15:
                continue

            # If queue is empty or same direction, push
            if not queue or queue[0][3] == side:
                queue.append([qty, bar, vwap, side, dt, dv01_fill])
                continue

            # Opposite direction → match FIFO
            remaining = qty
            while remaining > 1e-15 and queue:
                front = queue[0]
                match_qty = min(remaining, front[0])

                round_trips.append(
                    FIFORoundTrip(
                        instrument=inst,
                        qty=match_qty,
                        entry_bar=front[1],
                        exit_bar=bar,
                        entry_vwap=front[2],
                        exit_vwap=vwap,
                        side=front[3],
                        entry_datetime=front[4],
                        exit_datetime=dt,
                        entry_dv01=front[5],
                    )
                )

                front[0] -= match_qty
                remaining -= match_qty
                if front[0] <= 1e-15:
                    queue.popleft()

            # Leftover goes into queue as new position
            if remaining > 1e-15:
                queue.append([remaining, bar, vwap, side, dt, dv01_fill])

    return round_trips


def _compute_hedge_pnls(
    round_trips: list[FIFORoundTrip],
    bar_df: pd.DataFrame,
    hedge_ratios: np.ndarray | None,
    dv01: np.ndarray | None,
    hedge_dv01: np.ndarray | None,
    hedge_mid_px: np.ndarray | None,
    instrument_id_to_idx: dict[str, int] | None,
    hedge_bid_px: np.ndarray | None = None,
    hedge_ask_px: np.ndarray | None = None,
) -> None:
    """Compute theoretical and net hedged PnL for each round-trip.

    Sets ``theoretical_hedged_pnl`` and ``net_hedged_pnl`` on each round-trip
    in place.

    Parameters
    ----------
    round_trips : list[FIFORoundTrip]
        FIFO-matched round-trips.
    bar_df : pd.DataFrame
        Per-bar DataFrame with ``pnl`` and ``mid_px_*`` / ``position_*``.
    hedge_ratios : (T, B, H) or None
    dv01 : (T, B) or None
    hedge_dv01 : (T, H) or None
    hedge_mid_px : (T, H) or None
    instrument_id_to_idx : dict mapping instrument string → b index, or None
    hedge_bid_px : (T, H) or None — top-of-book bid for hedge instruments
    hedge_ask_px : (T, H) or None — top-of-book ask for hedge instruments
    """
    if not round_trips:
        return

    have_hedge = (
        hedge_ratios is not None
        and dv01 is not None
        and hedge_dv01 is not None
        and hedge_mid_px is not None
        and instrument_id_to_idx is not None
    )

    if not have_hedge:
        for rt in round_trips:
            rt.theoretical_hedged_pnl = rt.unhedged_pnl
            rt.net_hedged_pnl = rt.unhedged_pnl
        return

    H = hedge_mid_px.shape[1]
    # Use bid/ask when available, fall back to mid
    have_ba = hedge_bid_px is not None and hedge_ask_px is not None

    # --- (a) Theoretical hedged PnL ---
    for rt in round_trips:
        b_idx = instrument_id_to_idx.get(rt.instrument)
        if b_idx is None:
            rt.theoretical_hedged_pnl = rt.unhedged_pnl
            continue

        theo_hedge_pnl = 0.0
        for h in range(H):
            hr = hedge_ratios[rt.entry_bar, b_idx, h]
            hdv01 = hedge_dv01[rt.entry_bar, h]
            if abs(hdv01) < 1e-15:
                continue
            theo_hedge_qty = rt.side * rt.qty * dv01[rt.entry_bar, b_idx] * hr / hdv01

            if have_ba:
                # Buying hedge → pay ask; selling → receive bid
                # At exit, unwind: long sells at bid, short buys at ask
                if theo_hedge_qty > 0:
                    entry_px = hedge_ask_px[rt.entry_bar, h]
                    exit_px = hedge_bid_px[rt.exit_bar, h]
                else:
                    entry_px = hedge_bid_px[rt.entry_bar, h]
                    exit_px = hedge_ask_px[rt.exit_bar, h]
            else:
                entry_px = hedge_mid_px[rt.entry_bar, h]
                exit_px = hedge_mid_px[rt.exit_bar, h]

            theo_hedge_pnl += theo_hedge_qty * (exit_px - entry_px)
        rt.theoretical_hedged_pnl = rt.unhedged_pnl + theo_hedge_pnl

    # --- (b) Net hedged PnL: bar-level allocation by DV01 weight ---
    # Use hedge_pnl column tracked directly by the backtest loop
    if "hedge_pnl" not in bar_df.columns:
        for rt in round_trips:
            rt.net_hedged_pnl = rt.unhedged_pnl
        return

    hedge_pnl_per_bar = bar_df["hedge_pnl"].values  # (T,)
    T = len(hedge_pnl_per_bar)

    # For each bar, allocate hedge PnL pro-rata by required hedge DV01
    allocated = np.zeros(len(round_trips), dtype=np.float64)

    # Build bar → list of (rt_index, hedge_dv01_demand) mapping
    # hedge DV01 demand = sum_h |qty * dv01[t,b] * hedge_ratios[t,b,h]|
    bar_to_rts: dict[int, list[tuple[int, float]]] = {}
    for i, rt in enumerate(round_trips):
        b_idx = instrument_id_to_idx.get(rt.instrument)
        if b_idx is None:
            continue
        for t in range(rt.entry_bar, min(rt.exit_bar, T)):
            w = 0.0
            for h in range(H):
                w += abs(rt.qty * dv01[t, b_idx] * hedge_ratios[t, b_idx, h])
            bar_to_rts.setdefault(t, []).append((i, w))

    for t, rt_list in bar_to_rts.items():
        total_hedge_dv01 = sum(w for _, w in rt_list)
        if total_hedge_dv01 < 1e-15:
            continue
        bar_hedge = hedge_pnl_per_bar[t]
        for idx, w in rt_list:
            allocated[idx] += bar_hedge * (w / total_hedge_dv01)

    for i, rt in enumerate(round_trips):
        rt.net_hedged_pnl = rt.unhedged_pnl + allocated[i]


def _pnl_trade_stats(pnls: np.ndarray) -> dict[str, float]:
    """Compute hit_rate, profit_factor, avg_win, avg_loss, win_loss_ratio from PnL array."""
    if len(pnls) == 0:
        return {"hit_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "win_loss_ratio": 0.0}

    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    hit_rate = float((pnls > 0).mean())

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(losers).sum()) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else (
        np.inf if gross_profit > 0 else 0.0
    )

    avg_win = float(winners.mean()) if len(winners) > 0 else 0.0
    avg_loss = float(losers.mean()) if len(losers) > 0 else 0.0
    win_loss = avg_win / abs(avg_loss) if abs(avg_loss) > 1e-10 else (
        np.inf if avg_win > 0 else 0.0
    )

    return {
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss,
    }


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
    hedge_ratios : (T, B, H) or None
    dv01 : (T, B) or None
    hedge_dv01 : (T, H) or None
    hedge_mid_px : (T, H) or None
    hedge_bid_px : (T, H) or None — top-of-book bid for hedge instruments
    hedge_ask_px : (T, H) or None — top-of-book ask for hedge instruments
    instrument_ids : list of str or None
    """

    def __init__(
        self,
        bar_df: pd.DataFrame,
        trade_df: pd.DataFrame | None = None,
        *,
        annualization_factor: int | None = None,
        hedge_ratios: np.ndarray | None = None,
        dv01: np.ndarray | None = None,
        hedge_dv01: np.ndarray | None = None,
        hedge_mid_px: np.ndarray | None = None,
        hedge_bid_px: np.ndarray | None = None,
        hedge_ask_px: np.ndarray | None = None,
        instrument_ids: list | None = None,
    ):
        super().__init__(bar_df, trade_df, annualization_factor=annualization_factor)
        self._hedge_ratios = hedge_ratios
        self._dv01 = dv01
        self._hedge_dv01 = hedge_dv01
        self._hedge_mid_px = hedge_mid_px
        self._hedge_bid_px = hedge_bid_px
        self._hedge_ask_px = hedge_ask_px
        self._instrument_id_to_idx = (
            {id_: i for i, id_ in enumerate(instrument_ids)}
            if instrument_ids is not None
            else None
        )

    def _compute_trade_fields(self) -> dict[str, float | int]:
        """Compute trade-level metrics from per-fill trade_df via FIFO matching."""
        trade_df = self._trade_df
        state = self._state
        bar_df = self._bar_df

        _zero_stats = _pnl_trade_stats(np.array([]))

        _zero_cost_eff = {
            "inst_traded_notional": 0.0,
            "inst_traded_dv01": 0.0,
            "gross_pnl_per_inst_traded_dv01_bps": 0.0,
            "net_pnl_per_inst_traded_dv01_bps": 0.0,
            "inst_cost_per_inst_traded_dv01_bps": 0.0,
            "hedge_cost_per_inst_traded_dv01_bps": 0.0,
            "total_cost_per_inst_traded_dv01_bps": 0.0,
        }

        if trade_df.empty:
            result = {
                "turnover_annual": 0.0,
                "avg_holding_period": 0.0,
                "n_trades": 0,
                "pct_time_in_market": float((state != 0).mean()) if len(state) > 0 else 0.0,
            }
            result.update(_zero_stats)
            result.update({f"{k}_unhedged": v for k, v in _zero_stats.items()})
            result.update({f"{k}_theo_hedged": v for k, v in _zero_stats.items()})
            result.update(_zero_cost_eff)
            return result

        # --- FIFO round-trip matching ---
        round_trips = fifo_match(trade_df)

        # Compute theoretical and net hedged PnL for round-trips
        _compute_hedge_pnls(
            round_trips, bar_df,
            self._hedge_ratios, self._dv01, self._hedge_dv01,
            self._hedge_mid_px, self._instrument_id_to_idx,
            self._hedge_bid_px, self._hedge_ask_px,
        )

        if round_trips:
            rt_qtys = np.array([rt.qty for rt in round_trips])

            # Use holding_days when datetimes are available, else holding_bars
            if round_trips[0].holding_days is not None:
                rt_holds = np.array([rt.holding_days for rt in round_trips])
            else:
                rt_holds = np.array([rt.holding_bars for rt in round_trips])

            # Quantity-weighted average holding period
            total_qty = rt_qtys.sum()
            avg_hold = float(np.sum(rt_holds * rt_qtys) / total_qty) if total_qty > 1e-15 else 0.0

            # Trade stats for all three PnL views
            pnl_unhedged = np.array([rt.unhedged_pnl for rt in round_trips])
            pnl_theo = np.array([rt.theoretical_hedged_pnl for rt in round_trips])
            pnl_net = np.array([rt.net_hedged_pnl for rt in round_trips])

            stats_net = _pnl_trade_stats(pnl_net)
            stats_unhedged = _pnl_trade_stats(pnl_unhedged)
            stats_theo = _pnl_trade_stats(pnl_theo)
        else:
            avg_hold = 0.0
            stats_net = _zero_stats
            stats_unhedged = _zero_stats
            stats_theo = _zero_stats

        # n_trades = number of fills
        n_trades = len(trade_df)

        # pct_time_in_market from bar_df state
        pct_in_market = float((state != 0).mean()) if len(state) > 0 else 0.0

        # Turnover from position columns
        pos_cols = [c for c in bar_df.columns if c.startswith("position")]
        if pos_cols:
            positions = bar_df[pos_cols].values
            pos_delta = np.abs(np.diff(positions, axis=0, prepend=0.0))
            inst_traded_notional = pd.Series(np.sum(pos_delta, axis=1), dtype=np.float64)
            gross_notional = pd.Series(
                np.sum(np.abs(positions), axis=1), dtype=np.float64
            )
            turnover = compute_turnover(inst_traded_notional, gross_notional, self.annualization_factor)
        else:
            turnover = 0.0

        # --- Cost efficiency metrics ---
        inst_traded_notional = float(trade_df["qty_fill"].abs().sum())
        inst_traded_dv01 = (
            float(trade_df["dv01_fill"].abs().sum())
            if "dv01_fill" in trade_df.columns
            else 0.0
        )

        if inst_traded_dv01 > 1e-15:
            total_gross_pnl = float(bar_df["gross_pnl"].sum()) if "gross_pnl" in bar_df.columns else 0.0
            total_net_pnl = float(bar_df["pnl"].sum()) if "pnl" in bar_df.columns else 0.0
            total_inst_cost = float(bar_df["instrument_cost"].sum()) if "instrument_cost" in bar_df.columns else 0.0
            total_hedge_cost = float(bar_df["hedge_cost"].sum()) if "hedge_cost" in bar_df.columns else 0.0
            total_cost = total_inst_cost + total_hedge_cost

            bps = 1e4 / inst_traded_dv01
            cost_eff = {
                "inst_traded_notional": inst_traded_notional,
                "inst_traded_dv01": inst_traded_dv01,
                "gross_pnl_per_inst_traded_dv01_bps": total_gross_pnl * bps,
                "net_pnl_per_inst_traded_dv01_bps": total_net_pnl * bps,
                "inst_cost_per_inst_traded_dv01_bps": total_inst_cost * bps,
                "hedge_cost_per_inst_traded_dv01_bps": total_hedge_cost * bps,
                "total_cost_per_inst_traded_dv01_bps": total_cost * bps,
            }
        else:
            cost_eff = {
                "inst_traded_notional": inst_traded_notional,
                "inst_traded_dv01": inst_traded_dv01,
                "gross_pnl_per_inst_traded_dv01_bps": 0.0,
                "net_pnl_per_inst_traded_dv01_bps": 0.0,
                "inst_cost_per_inst_traded_dv01_bps": 0.0,
                "hedge_cost_per_inst_traded_dv01_bps": 0.0,
                "total_cost_per_inst_traded_dv01_bps": 0.0,
            }

        result = {
            "turnover_annual": turnover,
            "avg_holding_period": avg_hold,
            "n_trades": n_trades,
            "pct_time_in_market": pct_in_market,
        }
        result.update(stats_net)
        result.update({f"{k}_unhedged": v for k, v in stats_unhedged.items()})
        result.update({f"{k}_theo_hedged": v for k, v in stats_theo.items()})
        result.update(cost_eff)
        return result
