"""Result container for the mean-reversion backtester."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state import TradeType


# Maps tr_type int to human-readable exit type string
_EXIT_TYPE_NAMES: dict[int, str] = {
    TradeType.TRADE_EXIT_TP: "tp",
    TradeType.TRADE_EXIT_SL: "sl",
    TradeType.TRADE_EXIT_TIME: "time",
}


@dataclass
class MRBacktestResults:
    """Immutable container returned by :func:`run_backtest`.

    Per-bar arrays
    --------------
    positions : (T, N)  leg positions at end of each bar.
    cash      : (T,)    cash balance.
    equity    : (T,)    total equity (cash + MTM).
    pnl       : (T,)    net step PnL (equity[t] - equity[t-1], after costs).
    gross_pnl : (T,)    gross step PnL (before execution costs).
    codes     : (T,)    int32 attempt/outcome code per bar.
    state     : (T,)    int32 state at end of bar (0 flat, +1 long, -1 short, 2-4 cooldown).
    holding   : (T,)    int32 bars held in current position (0 when flat).

    Per-trade arrays (length *n_trades*)
    ------------------------------------
    tr_bar       : bar index of the trade.
    tr_type      : 0=entry, 1=exit_tp, 2=exit_sl, 3=exit_time.
    tr_side      : +1 long, -1 short.
    tr_sizes     : (n_trades, N)  signed leg sizes.
    tr_risks     : (n_trades, N)  signed DV01-weighted risk per leg.
    tr_vwaps     : (n_trades, N)  fill VWAPs per leg.
    tr_mids      : (n_trades, N)  mid prices at fill time per leg.
    tr_cost      : basket execution cost (always >= 0).
    tr_code      : int32 attempt/outcome code for the trade.
    tr_pkg_yield : package_yield_bps at time of trade.
    """

    # per-bar
    positions: np.ndarray
    cash: np.ndarray
    equity: np.ndarray
    pnl: np.ndarray
    gross_pnl: np.ndarray
    codes: np.ndarray
    state: np.ndarray
    holding: np.ndarray

    # per-trade
    tr_bar: np.ndarray
    tr_type: np.ndarray
    tr_side: np.ndarray
    tr_sizes: np.ndarray
    tr_risks: np.ndarray
    tr_vwaps: np.ndarray
    tr_mids: np.ndarray
    tr_cost: np.ndarray
    tr_code: np.ndarray
    tr_pkg_yield: np.ndarray
    n_trades: int
    datetimes: np.ndarray | None = field(default=None, repr=False)

    bar_df:pd.DataFrame() =None
    trade_df: pd.DataFrame() = None

    def __post_init__(self):
        self.bar_df = self.to_bar_df()
        self.trade_df = self.to_trade_df()

    @staticmethod
    def from_loop_output(
        out: tuple,
        *,
        datetimes: np.ndarray | None = None,
    ) -> "MRBacktestResults":
        """Construct from the raw tuple returned by :func:`mr_loop` / ``mr_loop_jit``."""
        n = int(out[17])
        return MRBacktestResults(
            positions=out[0],
            cash=out[1],
            equity=out[2],
            pnl=out[3],
            gross_pnl=out[18],
            codes=out[4],
            state=out[5],
            holding=out[6],
            tr_bar=out[7][:n],
            tr_type=out[8][:n],
            tr_side=out[9][:n],
            tr_sizes=out[10][:n],
            tr_risks=out[11][:n],
            tr_vwaps=out[12][:n],
            tr_mids=out[13][:n],
            tr_cost=out[14][:n],
            tr_code=out[15][:n],
            tr_pkg_yield=out[16][:n],
            n_trades=n,
            datetimes=datetimes,
        )

    # -------------------------------------------------------------------
    # DataFrame outputs
    # -------------------------------------------------------------------

    def to_bar_df(self) -> pd.DataFrame:
        """Per-bar DataFrame with equity/pnl + per-leg positions + state/codes.

        - pnl is net (after execution costs); gross_pnl is before costs.
        - Index is bar number. If datetimes is available, include a datetime column.
        - If positions is (T, N), we output position_0..position_{N-1}.
        """
        T = len(self.equity)
        data: dict[str, np.ndarray] = {}

        if self.datetimes is not None:
            data["datetime"] = self.datetimes[:T]

        data.update(
            {
                "equity": self.equity[:T],
                'cash': self.cash[:T],
                "pnl": self.pnl[:T],
                "gross_pnl": self.gross_pnl[:T],
                "cumulative_pnl": np.cumsum(self.pnl[:T]),
                "cumulative_gross_pnl": np.cumsum(self.gross_pnl[:T]),
                "state": self.state[:T],
                "code": self.codes[:T],
                "holding": self.holding[:T],
            }
        )

        pos = self.positions
        if pos is None:
            pass
        elif pos.ndim == 1:
            # backward-compatible single position series
            data["position"] = pos[:T]
        elif pos.ndim == 2:
            # per-leg positions
            N = pos.shape[1]
            for i in range(N):
                data[f"position_{i}"] = pos[:T, i]
        else:
            raise ValueError(f"positions must be 1D or 2D, got shape {pos.shape}")

        return pd.DataFrame(data)

    def to_trade_df(self) -> pd.DataFrame:
        """Round-trip trade DataFrame pairing each entry with its exit.

        Columns include entry/exit bars, side, holding_bars, exit_type,
        costs, yields, pnl, slippage, and per-leg sizes/vwaps/mids/risks.
        If *datetimes* is available, ``entry_datetime`` and ``exit_datetime``
        columns are included.
        """
        has_dt = self.datetimes is not None
        N = self.tr_sizes.shape[1] if self.n_trades > 0 else 0

        rows: list[dict] = []
        pending_entry: dict | None = None

        for i in range(self.n_trades):
            ttype = int(self.tr_type[i])
            if ttype == TradeType.TRADE_ENTRY:
                pending_entry = {
                    "idx": i,
                    "bar": int(self.tr_bar[i]),
                    "side": int(self.tr_side[i]),
                    "cost": float(self.tr_cost[i]),
                    "pkg_yield": float(self.tr_pkg_yield[i]),
                    "vwaps": self.tr_vwaps[i].copy(),
                    "mids": self.tr_mids[i].copy(),
                    "sizes": self.tr_sizes[i].copy(),
                    "risks": self.tr_risks[i].copy(),
                }
            elif pending_entry is not None:
                entry = pending_entry
                exit_bar = int(self.tr_bar[i])
                exit_cost = float(self.tr_cost[i])
                exit_pkg_yield = float(self.tr_pkg_yield[i])

                slip_entry = float(np.mean(np.abs(entry["vwaps"] - entry["mids"])))
                slip_exit = float(np.mean(np.abs(self.tr_vwaps[i] - self.tr_mids[i])))

                entry_bar = entry["bar"]
                pnl_slice = self.pnl[entry_bar : exit_bar + 1]
                trade_pnl = float(np.sum(pnl_slice))

                row: dict = {}
                if has_dt:
                    row["entry_datetime"] = self.datetimes[entry_bar]
                    row["exit_datetime"] = self.datetimes[exit_bar]
                    row["holding_days"] = (row["exit_datetime"]-row["entry_datetime"])/np.timedelta64(1, "D")

                row.update(
                    {
                        "entry_bar": entry_bar,
                        "exit_bar": exit_bar,
                        "side": entry["side"],
                        "holding_bars": exit_bar - entry_bar,
                        "exit_type": _EXIT_TYPE_NAMES.get(ttype, "unknown"),
                        "entry_cost": entry["cost"],
                        "exit_cost": exit_cost,
                        "total_cost": entry["cost"] + exit_cost,
                        "entry_pkg_yield": entry["pkg_yield"],
                        "exit_pkg_yield": exit_pkg_yield,
                        "yield_delta": exit_pkg_yield - entry["pkg_yield"],
                        "pnl": trade_pnl,
                        "slippage_entry": slip_entry,
                        "slippage_exit": slip_exit,
                    }
                )

                # Per-leg columns
                for leg in range(N):
                    row[f"entry_sizes_{leg}"] = float(entry["sizes"][leg])
                    row[f"exit_sizes_{leg}"] = float(self.tr_sizes[i][leg])
                    row[f"entry_vwaps_{leg}"] = float(entry["vwaps"][leg])
                    row[f"exit_vwaps_{leg}"] = float(self.tr_vwaps[i][leg])
                    row[f"entry_mids_{leg}"] = float(entry["mids"][leg])
                    row[f"exit_mids_{leg}"] = float(self.tr_mids[i][leg])
                    row[f"entry_risks_{leg}"] = float(entry["risks"][leg])
                    row[f"exit_risks_{leg}"] = float(self.tr_risks[i][leg])

                rows.append(row)
                pending_entry = None

        return pd.DataFrame(rows)
