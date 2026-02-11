"""Result container for the mean-reversion backtester."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MRBacktestResults:
    """Immutable container returned by :func:`run_backtest`.

    Per-bar arrays
    --------------
    positions : (T, N)  leg positions at end of each bar.
    cash      : (T,)    cash balance.
    equity    : (T,)    total equity (cash + MTM).
    pnl       : (T,)    step PnL (equity[t] - equity[t-1]).
    codes     : (T,)    int32 attempt/outcome code per bar.
    state     : (T,)    int32 state at end of bar (0 flat, +1 long, -1 short).
    holding   : (T,)    int32 bars held in current position (0 when flat).

    Per-trade arrays (length *n_trades*)
    ------------------------------------
    tr_bar       : bar index of the trade.
    tr_type      : 0=entry, 1=exit_tp, 2=exit_sl, 3=exit_time.
    tr_side      : +1 long, -1 short.
    tr_sizes     : (n_trades, N)  signed leg sizes.
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
    codes: np.ndarray
    state: np.ndarray
    holding: np.ndarray

    # per-trade
    tr_bar: np.ndarray
    tr_type: np.ndarray
    tr_side: np.ndarray
    tr_sizes: np.ndarray
    tr_vwaps: np.ndarray
    tr_mids: np.ndarray
    tr_cost: np.ndarray
    tr_code: np.ndarray
    tr_pkg_yield: np.ndarray
    n_trades: int

    @staticmethod
    def from_loop_output(out: tuple) -> "MRBacktestResults":
        """Construct from the raw tuple returned by :func:`mr_loop` / ``mr_loop_jit``."""
        n = int(out[16])
        return MRBacktestResults(
            positions=out[0],
            cash=out[1],
            equity=out[2],
            pnl=out[3],
            codes=out[4],
            state=out[5],
            holding=out[6],
            tr_bar=out[7][:n],
            tr_type=out[8][:n],
            tr_side=out[9][:n],
            tr_sizes=out[10][:n],
            tr_vwaps=out[11][:n],
            tr_mids=out[12][:n],
            tr_cost=out[13][:n],
            tr_code=out[14][:n],
            tr_pkg_yield=out[15][:n],
            n_trades=n,
        )
