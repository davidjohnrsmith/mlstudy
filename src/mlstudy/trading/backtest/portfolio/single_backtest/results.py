"""Result container for the LP portfolio backtester."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PortfolioBacktestResults:
    """Container returned by :func:`run_backtest`.

    Per-bar arrays
    --------------
    positions     : (T, B)  instrument positions at end of each bar.
    cash          : (T,)    cash balance.
    equity        : (T,)    total equity (cash + instrument MTM + hedge MTM).
    pnl           : (T,)    net step PnL (after execution costs).
    gross_pnl     : (T,)    gross step PnL (before execution costs).
    codes         : (T,)    int32 bar-level outcome code.
    n_trades_bar  : (T,)    number of instrument trades executed per bar.
    cooldown      : (T,)    remaining cooldown bars.
    hedge_positions : (T, H)  hedge instrument positions at end of each bar.
    hedge_pnl     : (T,)    hedge PnL per bar (MTM change minus cash outflows).
    instrument_position_mtm : (T,) mark-to-market of instrument positions.
    hedge_position_mtm      : (T,) mark-to-market of hedge positions.
    instrument_cash_mtm     : (T,) cumulative cash from instrument trades at mid.
    hedge_cash_mtm          : (T,) cumulative cash from hedge trades at mid.
    portfolio_mtm           : (T,) sum of instrument/hedge position + cash MTMs.
    instrument_cost         : (T,) execution cost of instrument trades per bar.
    hedge_cost_bar          : (T,) execution cost of hedge trades per bar.
    portfolio_cost          : (T,) total execution cost (instrument + hedge) per bar.

    Per-trade arrays (length *n_trades*)
    ------------------------------------
    tr_bar        : bar index.
    tr_instrument : instrument index.
    tr_side       : +1 buy / -1 sell.
    tr_qty_req    : requested notional quantity.
    tr_qty_fill   : filled notional quantity.
    tr_dv01_req   : requested DV01.
    tr_dv01_fill  : filled DV01.
    tr_alpha      : executable alpha (bps) at trade time.
    tr_fair_type  : 0=dec, 1=inc.
    tr_vwap       : fill VWAP.
    tr_mid        : mid price at fill time.
    tr_cost       : execution cost.
    tr_code       : per-trade fill code.
    tr_hedge_sizes : (n_trades, H) requested hedge qty per instrument.
    tr_hedge_vwaps : (n_trades, H) hedge fill VWAPs.
    tr_hedge_fills : (n_trades, H) actual hedge fills (signed).
    tr_hedge_cost  : (n_trades,)   total hedge execution cost per trade.
    """

    # per-bar
    positions: np.ndarray
    cash: np.ndarray
    equity: np.ndarray
    pnl: np.ndarray
    gross_pnl: np.ndarray
    codes: np.ndarray
    n_trades_bar: np.ndarray
    cooldown: np.ndarray
    hedge_positions: np.ndarray
    hedge_pnl: np.ndarray
    instrument_position_mtm: np.ndarray
    hedge_position_mtm: np.ndarray
    instrument_cash_mtm: np.ndarray
    hedge_cash_mtm: np.ndarray
    portfolio_mtm: np.ndarray
    instrument_cost: np.ndarray
    hedge_cost_bar: np.ndarray
    portfolio_cost: np.ndarray

    # per-trade
    tr_bar: np.ndarray
    tr_instrument: np.ndarray
    tr_side: np.ndarray
    tr_qty_req: np.ndarray
    tr_qty_fill: np.ndarray
    tr_dv01_req: np.ndarray
    tr_dv01_fill: np.ndarray
    tr_alpha: np.ndarray
    tr_fair_type: np.ndarray
    tr_vwap: np.ndarray
    tr_mid: np.ndarray
    tr_cost: np.ndarray
    tr_code: np.ndarray
    tr_hedge_sizes: np.ndarray
    tr_hedge_vwaps: np.ndarray
    tr_hedge_fills: np.ndarray
    tr_hedge_cost: np.ndarray
    n_trades: int
    instrument_ids: list

    # optional context
    datetimes: np.ndarray | None = field(default=None, repr=False)
    close_time: str | None = field(default=None, repr=False)
    mid_px: np.ndarray | None = field(default=None, repr=False)
    hedge_mid_px: np.ndarray | None = field(default=None, repr=False)
    hedge_bid_px: np.ndarray | None = field(default=None, repr=False)  # (T, H) top-of-book
    hedge_ask_px: np.ndarray | None = field(default=None, repr=False)  # (T, H) top-of-book
    hedge_ratios: np.ndarray | None = field(default=None, repr=False)  # (T, B, H)
    dv01: np.ndarray | None = field(default=None, repr=False)          # (T, B)
    hedge_dv01: np.ndarray | None = field(default=None, repr=False)    # (T, H)

    bar_df: pd.DataFrame = None
    trade_df: pd.DataFrame = None
    close_bar_df: pd.DataFrame = None

    def __post_init__(self):
        self.bar_df = self.to_bar_df()
        self.trade_df = self.to_trade_df()
        self.close_bar_df = self._build_close_bar_df()

    def _build_close_bar_df(self) -> pd.DataFrame | None:
        if self.close_time is None or self.datetimes is None:
            return None
        df = self.bar_df
        dt = pd.to_datetime(df["datetime"])
        target = pd.Timestamp(self.close_time).time()
        mask = dt.dt.time == target
        return df[mask].reset_index(drop=True)

    @staticmethod
    def from_loop_output(
        out: tuple,
        *,
        instrument_ids: list,
        datetimes: np.ndarray | None = None,
        close_time: str | None = None,
        mid_px: np.ndarray | None = None,
        hedge_mid_px: np.ndarray | None = None,
        hedge_bid_px: np.ndarray | None = None,
        hedge_ask_px: np.ndarray | None = None,
        hedge_ratios: np.ndarray | None = None,
        dv01: np.ndarray | None = None,
        hedge_dv01: np.ndarray | None = None,
    ) -> "PortfolioBacktestResults":
        """Construct from the raw tuple returned by :func:`lp_portfolio_loop`."""
        n = int(out[35])
        return PortfolioBacktestResults(
            positions=out[0],
            cash=out[1],
            equity=out[2],
            pnl=out[3],
            gross_pnl=out[4],
            codes=out[5],
            n_trades_bar=out[6],
            cooldown=out[7],
            hedge_positions=out[8],
            hedge_pnl=out[26],
            instrument_position_mtm=out[27],
            hedge_position_mtm=out[28],
            instrument_cash_mtm=out[29],
            hedge_cash_mtm=out[30],
            portfolio_mtm=out[31],
            instrument_cost=out[32],
            hedge_cost_bar=out[33],
            portfolio_cost=out[34],
            tr_bar=out[9][:n],
            tr_instrument=out[10][:n],
            tr_side=out[11][:n],
            tr_qty_req=out[12][:n],
            tr_qty_fill=out[13][:n],
            tr_dv01_req=out[14][:n],
            tr_dv01_fill=out[15][:n],
            tr_alpha=out[16][:n],
            tr_fair_type=out[17][:n],
            tr_vwap=out[18][:n],
            tr_mid=out[19][:n],
            tr_cost=out[20][:n],
            tr_code=out[21][:n],
            tr_hedge_sizes=out[22][:n],
            tr_hedge_vwaps=out[23][:n],
            tr_hedge_fills=out[24][:n],
            tr_hedge_cost=out[25][:n],
            n_trades=n,
            datetimes=datetimes,
            close_time=close_time,
            mid_px=mid_px,
            hedge_mid_px=hedge_mid_px,
            hedge_bid_px=hedge_bid_px,
            hedge_ask_px=hedge_ask_px,
            hedge_ratios=hedge_ratios,
            dv01=dv01,
            hedge_dv01=hedge_dv01,
            instrument_ids=instrument_ids,
        )

    # -------------------------------------------------------------------
    # DataFrame outputs
    # -------------------------------------------------------------------

    def to_bar_df(self) -> pd.DataFrame:
        """Per-bar DataFrame with equity/pnl + per-instrument positions + hedge positions."""
        T = len(self.equity)
        data: dict[str, np.ndarray] = {}

        if self.datetimes is not None:
            data["datetime"] = self.datetimes[:T]

        data.update(
            {
                "equity": self.equity[:T],
                "cash": self.cash[:T],
                "pnl": self.pnl[:T],
                "gross_pnl": self.gross_pnl[:T],
                "cumulative_pnl": np.cumsum(self.pnl[:T]),
                "cumulative_gross_pnl": np.cumsum(self.gross_pnl[:T]),
                "code": self.codes[:T],
                "state": self.codes[:T],
                "n_trades": self.n_trades_bar[:T],
                "cooldown": self.cooldown[:T],
                "instrument_position_mtm": self.instrument_position_mtm[:T],
                "hedge_position_mtm": self.hedge_position_mtm[:T],
                "instrument_cash_mtm": self.instrument_cash_mtm[:T],
                "hedge_cash_mtm": self.hedge_cash_mtm[:T],
                "portfolio_mtm": self.portfolio_mtm[:T],
                "instrument_cost": self.instrument_cost[:T],
                "hedge_cost": self.hedge_cost_bar[:T],
                "portfolio_cost": self.portfolio_cost[:T],
            }
        )

        # Instrument positions
        pos = self.positions
        if pos.ndim == 2:
            for b in range(pos.shape[1]):
                data[f"position_{b}"] = pos[:T, b]

        # Hedge positions and PnL
        hpos = self.hedge_positions
        if hpos.ndim == 2 and hpos.shape[1] > 0:
            for h in range(hpos.shape[1]):
                data[f"hedge_position_{h}"] = hpos[:T, h]
            data["hedge_pnl"] = self.hedge_pnl[:T]

        if self.mid_px is not None and self.mid_px.ndim == 2:
            for b in range(self.mid_px.shape[1]):
                data[f"mid_px_{b}"] = self.mid_px[:T, b]

        return pd.DataFrame(data)

    def to_trade_df(self) -> pd.DataFrame:
        """Per-trade DataFrame with fill details and hedge info."""
        has_dt = self.datetimes is not None
        H = self.tr_hedge_sizes.shape[1] if self.n_trades > 0 else 0

        rows: list[dict] = []
        for i in range(self.n_trades):
            row: dict = {}
            bar = int(self.tr_bar[i])
            if has_dt:
                row["datetime"] = self.datetimes[bar]

            inst_idx = int(self.tr_instrument[i])

            row.update(
                {
                    "bar": bar,
                    "instrument": self.instrument_ids[inst_idx],
                    "side": int(self.tr_side[i]),
                    "qty_req": float(self.tr_qty_req[i]),
                    "qty_fill": float(self.tr_qty_fill[i]),
                    "dv01_req": float(self.tr_dv01_req[i]),
                    "dv01_fill": float(self.tr_dv01_fill[i]),
                    "alpha_bps": float(self.tr_alpha[i]),
                    "fair_type": int(self.tr_fair_type[i]),
                    "vwap": float(self.tr_vwap[i]),
                    "mid": float(self.tr_mid[i]),
                    "cost": float(self.tr_cost[i]),
                    "code": int(self.tr_code[i]),
                    "hedge_cost": float(self.tr_hedge_cost[i]),
                }
            )

            for h in range(H):
                row[f"hedge_size_{h}"] = float(self.tr_hedge_sizes[i, h])
                row[f"hedge_fill_{h}"] = float(self.tr_hedge_fills[i, h])
                row[f"hedge_vwap_{h}"] = float(self.tr_hedge_vwaps[i, h])

            rows.append(row)

        return pd.DataFrame(rows)
