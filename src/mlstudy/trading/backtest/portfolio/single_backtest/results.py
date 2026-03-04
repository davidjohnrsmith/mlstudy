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
    net_instrument_dv01     : (T,) net instrument DV01 (sum of pos * dv01).
    gross_instrument_dv01   : (T,) gross instrument DV01 (sum of |pos * dv01|).
    net_hedge_dv01          : (T,) net hedge DV01 (sum of hedge_pos * hedge_dv01).
    gross_hedge_dv01        : (T,) gross hedge DV01 (sum of |hedge_pos * hedge_dv01|).

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
    net_instrument_dv01: np.ndarray
    gross_instrument_dv01: np.ndarray
    net_hedge_dv01: np.ndarray
    gross_hedge_dv01: np.ndarray

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
    hedge_ids: list | None = None

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
    maturity: np.ndarray | None = field(default=None, repr=False)      # (B,) years to maturity

    initial_capital: float = 0.0

    bar_df: pd.DataFrame = None
    trade_df: pd.DataFrame = None
    close_bar_df: pd.DataFrame = None
    instrument_pnl_df: pd.DataFrame = None

    def __post_init__(self):
        self.bar_df = self.to_bar_df()
        self.trade_df = self.to_trade_df()
        self.close_bar_df = self._build_close_bar_df()
        if self.mid_px is not None:
            self.instrument_pnl_df = self.to_instrument_pnl_df()

    # Columns that are per-bar flows (must be aggregated when downsampling)
    _FLOW_COLS = frozenset({
        "pnl", "gross_pnl", "instrument_cost", "hedge_cost",
        "portfolio_cost", "n_trades", "hedge_pnl",
    })

    def _build_close_bar_df(self) -> pd.DataFrame | None:
        if self.close_time is None or self.datetimes is None:
            return None
        df = self.bar_df
        dt = pd.to_datetime(df["datetime"])
        target = pd.Timestamp(self.close_time).time()
        mask = dt.dt.time == target
        close_idx = np.where(mask.values)[0]
        if len(close_idx) == 0:
            return None

        # Start with close-bar rows (snapshot columns are correct as-is)
        close_df = df.iloc[close_idx].copy()

        # Aggregate flow columns over the period between consecutive close bars
        for col in self._FLOW_COLS:
            if col not in df.columns:
                continue
            vals = df[col].values
            agg = np.empty(len(close_idx), dtype=np.float64)
            for i, ci in enumerate(close_idx):
                start = (close_idx[i - 1] + 1) if i > 0 else 0
                agg[i] = vals[start: ci + 1].sum()
            close_df[col] = agg

        return close_df.reset_index(drop=True)

    @staticmethod
    def from_loop_output(
        out: tuple,
        *,
        instrument_ids: list,
        hedge_ids: list | None = None,
        datetimes: np.ndarray | None = None,
        close_time: str | None = None,
        mid_px: np.ndarray | None = None,
        hedge_mid_px: np.ndarray | None = None,
        hedge_bid_px: np.ndarray | None = None,
        hedge_ask_px: np.ndarray | None = None,
        hedge_ratios: np.ndarray | None = None,
        dv01: np.ndarray | None = None,
        hedge_dv01: np.ndarray | None = None,
        maturity: np.ndarray | None = None,
        initial_capital: float = 0.0,
    ) -> "PortfolioBacktestResults":
        """Construct from the raw tuple returned by :func:`lp_portfolio_loop`."""
        n = int(out[39])
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
            net_instrument_dv01=out[35],
            gross_instrument_dv01=out[36],
            net_hedge_dv01=out[37],
            gross_hedge_dv01=out[38],
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
            maturity=maturity,
            instrument_ids=instrument_ids,
            hedge_ids=hedge_ids,
            initial_capital=initial_capital,
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
                "instrument_mtm": self.instrument_position_mtm[:T] + self.instrument_cash_mtm[:T],
                "hedge_mtm": self.hedge_position_mtm[:T] + self.hedge_cash_mtm[:T],
                "instrument_cash_mtm": self.instrument_cash_mtm[:T],
                "hedge_cash_mtm": self.hedge_cash_mtm[:T],
                "portfolio_mtm": self.portfolio_mtm[:T],
                "portfolio_mtm_with_cost": self.portfolio_mtm[:T] - np.cumsum(self.portfolio_cost[:T]),
                "instrument_cost": self.instrument_cost[:T],
                "hedge_cost": self.hedge_cost_bar[:T],
                "portfolio_cost": self.portfolio_cost[:T],
                "cumulative_instrument_cost": np.cumsum(self.instrument_cost[:T]),
                "cumulative_hedge_cost": np.cumsum(self.hedge_cost_bar[:T]),
                "cumulative_portfolio_cost": np.cumsum(self.portfolio_cost[:T]),
                "net_instrument_dv01": self.net_instrument_dv01[:T],
                "gross_instrument_dv01": self.gross_instrument_dv01[:T],
                "net_hedge_dv01": self.net_hedge_dv01[:T],
                "gross_hedge_dv01": self.gross_hedge_dv01[:T],
                "net_total_dv01": self.net_instrument_dv01[:T] + self.net_hedge_dv01[:T],
                "gross_total_dv01": self.gross_instrument_dv01[:T] + self.gross_hedge_dv01[:T],
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

    def to_instrument_pnl_df(self) -> pd.DataFrame:
        """Per-instrument PnL summary sorted by total_pnl (worst first).

        Columns
        -------
        instrument        : instrument id.
        final_position    : position at last bar.
        instrument_pnl    : direct PnL from instrument (position MTM + trade cash).
        instrument_cost   : execution cost on instrument trades.
        hedge_cost        : hedge execution cost attributed to this instrument.
        hedge_pnl         : hedge position PnL attributed to this instrument.
        total_pnl         : instrument_pnl + hedge_pnl.
        total_cost        : instrument_cost + hedge_cost.
        n_trades          : number of trades on this instrument.

        Requires *mid_px* and (for hedge attribution) *hedge_mid_px* to have
        been passed to :meth:`from_loop_output`.
        """
        T, B = self.positions.shape
        if self.mid_px is None:
            raise ValueError(
                "mid_px is required for per-instrument PnL; "
                "pass mid_px to from_loop_output"
            )

        H = self.tr_hedge_fills.shape[1] if self.n_trades > 0 else 0

        if self.n_trades > 0:
            inst_idx = self.tr_instrument.astype(np.intp)
            signed_qty = self.tr_side * self.tr_qty_fill

            # Trade cash per instrument: -Σ(signed_qty * vwap)
            trade_cash = np.zeros(B, dtype=np.float64)
            np.add.at(trade_cash, inst_idx, -signed_qty * self.tr_vwap)

            # Instrument execution cost per instrument
            inst_cost = np.zeros(B, dtype=np.float64)
            np.add.at(inst_cost, inst_idx, self.tr_cost)

            # Hedge execution cost per instrument
            hedge_cost = np.zeros(B, dtype=np.float64)
            np.add.at(hedge_cost, inst_idx, self.tr_hedge_cost)

            # Trade count per instrument
            n_trades_per = np.zeros(B, dtype=np.int64)
            np.add.at(n_trades_per, inst_idx, 1)

            # Hedge position contributed by each instrument: (B, H)
            hedge_pos_from_inst = np.zeros((B, max(H, 1)), dtype=np.float64)
            for h in range(H):
                np.add.at(hedge_pos_from_inst[:, h], inst_idx,
                          self.tr_hedge_fills[:, h])
        else:
            trade_cash = np.zeros(B, dtype=np.float64)
            inst_cost = np.zeros(B, dtype=np.float64)
            hedge_cost = np.zeros(B, dtype=np.float64)
            n_trades_per = np.zeros(B, dtype=np.int64)
            hedge_pos_from_inst = np.zeros((B, max(H, 1)), dtype=np.float64)

        # Direct instrument PnL = final position MTM + trade cash
        final_pos_mtm = self.positions[-1] * self.mid_px[-1]
        instrument_pnl = final_pos_mtm + trade_cash

        # -- Hedge PnL attribution --
        hedge_pnl_attr = np.zeros(B, dtype=np.float64)
        if H > 0 and self.hedge_mid_px is not None and self.n_trades > 0:
            # Reconstruct cash per hedge instrument from trade data
            hedge_cash_per_h = np.zeros(H, dtype=np.float64)
            for h in range(H):
                fills_h = self.tr_hedge_fills[:, h]
                vwaps_h = self.tr_hedge_vwaps[:, h]
                mask = np.abs(fills_h) > 1e-15
                if mask.any():
                    hedge_cash_per_h[h] = -np.sum(fills_h[mask] * vwaps_h[mask])

            # PnL per hedge instrument h = final MTM + cumulative cash
            hedge_pnl_per_h = (
                self.hedge_positions[-1, :H] * self.hedge_mid_px[-1, :H]
                + hedge_cash_per_h
            )

            # Attribute each hedge h's PnL to instruments proportionally
            for h in range(H):
                contribs = hedge_pos_from_inst[:, h]
                total_pos = self.hedge_positions[-1, h]
                if abs(total_pos) < 1e-15:
                    # Hedge fully unwound — attribute by absolute contribution
                    total_abs = np.sum(np.abs(contribs))
                    if total_abs < 1e-15:
                        continue
                    weights = np.abs(contribs) / total_abs
                else:
                    weights = contribs / total_pos
                hedge_pnl_attr += weights * hedge_pnl_per_h[h]

        # MTM = PnL as if traded at mid (no slippage)
        instrument_mtm = instrument_pnl + inst_cost
        hedge_mtm = hedge_pnl_attr + hedge_cost
        total_mtm = instrument_mtm + hedge_mtm
        total_cost = inst_cost + hedge_cost
        total_mtm_with_cost = total_mtm - total_cost

        df = pd.DataFrame({
            "instrument": self.instrument_ids,
            "final_position": self.positions[-1],
            "instrument_mtm": instrument_mtm,
            "instrument_cost": inst_cost,
            "hedge_mtm": hedge_mtm,
            "hedge_cost": hedge_cost,
            "total_mtm": total_mtm,
            "total_mtm_with_cost": total_mtm_with_cost,
            "n_trades": n_trades_per,
        })

        return df.sort_values("total_mtm_with_cost").reset_index(drop=True)
