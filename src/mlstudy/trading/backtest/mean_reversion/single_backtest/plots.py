from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults
from mlstudy.trading.backtest.mean_reversion.single_backtest.state import (
    ActionCode, CODE_NAMES, State, TradeType,
)

# ---------------------------------------------------------------------------
# Trade-type marker styles (shared by all subplot functions)
# ---------------------------------------------------------------------------

_TRADE_STYLE = {
    TradeType.TRADE_ENTRY: {"marker": "^", "color": "blue", "label": "Entry"},
    TradeType.TRADE_EXIT_TP: {"marker": "v", "color": "green", "label": "Exit TP"},
    TradeType.TRADE_EXIT_SL: {"marker": "X", "color": "red", "label": "Exit SL"},
    TradeType.TRADE_EXIT_TIME: {"marker": "s", "color": "orange", "label": "Exit Time"},
}


def _overlay_trades(ax, res: MRBacktestResults, bars: np.ndarray, y_data: np.ndarray) -> None:
    """Scatter trade markers on an axis, using y_data for vertical position."""
    if res.n_trades == 0:
        return

    for trade_type, style in _TRADE_STYLE.items():
        mask = res.tr_type == trade_type
        if not np.any(mask):
            continue
        trade_bars = res.tr_bar[mask].astype(int)
        valid = (trade_bars >= 0) & (trade_bars < len(y_data))
        trade_bars = trade_bars[valid]
        ax.scatter(
            trade_bars,
            y_data[trade_bars],
            marker=style["marker"],
            color=style["color"],
            s=60,
            zorder=5,
            label=style["label"],
            edgecolors="black",
            linewidths=0.5,
        )


# ---------------------------------------------------------------------------
# Reusable ax-level subplot functions
# ---------------------------------------------------------------------------

_LEGEND_KW = dict(
    loc="center right",
    bbox_to_anchor=(-0.1, 0.5),
    fontsize=6,
    framealpha=0.7,
    borderaxespad=0,
)


def _format_xaxis(ax, res: MRBacktestResults) -> None:
    """Set x-axis tick labels to datetimes when available.

    Data is still plotted against continuous bar indices (no gaps),
    but tick labels show the corresponding datetime from ``res.datetimes``.
    """
    if res.datetimes is None:
        return
    dts = res.datetimes
    T = len(dts)

    def _fmt(x, _pos):
        i = int(round(x))
        if i < 0 or i >= T:
            return ""
        dt = dts[i]
        ts = pd.Timestamp(dt)
        return ts.strftime("%Y-%m-%d\n%H:%M")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def plot_equity_on_ax(res: MRBacktestResults, ax=None) -> plt.Axes:
    """Equity line with state shading and trade markers."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    ax.plot(bars, res.equity, linewidth=0.8, color="steelblue", label="Equity")
    for t in range(T):
        if res.state[t] == State.STATE_LONG:
            ax.axvspan(t - 0.5, t + 0.5, alpha=0.05, color="green", linewidth=0)
        elif res.state[t] == State.STATE_SHORT:
            ax.axvspan(t - 0.5, t + 0.5, alpha=0.05, color="red", linewidth=0)
    _overlay_trades(ax, res, bars, res.equity)
    ax.set_ylabel("Equity")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_zscore_on_ax(res: MRBacktestResults, ax=None, entry_z=None) -> plt.Axes:
    """Z-score line with optional entry threshold bands and trade markers."""
    if ax is None:
        _, ax = plt.subplots()
    if res.zscore is None:
        return ax
    T = len(res.zscore)
    bars = np.arange(T)
    ax.plot(bars, res.zscore, linewidth=0.7, color="purple", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    if entry_z is not None:
        ax.axhline(entry_z, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(-entry_z, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.fill_between(
            bars, -entry_z, entry_z,
            alpha=0.04, color="blue", label=f"Entry band (|z|>{entry_z})",
        )
    _overlay_trades(ax, res, bars, res.zscore)
    ax.set_ylabel("Z-score")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_mid_and_vwap(res: MRBacktestResults, ax=None, leg: int = 0) -> plt.Axes:
    """Line plot of mid_px for a leg, with VWAP and mid scatter at trade bars."""
    if ax is None:
        _, ax = plt.subplots()
    if res.mid_px is None:
        return ax
    T = res.mid_px.shape[0]
    bars = np.arange(T)
    mid_series = res.mid_px[:, leg] if res.mid_px.ndim == 2 else res.mid_px
    ax.plot(bars, mid_series, linewidth=0.7, color="steelblue", alpha=0.8, label=f"Mid (leg {leg})")
    if res.n_trades > 0:
        trade_bars = res.tr_bar.astype(int)
        valid = (trade_bars >= 0) & (trade_bars < T)
        tb = trade_bars[valid]
        vwap_col = res.tr_vwaps[valid, leg] if res.tr_vwaps.ndim == 2 else res.tr_vwaps[valid]
        mid_col = res.tr_mids[valid, leg] if res.tr_mids.ndim == 2 else res.tr_mids[valid]
        ax.scatter(tb, vwap_col, marker="D", color="orange", s=40, zorder=5, label="VWAP", edgecolors="black", linewidths=0.5)
        ax.scatter(tb, mid_col, marker="o", color="purple", s=30, zorder=5, label="Mid@trade", edgecolors="black", linewidths=0.5)
    ax.set_ylabel("Price")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_package_yield(res: MRBacktestResults, ax=None) -> plt.Axes:
    """Line plot of package_yield_bps with trade markers."""
    if ax is None:
        _, ax = plt.subplots()
    if res.package_yield_bps is None:
        return ax
    T = len(res.package_yield_bps)
    bars = np.arange(T)
    ax.plot(bars, res.package_yield_bps, linewidth=0.7, color="teal", alpha=0.8, label="Pkg yield (bps)")
    _overlay_trades(ax, res, bars, res.package_yield_bps)
    ax.set_ylabel("Yield (bps)")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


_CODE_COLORS: dict[int, str] = {
    # 0xx: no-action — muted
    ActionCode.NO_ACTION: "silver",
    ActionCode.NO_ACTION_NO_SIGNAL: "lightgray",
    ActionCode.NO_ACTION_HOLD: "darkgray",
    # 1xx: entry
    ActionCode.ENTRY_OK: "blue",
    ActionCode.ENTRY_FAILED_INVALID_DV01: "lightsteelblue",
    ActionCode.ENTRY_FAILED_INVALID_BOOK: "lightsteelblue",
    ActionCode.ENTRY_FAILED_TOO_WIDE: "cornflowerblue",
    ActionCode.ENTRY_FAILED_NO_LIQUIDITY: "cornflowerblue",
    ActionCode.ENTRY_FAILED_IN_COOLDOWN: "slategray",
    # 2xx: exit TP
    ActionCode.EXIT_TP_OK: "green",
    ActionCode.EXIT_TP_OK_WITH_COOLDOWN: "limegreen",
    ActionCode.EXIT_FAILED_TP_INVALID_BOOK: "lightgreen",
    ActionCode.EXIT_FAILED_TP_TOO_WIDE: "palegreen",
    ActionCode.EXIT_FAILED_TP_NO_LIQUIDITY: "palegreen",
    # 3xx: exit SL
    ActionCode.EXIT_SL_OK: "red",
    ActionCode.EXIT_SL_OK_WITH_COOLDOWN: "tomato",
    ActionCode.EXIT_FAILED_SL_INVALID_BOOK: "lightsalmon",
    ActionCode.EXIT_FAILED_SL_NO_LIQUIDITY: "lightsalmon",
    # 4xx: exit time
    ActionCode.EXIT_TIME_OK: "darkorange",
    ActionCode.EXIT_TIME_OK_WITH_COOLDOWN: "orange",
    ActionCode.EXIT_FAILED_TIME_INVALID_BOOK: "moccasin",
    ActionCode.EXIT_FAILED_TIME_NO_LIQUIDITY: "moccasin",
}


def plot_codes_on_ax(res: MRBacktestResults, ax=None) -> plt.Axes:
    """Scatter plot of action codes, colored per ActionCode with legend."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.codes)
    bars = np.arange(T)
    for code in sorted(np.unique(res.codes)):
        code = int(code)
        color = _CODE_COLORS.get(code, "black")
        label = CODE_NAMES.get(code, str(code))
        mask = res.codes == code
        ax.scatter(bars[mask], res.codes[mask], c=color, s=8, alpha=0.7, label=label)
    ax.set_ylabel("Code")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW, ncol=1, markerscale=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_drawdown_on_ax(res: MRBacktestResults, ax=None) -> plt.Axes:
    """Drawdown fill plot."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    cum_max = np.maximum.accumulate(res.equity)
    drawdown = res.equity - cum_max
    ax.fill_between(bars, drawdown, 0, color="salmon", alpha=0.5)
    ax.set_ylabel("Drawdown")
    _format_xaxis(ax, res)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Equity curve + drawdown
# ---------------------------------------------------------------------------

def plot_equity_curve(res: MRBacktestResults) -> "matplotlib.figure.Figure":
    """Top: equity curve.  Bottom: drawdown fill."""
    
    df = res.bar_df

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(df.index, df["equity"], linewidth=0.8, color="steelblue")
    ax1.set_ylabel("Equity")
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)

    cum_pnl = df["cumulative_pnl"]
    rolling_max = cum_pnl.cummax()
    drawdown = cum_pnl - rolling_max
    ax2.fill_between(df.index, drawdown, 0, color="salmon", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Bar")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# State and codes
# ---------------------------------------------------------------------------
def plot_state_and_codes(res: MRBacktestResults) -> "matplotlib.figure.Figure":
    """Top: equity colored by state.  Bottom: attempt codes as scatter."""
    

    T = len(res.equity)
    bars = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Equity colored by state
    state_colors = {State.STATE_FLAT: "gray", State.STATE_LONG: "green", State.STATE_SHORT: "red"}
    for state_val, color in state_colors.items():
        mask = res.state == state_val
        ax1.scatter(
            bars[mask],
            res.equity[mask],
            c=color,
            s=2,
            label={State.STATE_FLAT: "flat", State.STATE_LONG: "long", State.STATE_SHORT: "short"}[
                state_val
            ],
        )
    ax1.set_ylabel("Equity")
    ax1.set_title("Equity by State")
    ax1.legend(loc="upper left", markerscale=4)
    ax1.grid(True, alpha=0.3)

    # Codes as scatter: color by category
    code_colors = {}
    for code in np.unique(res.codes):
        code = int(code)
        cat = code // 100
        if cat == 0:
            code_colors[code] = "gray"
        elif cat == 1:
            code_colors[code] = "blue"
        elif cat == 2:
            code_colors[code] = "green"
        elif cat == 3:
            code_colors[code] = "red"
        elif cat == 4:
            code_colors[code] = "orange"
        else:
            code_colors[code] = "black"

    for code_val, color in code_colors.items():
        mask = res.codes == code_val
        if np.any(mask):
            ax2.scatter(bars[mask], res.codes[mask], c=color, s=8, alpha=0.7)
    ax2.set_ylabel("Code")
    ax2.set_xlabel("Bar")
    ax2.set_title("Attempt/Outcome Codes")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Trade PnL bar chart
# ---------------------------------------------------------------------------
def plot_trade_pnl(round_trips: "pd.DataFrame") -> "matplotlib.figure.Figure":
    """Bar chart: per-trade PnL, colored by exit type."""
    

    fig, ax = plt.subplots(figsize=(12, 4))
    if round_trips.empty:
        ax.set_title("Trade PnL (no trades)")
        return fig

    exit_colors = {"tp": "green", "sl": "red", "time": "orange"}
    colors = [exit_colors.get(et, "gray") for et in round_trips["exit_type"]]
    ax.bar(range(len(round_trips)), round_trips["pnl"], color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL")
    ax.set_title("Per-Trade PnL by Exit Type")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Exit breakdown
# ---------------------------------------------------------------------------


def plot_exit_breakdown(round_trips: "pd.DataFrame") -> "matplotlib.figure.Figure":
    """Two subplots: (1) count by exit type, (2) mean PnL by exit type."""
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if round_trips.empty:
        ax1.set_title("Exit Count (no trades)")
        ax2.set_title("Mean PnL (no trades)")
        return fig

    grouped = round_trips.groupby("exit_type")
    counts = grouped.size()
    mean_pnl = grouped["pnl"].mean()

    exit_colors = {"tp": "green", "sl": "red", "time": "orange"}

    ax1.bar(
        counts.index,
        counts.values,
        color=[exit_colors.get(k, "gray") for k in counts.index],
    )
    ax1.set_title("Count by Exit Type")
    ax1.set_ylabel("Count")

    ax2.bar(
        mean_pnl.index,
        mean_pnl.values,
        color=[exit_colors.get(k, "gray") for k in mean_pnl.index],
    )
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Mean PnL by Exit Type")
    ax2.set_ylabel("Mean PnL")

    fig.tight_layout()
    return fig




# ---------------------------------------------------------------------------
# Holding distribution
# ---------------------------------------------------------------------------


def plot_holding_distribution(
    round_trips: "pd.DataFrame",
) -> "matplotlib.figure.Figure":
    """Histogram of holding_bars."""
    

    fig, ax = plt.subplots(figsize=(8, 4))
    if round_trips.empty:
        ax.set_title("Holding Distribution (no trades)")
        return fig

    ax.hist(round_trips["holding_bars"], bins="auto", color="steelblue", edgecolor="white")
    ax.set_xlabel("Holding Bars")
    ax.set_ylabel("Frequency")
    ax.set_title("Holding Period Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Slippage box plot
# ---------------------------------------------------------------------------


def plot_slippage(res: MRBacktestResults) -> "matplotlib.figure.Figure":
    """Per-leg boxplot of |vwap - mid| across all trades."""
    

    fig, ax = plt.subplots(figsize=(8, 4))

    if res.n_trades == 0:
        ax.set_title("Slippage (no trades)")
        return fig

    # Build per-leg slippage arrays
    slippage = np.abs(res.tr_vwaps - res.tr_mids)  # (n_trades, N)
    n_legs = slippage.shape[1] if slippage.ndim == 2 else 1

    if slippage.ndim == 2:
        data = [slippage[:, leg] for leg in range(n_legs)]
        labels = [f"Leg {leg}" for leg in range(n_legs)]
    else:
        data = [slippage]
        labels = ["Leg 0"]

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("|VWAP - Mid|")
    ax.set_title("Execution Slippage by Leg")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig
