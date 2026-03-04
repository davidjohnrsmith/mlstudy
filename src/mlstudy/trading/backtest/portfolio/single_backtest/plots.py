"""Matplotlib plots for a single LP portfolio backtest.

Axis-level functions accept an ``ax`` parameter (or create one),
draw on it, and return the Axes.  Figure-level functions create
complete multi-panel figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mlstudy.trading.backtest.portfolio.single_backtest.results import (
    PortfolioBacktestResults,
)
from mlstudy.trading.backtest.portfolio.single_backtest.state import (
    PortfolioActionCode,
)

if TYPE_CHECKING:
    import matplotlib.figure


# ---------------------------------------------------------------------------
# Shared legend / formatting
# ---------------------------------------------------------------------------

_LEGEND_KW = dict(
    loc="center right",
    bbox_to_anchor=(-0.1, 0.5),
    fontsize=6,
    framealpha=0.7,
    borderaxespad=0,
)


def _format_xaxis(ax, res: PortfolioBacktestResults) -> None:
    """Set x-axis tick labels to datetimes when available."""
    if res.datetimes is None:
        return
    dts = res.datetimes
    T = len(dts)

    def _fmt(x, _pos):
        i = int(round(x))
        if i < 0 or i >= T:
            return ""
        return pd.Timestamp(dts[i]).strftime("%Y-%m-%d\n%H:%M")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def _instrument_label(res: PortfolioBacktestResults, b: int) -> str:
    """Short label: instrument id + maturity (e.g. 'UST2Y 4.3y')."""
    if res.instrument_ids is not None and b < len(res.instrument_ids):
        name = str(res.instrument_ids[b])
    else:
        name = f"Inst {b}"
    if res.maturity is not None and b < len(res.maturity):
        mat = res.maturity[b]
        if mat < 1.0:
            name += f" {mat * 12:.0f}m"
        else:
            name += f" {mat:.1f}y"
    return name


def shade_inactive_hours(
    ax, res: PortfolioBacktestResults,
    inactive_start: str = "16:30",
    inactive_end: str = "08:30",
) -> None:
    """Add grey shading for inactive (non-trading) hours.

    Parameters
    ----------
    ax : matplotlib Axes
    res : PortfolioBacktestResults
        Must have ``datetimes`` set.
    inactive_start : str
        Time of day when inactive period starts (e.g. "16:30").
    inactive_end : str
        Time of day when inactive period ends (e.g. "08:30").
    """
    if res.datetimes is None:
        return
    dts = res.datetimes
    T = len(dts)
    if T == 0:
        return

    from datetime import time as dt_time

    h_s, m_s = (int(x) for x in inactive_start.split(":"))
    h_e, m_e = (int(x) for x in inactive_end.split(":"))
    t_start = dt_time(h_s, m_s)
    t_end = dt_time(h_e, m_e)

    # Identify contiguous inactive spans and shade them
    in_inactive = False
    span_start = 0
    for i in range(T):
        ts = pd.Timestamp(dts[i])
        t = ts.time()
        if t_start <= t_end:
            # simple range, e.g. 09:00 – 16:00
            is_inactive = t >= t_start or t < t_end
        else:
            # overnight range, e.g. 16:30 – 08:30
            is_inactive = t >= t_start or t < t_end
        if is_inactive and not in_inactive:
            span_start = i
            in_inactive = True
        elif not is_inactive and in_inactive:
            ax.axvspan(span_start - 0.5, i - 0.5, color="grey",
                       alpha=0.08, zorder=0)
            in_inactive = False
    if in_inactive:
        ax.axvspan(span_start - 0.5, T - 0.5, color="grey",
                   alpha=0.08, zorder=0)


# ---------------------------------------------------------------------------
# Trade overlay helpers
# ---------------------------------------------------------------------------

_SIDE_STYLE = {
    +1: {"marker": "^", "color": "blue", "label": "Buy"},
    -1: {"marker": "v", "color": "red", "label": "Sell"},
}


def _overlay_trades(
    ax, res: PortfolioBacktestResults, y_data: np.ndarray,
) -> None:
    """Scatter buy/sell markers at trade bars, mapped to *y_data*."""
    if res.n_trades == 0:
        return
    T = len(y_data)
    for side, style in _SIDE_STYLE.items():
        mask = res.tr_side == side
        if not np.any(mask):
            continue
        bars = res.tr_bar[mask].astype(int)
        valid = (bars >= 0) & (bars < T)
        bars = bars[valid]
        ax.scatter(
            bars, y_data[bars],
            marker=style["marker"], color=style["color"],
            s=40, zorder=5, label=style["label"],
            edgecolors="black", linewidths=0.5,
        )


# ---------------------------------------------------------------------------
# Ax-level plots
# ---------------------------------------------------------------------------

def plot_equity_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Equity line with trade markers."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    ax.plot(bars, res.equity, linewidth=0.8, color="steelblue", label="Equity")
    _overlay_trades(ax, res, res.equity)
    ax.set_ylabel("Equity")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_drawdown_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
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


def plot_portfolio_mtm_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Portfolio MTM line."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    ax.plot(bars, res.portfolio_mtm[:T], linewidth=0.8, color="steelblue",
            label="Portfolio MTM")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Portfolio MTM")
    _format_xaxis(ax, res)

    # Portfolio MTM with cost on right axis
    ax2 = ax.twinx()
    mtm_with_cost = res.portfolio_mtm[:T] - np.cumsum(res.portfolio_cost[:T])
    ax2.plot(bars, mtm_with_cost, linewidth=0.8, color="darkorange",
             label="MTM with cost")
    ax2.set_ylabel("MTM with cost")

    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, **_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_cost_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Cumulative cost breakdown: portfolio on left, instrument/hedge on right."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    cum_portfolio = np.cumsum(res.portfolio_cost[:T])
    ax.plot(bars, cum_portfolio, linewidth=0.8, color="steelblue",
            label="Portfolio cost")
    ax.set_ylabel("Portfolio cost")
    _format_xaxis(ax, res)

    ax2 = ax.twinx()
    cum_inst = np.cumsum(res.instrument_cost[:T])
    cum_hedge = np.cumsum(res.hedge_cost_bar[:T])
    ax2.plot(bars, cum_inst, linewidth=0.8, color="darkorange",
             label="Instrument cost")
    ax2.plot(bars, cum_hedge, linewidth=0.8, color="green",
             label="Hedge cost")
    ax2.set_ylabel("Component cost")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, **_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_component_mtm_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Instrument MTM and hedge MTM lines."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    inst_mtm = res.instrument_position_mtm[:T] + res.instrument_cash_mtm[:T]
    hedge_mtm = res.hedge_position_mtm[:T] + res.hedge_cash_mtm[:T]
    ax.plot(bars, inst_mtm, linewidth=0.8, color="steelblue",
            label="Instrument MTM")
    ax.plot(bars, hedge_mtm, linewidth=0.8, color="darkorange",
            label="Hedge MTM")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Component MTM")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_positions_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Stacked area of per-instrument positions."""
    if ax is None:
        _, ax = plt.subplots()
    pos = res.positions
    if pos.ndim != 2 or pos.shape[1] == 0:
        return ax
    T = pos.shape[0]
    bars = np.arange(T)
    for b in range(pos.shape[1]):
        ax.plot(bars, pos[:, b], linewidth=0.7, alpha=0.8, label=f"Inst {b}")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Position")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_hedge_positions_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Line plot of hedge positions per instrument."""
    if ax is None:
        _, ax = plt.subplots()
    hpos = res.hedge_positions
    if hpos.ndim != 2 or hpos.shape[1] == 0:
        return ax
    T = hpos.shape[0]
    bars = np.arange(T)
    H = hpos.shape[1]
    for h in range(H):
        label = f"Hedge {h}"
        if res.hedge_ids is not None and h < len(res.hedge_ids):
            label = str(res.hedge_ids[h])
        ax.plot(bars, hpos[:, h], linewidth=0.7, alpha=0.8, label=label)
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Hedge Pos")
    _format_xaxis(ax, res)
    ncol = 1 if H <= 15 else 2
    ax.legend(**_LEGEND_KW, ncol=ncol)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pnl_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Bar chart of per-bar PnL."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.pnl)
    bars = np.arange(T)
    colors = np.where(res.pnl >= 0, "green", "red")
    ax.bar(bars, res.pnl, color=colors, width=1.0, alpha=0.6)
    ax.set_ylabel("PnL")
    _format_xaxis(ax, res)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_gross_dv01_on_ax(res: PortfolioBacktestResults, ax=None, cap: float | None = None) -> plt.Axes:
    """Gross DV01: left=total+cap, right=instrument+hedge."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    gross_total = res.gross_instrument_dv01[:T] + res.gross_hedge_dv01[:T]
    ax.plot(bars, gross_total, linewidth=0.8, color="steelblue",
            label="Gross Total")
    if cap is not None:
        ax.axhline(cap, color="red", linewidth=0.5, linestyle="--",
                   alpha=0.7, label=f"Cap={cap}")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Gross DV01")
    _format_xaxis(ax, res)

    ax2 = ax.twinx()
    ax2.plot(bars, res.gross_instrument_dv01[:T], linewidth=0.8,
             color="darkorange", label="Gross Instrument")
    ax2.plot(bars, res.gross_hedge_dv01[:T], linewidth=0.8,
             color="green", label="Gross Hedge")
    ax2.set_ylabel("Component Gross DV01")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, **_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_net_dv01_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Net DV01: left=net total, right=net instrument+hedge."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    net_total = res.net_instrument_dv01[:T] + res.net_hedge_dv01[:T]
    ax.plot(bars, net_total, linewidth=0.8, color="steelblue",
            label="Net Total")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Net DV01")
    _format_xaxis(ax, res)

    ax2 = ax.twinx()
    ax2.plot(bars, res.net_instrument_dv01[:T], linewidth=0.8,
             color="darkorange", label="Net Instrument")
    ax2.plot(bars, res.net_hedge_dv01[:T], linewidth=0.8,
             color="green", label="Net Hedge")
    ax2.set_ylabel("Component Net DV01")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, **_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_position_count_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Number of non-zero positions per bar."""
    if ax is None:
        _, ax = plt.subplots()
    pos = res.positions
    if pos.ndim != 2 or pos.shape[1] == 0:
        ax.set_ylabel("# Positions")
        return ax
    T = pos.shape[0]
    bars = np.arange(T)
    count = np.sum(np.abs(pos) > 1e-15, axis=1)
    ax.step(bars, count, where="mid", linewidth=0.7, color="purple")
    ax.set_ylabel("# Positions")
    _format_xaxis(ax, res)
    ax.grid(True, alpha=0.3)
    return ax


def plot_position_heatmap_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Heatmap of positions (T x B), instruments on y-axis, time on x-axis."""
    if ax is None:
        _, ax = plt.subplots()
    pos = res.positions
    if pos.ndim != 2 or pos.shape[1] == 0:
        return ax
    vmax = np.nanmax(np.abs(pos))
    if vmax < 1e-15:
        vmax = 1.0
    B = pos.shape[1]
    # Sort instruments by maturity if available
    if res.maturity is not None and len(res.maturity) == B:
        order = np.argsort(res.maturity)[::-1]
    else:
        order = np.arange(B)
    im = ax.imshow(
        pos[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    # Label y-axis with instrument names (readable for up to ~30; skip for many)
    if B <= 30:
        labels = [_instrument_label(res, b) for b in order]
        ax.set_yticks(range(B))
        ax.set_yticklabels(labels, fontsize=max(5, 8 - B // 10))
    else:
        ax.set_ylabel("Instrument (by maturity)")
    ax.set_xlabel("Bar")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Position")
    return ax


def plot_top_k_positions_on_ax(
    res: PortfolioBacktestResults, ax=None, k: int = 10,
) -> plt.Axes:
    """Line plot of top-K instruments by max |pos * dv01|."""
    if ax is None:
        _, ax = plt.subplots()
    pos = res.positions
    if pos.ndim != 2 or pos.shape[1] == 0:
        return ax
    T, B = pos.shape
    bars = np.arange(T)
    # Rank by max absolute DV01 contribution
    if res.dv01 is not None and res.dv01.shape == (T, B):
        exposure = np.abs(pos * res.dv01)
    else:
        exposure = np.abs(pos)
    max_exp = np.nanmax(exposure, axis=0)  # (B,)
    top_idx = np.argsort(max_exp)[::-1][:k]
    n_plotted = 0
    for rank, b in enumerate(top_idx):
        if max_exp[b] < 1e-15:
            break
        label = _instrument_label(res, b)
        ax.plot(bars, pos[:, b], linewidth=0.7, alpha=0.8, label=label)
        n_plotted += 1
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Position")
    ax.set_title(f"Top {n_plotted} Positions (by max DV01 exposure)", fontsize=9)
    _format_xaxis(ax, res)
    ncol = 1 if n_plotted <= 15 else 2
    ax.legend(**_LEGEND_KW, ncol=ncol)
    ax.grid(True, alpha=0.3)
    return ax


def plot_dv01_breakdown_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Stacked area of net instrument DV01 vs net hedge DV01."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.equity)
    bars = np.arange(T)
    inst = res.net_instrument_dv01[:T]
    hedge = res.net_hedge_dv01[:T]
    ax.fill_between(bars, 0, inst, alpha=0.5, color="steelblue",
                    label="Instrument")
    ax.fill_between(bars, inst, inst + hedge, alpha=0.5, color="darkorange",
                    label="Hedge")
    ax.plot(bars, inst + hedge, linewidth=0.8, color="black",
            label="Net Total", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Net DV01")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
    ax.grid(True, alpha=0.3)
    return ax


def plot_n_trades_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Trades-per-bar as a step plot."""
    if ax is None:
        _, ax = plt.subplots()
    T = len(res.n_trades_bar)
    bars = np.arange(T)
    ax.step(bars, res.n_trades_bar, where="mid", linewidth=0.7, color="teal")
    ax.set_ylabel("# Trades")
    _format_xaxis(ax, res)
    ax.grid(True, alpha=0.3)
    return ax


# Code colour mapping: group codes into semantic categories for visual clarity.
_CODE_COLORS = {
    PortfolioActionCode.NO_ACTION: ("lightgrey", "No action"),
    PortfolioActionCode.NO_CANDIDATES: ("silver", "No candidates"),
    PortfolioActionCode.SKIP_COOLDOWN: ("gold", "Cooldown"),
    PortfolioActionCode.SKIP_COOLDOWN_RISK_ONLY: ("khaki", "Cooldown (risk)"),
    PortfolioActionCode.EXEC_OK: ("mediumseagreen", "Exec OK"),
    PortfolioActionCode.EXEC_PARTIAL: ("yellowgreen", "Exec partial"),
    PortfolioActionCode.EXEC_NO_LIQUIDITY: ("darkorange", "No liquidity"),
    PortfolioActionCode.EXEC_GREEDY: ("dodgerblue", "Greedy"),
    PortfolioActionCode.LP_INFEASIBLE: ("tomato", "LP infeasible"),
    PortfolioActionCode.LP_NO_CANDIDATES: ("lightsalmon", "LP no cands"),
    PortfolioActionCode.INVALID_BOOK: ("red", "Invalid book"),
    PortfolioActionCode.INVALID_DV01: ("darkred", "Invalid DV01"),
}


def plot_codes_on_ax(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Color-coded bar-level action codes over time."""
    if ax is None:
        _, ax = plt.subplots()
    codes = res.codes
    T = len(codes)
    if T == 0:
        ax.set_ylabel("Code")
        return ax

    # Map each code to a numeric id for pcolormesh
    unique_codes = sorted(set(int(c) for c in codes))
    code_to_idx = {c: i for i, c in enumerate(unique_codes)}
    mapped = np.array([code_to_idx[int(c)] for c in codes])

    # Build colourmap from known codes
    from matplotlib.colors import ListedColormap
    colors = []
    labels = []
    for c in unique_codes:
        try:
            pac = PortfolioActionCode(c)
            col, lbl = _CODE_COLORS.get(pac, ("grey", pac.name))
        except ValueError:
            col, lbl = "grey", f"Code {c}"
        colors.append(col)
        labels.append(lbl)

    cmap = ListedColormap(colors)

    # pcolormesh with shading="flat" needs edges: T+1 x-edges for T cells
    x_edges = np.arange(T + 1) - 0.5
    ax.pcolormesh(
        x_edges, [0, 1], mapped.reshape(1, -1),
        cmap=cmap, vmin=-0.5, vmax=len(unique_codes) - 0.5,
        shading="flat",
    )
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_yticks([])
    ax.set_ylabel("Code")
    _format_xaxis(ax, res)

    # Legend with code labels
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=col, label=lbl) for col, lbl in zip(colors, labels)]
    ax.legend(handles=handles, **_LEGEND_KW)

    return ax


def plot_codes_distribution(res: PortfolioBacktestResults, ax=None) -> plt.Axes:
    """Horizontal bar chart of action code counts."""
    if ax is None:
        _, ax = plt.subplots()
    codes = res.codes
    unique, counts = np.unique(codes, return_counts=True)

    labels = []
    colors = []
    for c in unique:
        try:
            pac = PortfolioActionCode(int(c))
            col, lbl = _CODE_COLORS.get(pac, ("grey", pac.name))
        except ValueError:
            col, lbl = "grey", f"Code {int(c)}"
        labels.append(lbl)
        colors.append(col)

    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Action Code Distribution", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    return ax


# ---------------------------------------------------------------------------
# Figure-level plots
# ---------------------------------------------------------------------------

def plot_equity_curve(res: PortfolioBacktestResults) -> matplotlib.figure.Figure:
    """Two-panel: equity + drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plot_equity_on_ax(res, ax=ax1)
    ax1.set_title("Portfolio Equity Curve")
    plot_drawdown_on_ax(res, ax=ax2)
    fig.tight_layout()
    return fig


def plot_portfolio_dashboard(
    res: PortfolioBacktestResults,
    gross_dv01_cap: float | None = None,
    figsize: tuple[float, float] = (14, 12),
) -> matplotlib.figure.Figure:
    """Multi-panel dashboard: equity, drawdown, DV01, position count, PnL.

    Panels included dynamically based on data availability.
    Uses aggregate panels that scale to any number of instruments.
    """
    has_positions = res.positions.ndim == 2 and res.positions.shape[1] > 0

    panels: list[tuple[str, int]] = [
        ("equity", 3),
        ("drawdown", 1),
        ("dv01", 2),
    ]
    if has_positions:
        panels.append(("pos_count", 1))
    panels.append(("pnl", 1))
    panels.append(("codes", 1))

    n = len(panels)
    height_ratios = [h for _, h in panels]

    fig, axes = plt.subplots(
        n, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, panels):
        if name == "equity":
            plot_equity_on_ax(res, ax=ax)
            ax.set_title("Portfolio Backtest", fontsize=12, fontweight="bold")
        elif name == "drawdown":
            plot_drawdown_on_ax(res, ax=ax)
        elif name == "dv01":
            plot_gross_dv01_on_ax(res, ax=ax, cap=gross_dv01_cap)
        elif name == "pos_count":
            plot_position_count_on_ax(res, ax=ax)
        elif name == "pnl":
            plot_pnl_on_ax(res, ax=ax)
        elif name == "codes":
            plot_codes_on_ax(res, ax=ax)

    fig.tight_layout()
    return fig


def plot_trade_alpha_distribution(res: PortfolioBacktestResults) -> matplotlib.figure.Figure:
    """Histogram of executable alpha (bps) at trade time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if res.n_trades == 0:
        ax.set_title("Trade Alpha (no trades)")
        return fig
    ax.hist(res.tr_alpha, bins="auto", color="steelblue", edgecolor="white")
    ax.set_xlabel("Alpha (bps)")
    ax.set_ylabel("Frequency")
    ax.set_title("Executable Alpha Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_execution_cost(res: PortfolioBacktestResults) -> matplotlib.figure.Figure:
    """Per-trade cost breakdown: instrument cost + hedge cost."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if res.n_trades == 0:
        ax.set_title("Execution Cost (no trades)")
        return fig
    x = np.arange(res.n_trades)
    ax.bar(x, res.tr_cost, label="Instrument cost", alpha=0.7, color="steelblue")
    ax.bar(x, res.tr_hedge_cost, bottom=res.tr_cost, label="Hedge cost", alpha=0.7, color="darkorange")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cost")
    ax.set_title("Per-Trade Execution Cost")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
