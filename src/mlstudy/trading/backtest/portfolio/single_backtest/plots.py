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
    for h in range(hpos.shape[1]):
        ax.plot(bars, hpos[:, h], linewidth=0.7, alpha=0.8, label=f"Hedge {h}")
    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_ylabel("Hedge Pos")
    _format_xaxis(ax, res)
    ax.legend(**_LEGEND_KW)
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
    """Gross DV01 exposure over time."""
    if ax is None:
        _, ax = plt.subplots()
    pos = res.positions
    if pos.ndim != 2:
        return ax
    T = pos.shape[0]
    bars = np.arange(T)
    gross = np.sum(np.abs(pos), axis=1)
    ax.plot(bars, gross, linewidth=0.8, color="darkorange", label="Gross DV01")
    if cap is not None:
        ax.axhline(cap, color="red", linewidth=0.5, linestyle="--", alpha=0.7, label=f"Cap={cap}")
    ax.set_ylabel("Gross DV01")
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
    """Multi-panel dashboard: equity, drawdown, positions, hedges, DV01, PnL.

    Panels included dynamically based on data availability.
    """
    panels: list[tuple[str, int]] = [
        ("equity", 3),
        ("drawdown", 1),
        ("positions", 2),
    ]
    if res.hedge_positions.ndim == 2 and res.hedge_positions.shape[1] > 0:
        panels.append(("hedges", 2))
    panels.append(("dv01", 2))
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
        elif name == "positions":
            plot_positions_on_ax(res, ax=ax)
        elif name == "hedges":
            plot_hedge_positions_on_ax(res, ax=ax)
        elif name == "dv01":
            plot_gross_dv01_on_ax(res, ax=ax, cap=gross_dv01_cap)
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
