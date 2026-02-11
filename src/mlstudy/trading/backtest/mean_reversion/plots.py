"""Matplotlib-based plots for MR backtest results.

Each function returns a :class:`matplotlib.figure.Figure`.
Import is optional — functions raise ImportError with a clear message
if matplotlib is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .analysis import compute_round_trips, to_dataframe
from .results import MRBacktestResults
from .types import (
    STATE_FLAT,
    STATE_LONG,
    STATE_SHORT,
    TRADE_ENTRY,
    TRADE_EXIT_SL,
    TRADE_EXIT_TIME,
    TRADE_EXIT_TP,
)

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


def _import_mpl():
    """Lazy import matplotlib; raise clear error if missing."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from None


# ---------------------------------------------------------------------------
# Equity curve + drawdown
# ---------------------------------------------------------------------------


def plot_equity_curve(res: MRBacktestResults) -> "matplotlib.figure.Figure":
    """Top: equity curve.  Bottom: drawdown fill."""
    plt = _import_mpl()
    df = to_dataframe(res)

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
    plt = _import_mpl()

    T = len(res.equity)
    bars = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Equity colored by state
    state_colors = {STATE_FLAT: "gray", STATE_LONG: "green", STATE_SHORT: "red"}
    for state_val, color in state_colors.items():
        mask = res.state == state_val
        ax1.scatter(
            bars[mask],
            res.equity[mask],
            c=color,
            s=2,
            label={STATE_FLAT: "flat", STATE_LONG: "long", STATE_SHORT: "short"}[
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
    plt = _import_mpl()

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
    plt = _import_mpl()

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
    plt = _import_mpl()

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
    plt = _import_mpl()

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
