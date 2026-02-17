"""Matplotlib-based plots for MR backtest results.

Each function returns a :class:`matplotlib.figure.Figure`.
Import is optional — functions raise ImportError with a clear message
if matplotlib is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import FullScenario
from .analysis import to_dataframe
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
    from pathlib import Path

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


# ---------------------------------------------------------------------------
# Trade-type marker styles
# ---------------------------------------------------------------------------

_TRADE_STYLE = {
    TRADE_ENTRY: {"marker": "^", "color": "blue", "label": "Entry"},
    TRADE_EXIT_TP: {"marker": "v", "color": "green", "label": "Exit TP"},
    TRADE_EXIT_SL: {"marker": "X", "color": "red", "label": "Exit SL"},
    TRADE_EXIT_TIME: {"marker": "s", "color": "orange", "label": "Exit Time"},
}


# ---------------------------------------------------------------------------
# Scenario dashboard: equity + zscore + trades + params/metrics
# ---------------------------------------------------------------------------


def plot_scenario(
    scenario: FullScenario,
    zscore: np.ndarray | None = None,
    save_path: "str | Path | None" = None,
    figsize: tuple[float, float] = (16, 10),
) -> "matplotlib.figure.Figure":
    """Plot a single scenario dashboard: equity, zscore, trades, and stats.

    Parameters
    ----------
    scenario : FullScenario
        Loaded scenario from ``load_sweep_run(...).full_scenarios[i]``.
    zscore : ndarray (T,), optional
        Z-score signal array.  When provided, a second panel shows the
        z-score time series with entry threshold bands.
    save_path : str or Path, optional
        If provided, the figure is saved to this path (PNG, PDF, etc.).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_mpl()
    res = scenario.results
    T = len(res.equity)
    bars = np.arange(T)

    has_zscore = zscore is not None and len(zscore) == T
    n_panels = 3 if has_zscore else 2
    height_ratios = [3, 2, 2] if has_zscore else [3, 2]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # --- Panel 1: Equity curve with trade markers ---
    ax_eq = axes[0]
    ax_eq.plot(bars, res.equity, linewidth=0.8, color="steelblue", label="Equity")

    # Shade position periods
    for t in range(T):
        if res.state[t] == STATE_LONG:
            ax_eq.axvspan(t - 0.5, t + 0.5, alpha=0.05, color="green", linewidth=0)
        elif res.state[t] == STATE_SHORT:
            ax_eq.axvspan(t - 0.5, t + 0.5, alpha=0.05, color="red", linewidth=0)

    # Trade markers on equity curve
    _overlay_trades(ax_eq, res, bars, res.equity)

    ax_eq.set_ylabel("Equity")
    ax_eq.set_title(scenario.name, fontsize=12, fontweight="bold")
    ax_eq.legend(loc="upper left", fontsize=8, ncol=3)
    ax_eq.grid(True, alpha=0.3)

    # --- Panel 2 (optional): Z-score ---
    if has_zscore:
        ax_z = axes[1]
        ax_z.plot(bars, zscore, linewidth=0.7, color="purple", alpha=0.8)
        ax_z.axhline(0, color="gray", linewidth=0.5)

        # Entry threshold bands from config
        entry_z = scenario.config.get("entry_z_threshold")
        if entry_z is not None:
            ax_z.axhline(entry_z, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
            ax_z.axhline(-entry_z, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
            ax_z.fill_between(
                bars, -entry_z, entry_z,
                alpha=0.04, color="blue", label=f"Entry band (|z|>{entry_z})",
            )

        # Trade markers on zscore
        _overlay_trades(ax_z, res, bars, zscore)

        ax_z.set_ylabel("Z-score")
        ax_z.legend(loc="upper left", fontsize=8, ncol=3)
        ax_z.grid(True, alpha=0.3)

    # --- Panel: Drawdown ---
    ax_dd = axes[-1]
    cum_max = np.maximum.accumulate(res.equity)
    drawdown = res.equity - cum_max
    ax_dd.fill_between(bars, drawdown, 0, color="salmon", alpha=0.5)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Bar")
    ax_dd.grid(True, alpha=0.3)

    # --- Text box: parameters + metrics ---
    _add_stats_textbox(fig, scenario)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_top_scenarios(
    scenarios: list[FullScenario],
    zscore: np.ndarray | None = None,
    save_dir: "str | Path | None" = None,
    figsize: tuple[float, float] = (16, 10),
) -> list["matplotlib.figure.Figure"]:
    """Plot dashboards for a list of scenarios (e.g. top-k from a sweep run).

    Parameters
    ----------
    scenarios : list[FullScenario]
        Scenarios to plot.
    zscore : ndarray (T,), optional
        Z-score signal (shared across scenarios since the signal is the same).
    save_dir : str or Path, optional
        If provided, each figure is saved as
        ``<save_dir>/scenario_<rank>.png``.
    figsize : tuple
        Figure size per plot.

    Returns
    -------
    list[matplotlib.figure.Figure]
    """
    from pathlib import Path

    figs = []
    for rank, sc in enumerate(scenarios):
        save_path = None
        if save_dir is not None:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            save_path = sd / f"scenario_{rank:03d}.png"

        fig = plot_scenario(sc, zscore=zscore, save_path=save_path, figsize=figsize)
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _overlay_trades(ax, res: MRBacktestResults, bars: np.ndarray, y_data: np.ndarray) -> None:
    """Scatter trade markers on an axis, using y_data for vertical position."""
    if res.n_trades == 0:
        return

    for trade_type, style in _TRADE_STYLE.items():
        mask = res.tr_type == trade_type
        if not np.any(mask):
            continue
        trade_bars = res.tr_bar[mask].astype(int)
        # Clip to valid range
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


def _add_stats_textbox(fig, scenario: FullScenario) -> None:
    """Add a text box on the right margin with parameters and metrics."""
    lines = []

    # Parameters
    tags = scenario.tags
    if tags:
        lines.append("PARAMETERS")
        lines.append("-" * 28)
        for k, v in tags.items():
            label = k.replace("_", " ")
            if isinstance(v, float):
                lines.append(f"  {label}: {v:.4g}")
            else:
                lines.append(f"  {label}: {v}")
        lines.append("")

    # Metrics
    metrics = scenario.metrics
    if metrics:
        lines.append("METRICS")
        lines.append("-" * 28)
        _KEY_ORDER = [
            "total_pnl", "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "n_trades", "hit_rate", "profit_factor", "calmar_ratio",
            "avg_holding_period", "pct_time_in_market",
            "avg_win", "avg_loss", "win_loss_ratio",
            "var_95", "cvar_95",
        ]
        shown = set()
        for k in _KEY_ORDER:
            if k in metrics:
                _append_metric_line(lines, k, metrics[k])
                shown.add(k)
        # Any remaining metrics not in _KEY_ORDER
        for k, v in metrics.items():
            if k not in shown:
                _append_metric_line(lines, k, v)

    text = "\n".join(lines)
    fig.text(
        0.80, 0.95, text,
        fontsize=7,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )


def _append_metric_line(lines: list[str], key: str, value) -> None:
    label = key.replace("_", " ")
    if isinstance(value, float):
        if abs(value) >= 100:
            lines.append(f"  {label}: {value:,.1f}")
        else:
            lines.append(f"  {label}: {value:.4f}")
    else:
        lines.append(f"  {label}: {value}")
