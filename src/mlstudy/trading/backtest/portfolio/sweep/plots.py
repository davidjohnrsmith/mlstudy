"""Sweep-level plots for portfolio backtests.

Provides scenario dashboards and top-N plotting, mirroring the MR
sweep plots but using portfolio-specific panels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from mlstudy.trading.backtest.portfolio.single_backtest.plots import (
    plot_equity_on_ax,
    plot_drawdown_on_ax,
    plot_positions_on_ax,
    plot_hedge_positions_on_ax,
    plot_gross_dv01_on_ax,
    plot_pnl_on_ax,
    plot_n_trades_on_ax,
)
from mlstudy.trading.backtest.portfolio.sweep.sweep_results_reader import (
    PortfolioFullScenario,
)

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.figure


# ---------------------------------------------------------------------------
# Scenario dashboard
# ---------------------------------------------------------------------------

def plot_scenario(
    scenario: PortfolioFullScenario,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (16, 12),
) -> matplotlib.figure.Figure:
    """Multi-panel dashboard for a single portfolio sweep scenario.

    Panels (included when data is available):
      1. Equity        (height 3) — always
      2. Drawdown      (height 1) — always
      3. Positions     (height 2) — if positions has instruments
      4. Hedge pos     (height 2) — if hedge positions present
      5. Gross DV01    (height 2) — if positions available
      6. PnL bars      (height 1) — always
      7. Trades/bar    (height 1) — always

    Parameters
    ----------
    scenario : PortfolioFullScenario
        Loaded scenario from disk.
    save_path : str or Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size in inches.
    """
    res = scenario.results

    has_positions = res.positions.ndim == 2 and res.positions.shape[1] > 0
    has_hedges = (
        res.hedge_positions.ndim == 2 and res.hedge_positions.shape[1] > 0
    )
    cap = scenario.config.get("gross_dv01_cap")

    panels: list[tuple[str, int]] = [("equity", 3), ("drawdown", 1)]
    if has_positions:
        panels.append(("positions", 2))
    if has_hedges:
        panels.append(("hedges", 2))
    if has_positions:
        panels.append(("dv01", 2))
    panels.append(("pnl", 1))
    panels.append(("trades", 1))

    n_panels = len(panels)
    height_ratios = [h for _, h in panels]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, panels):
        if name == "equity":
            plot_equity_on_ax(res, ax=ax)
            ax.set_title(scenario.name, fontsize=12, fontweight="bold")
        elif name == "drawdown":
            plot_drawdown_on_ax(res, ax=ax)
        elif name == "positions":
            plot_positions_on_ax(res, ax=ax)
        elif name == "hedges":
            plot_hedge_positions_on_ax(res, ax=ax)
        elif name == "dv01":
            plot_gross_dv01_on_ax(res, ax=ax, cap=cap)
        elif name == "pnl":
            plot_pnl_on_ax(res, ax=ax)
        elif name == "trades":
            plot_n_trades_on_ax(res, ax=ax)

    _add_stats_textbox(fig, scenario)

    fig.tight_layout()
    fig.subplots_adjust(left=0.25, right=0.78)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Top-N plotting
# ---------------------------------------------------------------------------


def plot_top_scenarios(
    scenarios: list[PortfolioFullScenario],
    save_dir: str | Path | None = None,
    figsize: tuple[float, float] = (16, 12),
) -> list[matplotlib.figure.Figure]:
    """Plot dashboards for a list of scenarios (e.g. top-k from a sweep).

    Parameters
    ----------
    scenarios : list[PortfolioFullScenario]
        Scenarios to plot, in rank order.
    save_dir : str or Path, optional
        If provided, each figure is saved as ``scenario_<rank>.png``.
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

        fig = plot_scenario(sc, save_path=save_path, figsize=figsize)
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Stats text box
# ---------------------------------------------------------------------------


def _add_stats_textbox(fig, scenario: PortfolioFullScenario) -> None:
    """Add a text box on the right margin with parameters and metrics."""
    lines: list[str] = []

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

    metrics = scenario.metrics
    if metrics:
        lines.append("METRICS")
        lines.append("-" * 28)
        _KEY_ORDER = [
            "total_pnl", "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "n_trades", "hit_rate", "profit_factor", "calmar_ratio",
            "var_95", "cvar_95",
        ]
        shown: set[str] = set()
        for k in _KEY_ORDER:
            if k in metrics:
                _append_metric_line(lines, k, metrics[k])
                shown.add(k)
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
