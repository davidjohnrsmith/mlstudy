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
    plot_portfolio_mtm_on_ax,
    plot_cost_on_ax,
    plot_component_mtm_on_ax,
    plot_net_dv01_on_ax,
    plot_positions_on_ax,
    plot_hedge_positions_on_ax,
    plot_gross_dv01_on_ax,
    plot_pnl_on_ax,
    plot_n_trades_on_ax,
    plot_codes_on_ax,
    plot_position_count_on_ax,
    plot_position_heatmap_on_ax,
    plot_top_k_positions_on_ax,
    plot_dv01_breakdown_on_ax,
    shade_inactive_hours,
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
    plot_config: dict | None = None,
) -> matplotlib.figure.Figure:
    """Multi-panel dashboard for a single portfolio sweep scenario.

    Panels (aggregate-level, scales to any number of instruments):
      1. Equity           (height 3) — always
      2. Drawdown         (height 1) — always
      3. Portfolio MTM    (height 2) — always
      4. Component MTM    (height 2) — instrument vs hedge MTM
      5. DV01 net/gross   (height 2) — always
      6. Position count   (height 1) — if positions available
      7. PnL bars         (height 1) — always
      8. Trades/bar       (height 1) — always
      9. Codes            (height 1) — always

    Parameters
    ----------
    scenario : PortfolioFullScenario
        Loaded scenario from disk.
    save_path : str or Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size in inches.
    plot_config : dict, optional
        Plot configuration (e.g. ``inactive_start``, ``inactive_end``
        for shading non-trading hours).
    """
    res = scenario.results

    has_positions = res.positions.ndim == 2 and res.positions.shape[1] > 0
    panels: list[tuple[str, int]] = [
        ("equity", 3),
        ("drawdown", 1),
        ("portfolio_mtm", 2),
        ("cost", 2),
        ("component_mtm", 2),
        ("net_dv01", 2),
        ("gross_dv01", 2),
    ]
    if has_positions:
        panels.append(("pos_count", 1))
    panels.append(("pnl", 1))
    panels.append(("trades", 1))
    panels.append(("codes", 1))

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
        elif name == "portfolio_mtm":
            plot_portfolio_mtm_on_ax(res, ax=ax)
        elif name == "cost":
            plot_cost_on_ax(res, ax=ax)
        elif name == "component_mtm":
            plot_component_mtm_on_ax(res, ax=ax)
        elif name == "net_dv01":
            plot_net_dv01_on_ax(res, ax=ax)
        elif name == "gross_dv01":
            plot_gross_dv01_on_ax(res, ax=ax)
        elif name == "pos_count":
            plot_position_count_on_ax(res, ax=ax)
        elif name == "pnl":
            plot_pnl_on_ax(res, ax=ax)
        elif name == "trades":
            plot_n_trades_on_ax(res, ax=ax)
        elif name == "codes":
            plot_codes_on_ax(res, ax=ax)

    # Shade inactive hours on all panels
    pcfg = plot_config or {}
    if "inactive_start" in pcfg and "inactive_end" in pcfg:
        for ax in axes:
            shade_inactive_hours(
                ax, res,
                inactive_start=pcfg["inactive_start"],
                inactive_end=pcfg["inactive_end"],
            )

    _add_stats_textbox(fig, scenario)

    fig.tight_layout()
    fig.subplots_adjust(left=0.25, right=0.78)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_scenario_detail(
    scenario: PortfolioFullScenario,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (16, 14),
    top_k: int = 10,
) -> matplotlib.figure.Figure:
    """Detail dashboard with per-instrument breakdowns.

    Panels:
      1. Position heatmap     (height 3) — all instruments
      2. Top-K positions      (height 2) — top K by max |pos * dv01|
      3. DV01 breakdown       (height 2) — instrument vs hedge stacked
      4. Hedge positions      (height 2) — if hedges present

    Parameters
    ----------
    scenario : PortfolioFullScenario
        Loaded scenario from disk.
    save_path : str or Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size in inches.
    top_k : int
        Number of top instruments to show in line plot.
    """
    res = scenario.results

    has_positions = res.positions.ndim == 2 and res.positions.shape[1] > 0
    has_hedges = (
        res.hedge_positions.ndim == 2 and res.hedge_positions.shape[1] > 0
    )

    panels: list[tuple[str, int]] = []
    if has_positions:
        panels.append(("heatmap", 3))
        panels.append(("top_k", 2))
    if has_hedges:
        panels.append(("hedges", 2))

    if not panels:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No position data", ha="center", va="center")
        return fig

    n_panels = len(panels)
    height_ratios = [h for _, h in panels]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, panels):
        if name == "heatmap":
            plot_position_heatmap_on_ax(res, ax=ax)
            ax.set_title(
                f"{scenario.name} — Detail", fontsize=12, fontweight="bold",
            )
        elif name == "top_k":
            plot_top_k_positions_on_ax(res, ax=ax, k=top_k)
        elif name == "hedges":
            plot_hedge_positions_on_ax(res, ax=ax)

    fig.tight_layout()

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
    detail: bool = True,
    top_k: int = 10,
) -> list[matplotlib.figure.Figure]:
    """Plot dashboards for a list of scenarios (e.g. top-k from a sweep).

    Parameters
    ----------
    scenarios : list[PortfolioFullScenario]
        Scenarios to plot, in rank order.
    save_dir : str or Path, optional
        If provided, each figure is saved as ``scenario_<rank>.png``
        (and ``scenario_<rank>_detail.png`` when *detail* is True).
    figsize : tuple
        Figure size per plot.
    detail : bool
        If True, also generate a detail figure per scenario with
        position heatmap, top-K positions, and DV01 breakdown.
    top_k : int
        Number of top instruments in the detail line plot.

    Returns
    -------
    list[matplotlib.figure.Figure]
    """
    from pathlib import Path

    figs = []
    for rank, sc in enumerate(scenarios):
        save_path = None
        detail_save = None
        if save_dir is not None:
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            save_path = sd / f"scenario_{rank:03d}.png"
            if detail:
                detail_save = sd / f"scenario_{rank:03d}_detail.png"

        fig = plot_scenario(sc, save_path=save_path, figsize=figsize)
        figs.append(fig)

        if detail:
            fig_d = plot_scenario_detail(
                sc, save_path=detail_save, figsize=figsize, top_k=top_k,
            )
            figs.append(fig_d)

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
        0.88, 0.95, text,
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
