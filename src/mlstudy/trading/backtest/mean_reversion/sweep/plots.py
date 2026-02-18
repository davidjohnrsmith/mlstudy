"""Matplotlib-based plots for MR backtest results.

Each function returns a :class:`matplotlib.figure.Figure`.
Import is optional — functions raise ImportError with a clear message
if matplotlib is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import (
    _TRADE_STYLE,
    _overlay_trades,
    plot_equity_on_ax,
    plot_zscore_on_ax,
    plot_mid_and_vwap,
    plot_package_yield,
    plot_codes_on_ax,
    plot_drawdown_on_ax,
)
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import FullScenario

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.figure


# ---------------------------------------------------------------------------
# Scenario dashboard: equity + zscore + mid/vwap + yield + codes + drawdown
# ---------------------------------------------------------------------------


def plot_scenario(
    scenario: FullScenario,
    save_path: "str | Path | None" = None,
    figsize: tuple[float, float] = (16, 10),
) -> "matplotlib.figure.Figure":
    """Plot a single scenario dashboard with up to 6 panels.

    Panels (included when data is available on ``scenario.results``):
      1. Equity — always shown (height 3)
      2. Z-score — if ``res.zscore`` is not None (height 2)
      3. Mid + VWAP — if ``res.mid_px`` is not None (height 2)
      4. Package yield — if ``res.package_yield_bps`` is not None (height 2)
      5. Codes — always shown (height 1)
      6. Drawdown — always shown (height 1)

    Parameters
    ----------
    scenario : FullScenario
        Loaded scenario from ``load_sweep_run(...).full_scenarios[i]``.
    save_path : str or Path, optional
        If provided, the figure is saved to this path (PNG, PDF, etc.).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """

    res = scenario.results
    T = len(res.equity)

    has_zscore = res.zscore is not None and len(res.zscore) == T
    has_mid = res.mid_px is not None and res.mid_px.shape[0] == T
    has_yield = res.package_yield_bps is not None and len(res.package_yield_bps) == T

    # Build panel specs: (name, height, draw_fn_kwargs)
    panels: list[tuple[str, int]] = []
    panels.append(("equity", 3))
    if has_zscore:
        panels.append(("zscore", 2))
    if has_mid:
        panels.append(("mid_vwap", 2))
    if has_yield:
        panels.append(("pkg_yield", 2))
    panels.append(("codes", 1))
    panels.append(("drawdown", 1))

    n_panels = len(panels)
    height_ratios = [h for _, h in panels]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (panel_name, _) in zip(axes, panels):
        if panel_name == "equity":
            plot_equity_on_ax(res, ax=ax)
            ax.set_title(scenario.name, fontsize=12, fontweight="bold")
        elif panel_name == "zscore":
            entry_z = scenario.config.get("entry_z_threshold")
            plot_zscore_on_ax(res, ax=ax, entry_z=entry_z)
        elif panel_name == "mid_vwap":
            ref_leg = scenario.config.get("ref_leg_idx", 0)
            plot_mid_and_vwap(res, ax=ax, leg=ref_leg)
        elif panel_name == "pkg_yield":
            plot_package_yield(res, ax=ax)
        elif panel_name == "codes":
            plot_codes_on_ax(res, ax=ax)
        elif panel_name == "drawdown":
            plot_drawdown_on_ax(res, ax=ax)

    # --- Text box: parameters + metrics ---
    _add_stats_textbox(fig, scenario)

    fig.tight_layout()
    fig.subplots_adjust(left=0.25, right=0.78)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_top_scenarios(
    scenarios: list[FullScenario],
    save_dir: "str | Path | None" = None,
    figsize: tuple[float, float] = (16, 10),
) -> list["matplotlib.figure.Figure"]:
    """Plot dashboards for a list of scenarios (e.g. top-k from a sweep run).

    Parameters
    ----------
    scenarios : list[FullScenario]
        Scenarios to plot.
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

        fig = plot_scenario(sc, save_path=save_path, figsize=figsize)
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
