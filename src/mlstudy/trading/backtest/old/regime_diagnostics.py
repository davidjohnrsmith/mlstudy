"""Regime diagnostics for backtest reports.

Provides visualizations and statistics for regime-aware backtesting,
including regime overlays, signal/stop event plots, and performance
breakdown by regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlstudy.trading.strategy.alpha.regime.regime import Regime

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class RegimeDiagnostics:
    """Regime diagnostic statistics.

    Attributes:
        pct_mean_revert: Percentage of time in mean-reverting regime.
        pct_trend: Percentage of time in trending regime.
        pct_uncertain: Percentage of time in uncertain regime.
        pnl_mean_revert: Total P&L during mean-reverting regime.
        pnl_trend: Total P&L during trending regime.
        pnl_uncertain: Total P&L during uncertain regime.
        sharpe_mean_revert: Sharpe ratio during mean-reverting regime.
        sharpe_trend: Sharpe ratio during trending regime.
        sharpe_uncertain: Sharpe ratio during uncertain regime.
        n_entries_mean_revert: Number of entries during mean-reverting regime.
        n_entries_trend: Number of entries during trending regime.
        n_entries_uncertain: Number of entries during uncertain regime.
        n_stops: Number of stop-loss exits.
        n_max_hold_exits: Number of max-hold exits.
        n_normal_exits: Number of normal exits.
    """

    pct_mean_revert: float
    pct_trend: float
    pct_uncertain: float
    pnl_mean_revert: float
    pnl_trend: float
    pnl_uncertain: float
    sharpe_mean_revert: float
    sharpe_trend: float
    sharpe_uncertain: float
    n_entries_mean_revert: int
    n_entries_trend: int
    n_entries_uncertain: int
    n_stops: int
    n_max_hold_exits: int
    n_normal_exits: int

    def to_dict(self) -> dict:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "pct_mean_revert": float(self.pct_mean_revert),
            "pct_trend": float(self.pct_trend),
            "pct_uncertain": float(self.pct_uncertain),
            "pnl_mean_revert": float(self.pnl_mean_revert),
            "pnl_trend": float(self.pnl_trend),
            "pnl_uncertain": float(self.pnl_uncertain),
            "sharpe_mean_revert": float(self.sharpe_mean_revert),
            "sharpe_trend": float(self.sharpe_trend),
            "sharpe_uncertain": float(self.sharpe_uncertain),
            "n_entries_mean_revert": int(self.n_entries_mean_revert),
            "n_entries_trend": int(self.n_entries_trend),
            "n_entries_uncertain": int(self.n_entries_uncertain),
            "n_stops": int(self.n_stops),
            "n_max_hold_exits": int(self.n_max_hold_exits),
            "n_normal_exits": int(self.n_normal_exits),
        }


def compute_regime_diagnostics(
    pnl_df: pd.DataFrame,
    regime: pd.Series,
    exit_reasons: pd.Series | None = None,
    datetime_col: str = "datetime",
    return_col: str = "net_return",
    position_col: str = "position",
) -> RegimeDiagnostics:
    """Compute regime diagnostic statistics.

    Args:
        pnl_df: P&L DataFrame from mlstudy.trading.backtest.
        regime: Series of Regime enum values aligned with pnl_df.
        exit_reasons: Optional series of ExitReason values.
        datetime_col: Datetime column name.
        return_col: Return column name.
        position_col: Position column name.

    Returns:
        RegimeDiagnostics with computed statistics.
    """
    # Align regime with pnl_df
    if datetime_col in pnl_df.columns:
        df = pnl_df.set_index(datetime_col).copy()
        regime_aligned = regime.reindex(df.index)
    else:
        df = pnl_df.copy()
        regime_aligned = regime.iloc[: len(df)]

    # Map regime to string for grouping
    regime_str = regime_aligned.apply(
        lambda x: x.value if isinstance(x, Regime) else str(x)
    )

    # Time percentages
    regime_counts = regime_str.value_counts(normalize=True)
    pct_mean_revert = regime_counts.get("mean_revert", 0.0)
    pct_trend = regime_counts.get("trend", 0.0)
    pct_uncertain = regime_counts.get("uncertain", 0.0)

    # P&L by regime
    df["regime"] = regime_str.values
    pnl_by_regime = df.groupby("regime")[return_col].sum()
    pnl_mean_revert = pnl_by_regime.get("mean_revert", 0.0)
    pnl_trend = pnl_by_regime.get("trend", 0.0)
    pnl_uncertain = pnl_by_regime.get("uncertain", 0.0)

    # Sharpe by regime
    def _regime_sharpe(returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    sharpe_by_regime = df.groupby("regime")[return_col].apply(_regime_sharpe)
    sharpe_mean_revert = sharpe_by_regime.get("mean_revert", 0.0)
    sharpe_trend = sharpe_by_regime.get("trend", 0.0)
    sharpe_uncertain = sharpe_by_regime.get("uncertain", 0.0)

    # Count entries by regime
    position = df[position_col]
    entries = (position != 0) & (position.shift(1).fillna(0) == 0)

    n_entries_mean_revert = int(entries[df["regime"] == "mean_revert"].sum())
    n_entries_trend = int(entries[df["regime"] == "trend"].sum())
    n_entries_uncertain = int(entries[df["regime"] == "uncertain"].sum())

    # Count exit types
    n_stops = 0
    n_max_hold_exits = 0
    n_normal_exits = 0

    if exit_reasons is not None:
        from mlstudy.trading.strategy.alpha.mean_reversion.signals import ExitReason

        exit_aligned = exit_reasons.reindex(df.index) if datetime_col in pnl_df.columns else exit_reasons.iloc[: len(df)]
        n_stops = int((exit_aligned == ExitReason.STOP_LOSS).sum())
        n_max_hold_exits = int((exit_aligned == ExitReason.MAX_HOLD).sum())
        n_normal_exits = int((exit_aligned == ExitReason.NORMAL_EXIT).sum())

    return RegimeDiagnostics(
        pct_mean_revert=pct_mean_revert,
        pct_trend=pct_trend,
        pct_uncertain=pct_uncertain,
        pnl_mean_revert=pnl_mean_revert,
        pnl_trend=pnl_trend,
        pnl_uncertain=pnl_uncertain,
        sharpe_mean_revert=sharpe_mean_revert,
        sharpe_trend=sharpe_trend,
        sharpe_uncertain=sharpe_uncertain,
        n_entries_mean_revert=n_entries_mean_revert,
        n_entries_trend=n_entries_trend,
        n_entries_uncertain=n_entries_uncertain,
        n_stops=n_stops,
        n_max_hold_exits=n_max_hold_exits,
        n_normal_exits=n_normal_exits,
    )


def plot_fly_with_regime(
    fly_yield: pd.Series,
    regime: pd.Series,
    ax: plt.Axes | None = None,
    title: str = "Fly Yield with Regime Overlay",
) -> plt.Figure | None:
    """Plot fly yield series with regime background overlay.

    Args:
        fly_yield: Fly yield time series.
        regime: Series of Regime enum values.
        ax: Optional matplotlib axes. Creates new figure if None.
        title: Plot title.

    Returns:
        Figure if ax was None, else None.
    """
    if not HAS_MATPLOTLIB:
        return None

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = None

    # Color mapping for regimes
    regime_colors = {
        Regime.MEAN_REVERT: "#90EE90",  # Light green
        Regime.TREND: "#FFB6C1",  # Light pink/red
        Regime.UNCERTAIN: "#D3D3D3",  # Light gray
        "mean_revert": "#90EE90",
        "trend": "#FFB6C1",
        "uncertain": "#D3D3D3",
    }

    # Align series
    common_idx = fly_yield.index.intersection(regime.index)
    fly_aligned = fly_yield.loc[common_idx]
    regime_aligned = regime.loc[common_idx]

    # Plot regime background spans
    current_regime = None
    span_start = None

    for idx, reg in regime_aligned.items():
        reg_key = reg.value if isinstance(reg, Regime) else reg

        if reg_key != current_regime:
            # End previous span
            if current_regime is not None and span_start is not None:
                color = regime_colors.get(current_regime, "#D3D3D3")
                ax.axvspan(span_start, idx, alpha=0.3, color=color, linewidth=0)

            # Start new span
            current_regime = reg_key
            span_start = idx

    # Close final span
    if current_regime is not None and span_start is not None:
        color = regime_colors.get(current_regime, "#D3D3D3")
        ax.axvspan(span_start, fly_aligned.index[-1], alpha=0.3, color=color, linewidth=0)

    # Plot fly yield
    ax.plot(fly_aligned.index, fly_aligned.values, color="blue", linewidth=1.0, label="Fly Yield")
    ax.axhline(y=fly_aligned.mean(), color="gray", linestyle="--", alpha=0.5, label="Mean")

    # Legend for regimes
    legend_elements = [
        Patch(facecolor="#90EE90", alpha=0.3, label="Mean-Revert"),
        Patch(facecolor="#FFB6C1", alpha=0.3, label="Trend"),
        Patch(facecolor="#D3D3D3", alpha=0.3, label="Uncertain"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Fly Yield")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_zscore_positions_stops(
    zscore: pd.Series,
    signal: pd.Series,
    exit_reasons: pd.Series | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float | None = 4.0,
    ax: plt.Axes | None = None,
    title: str = "Z-Score, Positions & Stop Events",
) -> plt.Figure | None:
    """Plot z-score with positions and stop/exit events.

    Args:
        zscore: Z-score time series.
        signal: Signal/position series (-1, 0, 1).
        exit_reasons: Optional ExitReason series for marking stop events.
        entry_z: Entry threshold for reference lines.
        exit_z: Exit threshold for reference lines.
        stop_z: Stop threshold for reference lines.
        ax: Optional matplotlib axes.
        title: Plot title.

    Returns:
        Figure if ax was None, else None.
    """
    if not HAS_MATPLOTLIB:
        return None

    created_fig = ax is None
    if ax is None:
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )
        ax_zscore = axes[0]
        ax_position = axes[1]
    else:
        ax_zscore = ax
        ax_position = None
        fig = None

    # Align series
    common_idx = zscore.index.intersection(signal.index)
    z_aligned = zscore.loc[common_idx]
    sig_aligned = signal.loc[common_idx]

    # Plot z-score
    ax_zscore.plot(z_aligned.index, z_aligned.values, color="blue", linewidth=1.0, label="Z-Score")

    # Entry/exit/stop thresholds
    ax_zscore.axhline(y=entry_z, color="green", linestyle="--", alpha=0.7, label=f"Entry ({entry_z})")
    ax_zscore.axhline(y=-entry_z, color="green", linestyle="--", alpha=0.7)
    ax_zscore.axhline(y=exit_z, color="orange", linestyle=":", alpha=0.7, label=f"Exit ({exit_z})")
    ax_zscore.axhline(y=-exit_z, color="orange", linestyle=":", alpha=0.7)
    if stop_z is not None:
        ax_zscore.axhline(y=stop_z, color="red", linestyle="-.", alpha=0.7, label=f"Stop ({stop_z})")
        ax_zscore.axhline(y=-stop_z, color="red", linestyle="-.", alpha=0.7)
    ax_zscore.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # Mark stop events
    if exit_reasons is not None:
        from mlstudy.trading.strategy.alpha.mean_reversion.signals import ExitReason

        exit_aligned = exit_reasons.reindex(common_idx)

        # Stop-loss markers
        stops = exit_aligned == ExitReason.STOP_LOSS
        if stops.any():
            stop_idx = stops[stops].index
            stop_z_vals = z_aligned.loc[stop_idx]
            ax_zscore.scatter(
                stop_idx, stop_z_vals, color="red", marker="x", s=100,
                zorder=5, label="Stop-Loss"
            )

        # Max-hold markers
        max_holds = exit_aligned == ExitReason.MAX_HOLD
        if max_holds.any():
            mh_idx = max_holds[max_holds].index
            mh_z_vals = z_aligned.loc[mh_idx]
            ax_zscore.scatter(
                mh_idx, mh_z_vals, color="purple", marker="s", s=80,
                zorder=5, label="Max-Hold Exit"
            )

    ax_zscore.set_ylabel("Z-Score")
    ax_zscore.set_title(title)
    ax_zscore.legend(loc="upper right", fontsize=8)
    ax_zscore.grid(True, alpha=0.3)

    # Plot position on second axis
    if ax_position is not None:
        ax_position.fill_between(
            sig_aligned.index, 0, sig_aligned.values,
            where=sig_aligned > 0, color="green", alpha=0.5, step="post", label="Long"
        )
        ax_position.fill_between(
            sig_aligned.index, 0, sig_aligned.values,
            where=sig_aligned < 0, color="red", alpha=0.5, step="post", label="Short"
        )
        ax_position.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax_position.set_xlabel("Date")
        ax_position.set_ylabel("Position")
        ax_position.set_ylim(-1.5, 1.5)
        ax_position.legend(loc="upper right")
        ax_position.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def save_regime_plots(
    output_path: Path,
    fly_yield: pd.Series,
    zscore: pd.Series,
    signal: pd.Series,
    regime: pd.Series,
    exit_reasons: pd.Series | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float | None = 4.0,
) -> dict:
    """Save regime diagnostic plots to files.

    Args:
        output_path: Directory to save plots.
        fly_yield: Fly yield time series.
        zscore: Z-score time series.
        signal: Signal/position series.
        regime: Regime classification series.
        exit_reasons: Optional ExitReason series.
        entry_z: Entry threshold.
        exit_z: Exit threshold.
        stop_z: Stop threshold.

    Returns:
        Dict with paths to saved plot files.
    """
    if not HAS_MATPLOTLIB:
        return {}

    saved = {}

    # Plot 1: Fly yield with regime overlay
    fig = plot_fly_with_regime(fly_yield, regime)
    if fig is not None:
        path = output_path / "fly_regime_overlay.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved["plot_fly_regime"] = str(path)

    # Plot 2: Z-score with positions and stops
    fig = plot_zscore_positions_stops(
        zscore, signal, exit_reasons,
        entry_z=entry_z, exit_z=exit_z, stop_z=stop_z
    )
    if fig is not None:
        path = output_path / "zscore_positions_stops.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved["plot_zscore_positions"] = str(path)

    return saved


def print_regime_summary(diagnostics: RegimeDiagnostics) -> None:
    """Print formatted regime diagnostic summary.

    Args:
        diagnostics: RegimeDiagnostics from compute_regime_diagnostics.
    """
    print("\n" + "=" * 60)
    print("REGIME DIAGNOSTICS")
    print("=" * 60)

    print("\n--- Time in Regime ---")
    print(f"  Mean-Revert:      {diagnostics.pct_mean_revert:>15.1%}")
    print(f"  Trend:            {diagnostics.pct_trend:>15.1%}")
    print(f"  Uncertain:        {diagnostics.pct_uncertain:>15.1%}")

    print("\n--- P&L by Regime ---")
    print(f"  Mean-Revert:      {diagnostics.pnl_mean_revert:>15,.4f}")
    print(f"  Trend:            {diagnostics.pnl_trend:>15,.4f}")
    print(f"  Uncertain:        {diagnostics.pnl_uncertain:>15,.4f}")

    print("\n--- Sharpe by Regime ---")
    print(f"  Mean-Revert:      {diagnostics.sharpe_mean_revert:>15.2f}")
    print(f"  Trend:            {diagnostics.sharpe_trend:>15.2f}")
    print(f"  Uncertain:        {diagnostics.sharpe_uncertain:>15.2f}")

    print("\n--- Entries by Regime ---")
    print(f"  Mean-Revert:      {diagnostics.n_entries_mean_revert:>15d}")
    print(f"  Trend:            {diagnostics.n_entries_trend:>15d}")
    print(f"  Uncertain:        {diagnostics.n_entries_uncertain:>15d}")

    print("\n--- Exit Types ---")
    print(f"  Normal Exits:     {diagnostics.n_normal_exits:>15d}")
    print(f"  Stop-Loss:        {diagnostics.n_stops:>15d}")
    print(f"  Max-Hold:         {diagnostics.n_max_hold_exits:>15d}")

    print("\n" + "=" * 60)
