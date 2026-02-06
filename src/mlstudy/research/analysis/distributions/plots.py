"""Visualization functions for distribution comparison."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike


def plot_histogram_overlay(
    data: dict[str, ArrayLike],
    title: str = "Distribution Comparison",
    xlabel: str = "Value",
    bins: int | str = "auto",
    kde: bool = True,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot overlaid histograms/KDEs for multiple groups.

    Args:
        data: Dict mapping group name to array of values.
        title: Plot title.
        xlabel: X-axis label.
        bins: Number of bins or binning strategy.
        kde: Whether to overlay KDE curves.
        figsize: Figure size (width, height).
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, values in data.items():
        values = np.asarray(values)
        sns.histplot(
            values,
            bins=bins,
            kde=kde,
            label=group_name,
            alpha=0.5,
            stat="density",
            ax=ax,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ecdf_overlay(
    data: dict[str, ArrayLike],
    title: str = "ECDF Comparison",
    xlabel: str = "Value",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot overlaid empirical CDFs for multiple groups.

    Args:
        data: Dict mapping group name to array of values.
        title: Plot title.
        xlabel: X-axis label.
        figsize: Figure size (width, height).
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, values in data.items():
        values = np.asarray(values)
        sorted_vals = np.sort(values)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.step(sorted_vals, ecdf, label=group_name, where="post")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distance_heatmap(
    distance_matrix: pd.DataFrame,
    title: str = "Distribution Distance Matrix",
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".3f",
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot heatmap of a distance matrix.

    Args:
        distance_matrix: Square DataFrame with groups as index and columns.
        title: Plot title.
        cmap: Colormap name.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
        figsize: Figure size. If None, auto-computed based on matrix size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    n = len(distance_matrix)
    if figsize is None:
        size = max(6, n * 0.8)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        distance_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_group_distributions(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: Sequence[str] | None = None,
    plot_type: str = "both",
    outdir: str | Path | None = None,
) -> list[plt.Figure]:
    """Convenience function to generate distribution plots.

    Args:
        df: Input DataFrame.
        group_col: Column name for grouping.
        value_col: Column name for values to compare.
        groups: List of groups to include. If None, use all groups.
        plot_type: One of "hist", "ecdf", or "both".
        outdir: If provided, save plots to this directory.

    Returns:
        List of generated Figure objects.
    """
    if groups is None:
        groups = df[group_col].unique().tolist()

    data = {g: df[df[group_col] == g][value_col].dropna().values for g in groups}

    figures = []

    if plot_type in ("hist", "both"):
        save_path = Path(outdir) / "histogram_overlay.png" if outdir else None
        fig = plot_histogram_overlay(
            data,
            title=f"Distribution of {value_col} by {group_col}",
            xlabel=value_col,
            save_path=save_path,
        )
        figures.append(fig)

    if plot_type in ("ecdf", "both"):
        save_path = Path(outdir) / "ecdf_overlay.png" if outdir else None
        fig = plot_ecdf_overlay(
            data,
            title=f"ECDF of {value_col} by {group_col}",
            xlabel=value_col,
            save_path=save_path,
        )
        figures.append(fig)

    return figures
