"""Main API for grouped distribution comparison."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from mlstudy.research.analysis.distributions.metrics import (
    effect_size_metrics,
    jensen_shannon_divergence,
    ks_test,
    wasserstein,
)
from mlstudy.research.analysis.distributions.report import DistributionComparisonReport


def compute_group_summaries(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compute summary statistics for each group.

    Args:
        df: Input DataFrame.
        group_col: Column name for grouping.
        value_col: Column name for values to summarize.

    Returns:
        DataFrame with groups as index and statistics as columns:
        count, missing, mean, std, min, max, q05, q25, q50, q75, q95,
        skew, kurtosis.
    """
    results = []

    for group_name, group_df in df.groupby(group_col):
        values = group_df[value_col]
        valid = values.dropna()

        if len(valid) == 0:
            row = {
                "group": group_name,
                "count": 0,
                "missing": len(values),
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q50": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "skew": np.nan,
                "kurtosis": np.nan,
            }
        else:
            quantiles = np.percentile(valid, [5, 25, 50, 75, 95])
            row = {
                "group": group_name,
                "count": len(valid),
                "missing": len(values) - len(valid),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "q05": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "q50": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "q95": float(quantiles[4]),
                "skew": float(stats.skew(valid)),
                "kurtosis": float(stats.kurtosis(valid)),
            }
        results.append(row)

    summary_df = pd.DataFrame(results).set_index("group")
    return summary_df


def compute_pairwise_stats(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise distribution comparison metrics.

    Args:
        df: Input DataFrame.
        group_col: Column name for grouping.
        value_col: Column name for values to compare.
        groups: List of groups to compare. If None, use all groups.

    Returns:
        Tidy DataFrame with columns:
        group_a, group_b, ks_stat, ks_pvalue, wasserstein, js_divergence,
        median_diff, iqr_diff, n_a, n_b.
    """
    if groups is None:
        groups = df[group_col].unique().tolist()

    # Extract data for each group
    group_data: dict[str, np.ndarray] = {}
    for g in groups:
        values = df[df[group_col] == g][value_col].dropna().values
        group_data[g] = values

    results = []

    for group_a, group_b in combinations(groups, 2):
        a = group_data[group_a]
        b = group_data[group_b]

        if len(a) == 0 or len(b) == 0:
            continue

        ks_result = ks_test(a, b)
        ws = wasserstein(a, b)
        js = jensen_shannon_divergence(a, b)
        effect = effect_size_metrics(a, b)

        results.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "ks_stat": ks_result.statistic,
                "ks_pvalue": ks_result.pvalue,
                "wasserstein": ws,
                "js_divergence": js,
                "median_diff": effect.median_diff,
                "iqr_diff": effect.iqr_diff,
                "n_a": len(a),
                "n_b": len(b),
            }
        )

    return pd.DataFrame(results)


def compute_distance_matrix(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    metric: str = "wasserstein",
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute distance matrix across all groups.

    Args:
        df: Input DataFrame.
        group_col: Column name for grouping.
        value_col: Column name for values to compare.
        metric: Distance metric ("wasserstein" or "js_divergence").
        groups: List of groups. If None, use all groups.

    Returns:
        Square DataFrame with groups as both index and columns.
        Diagonal is 0, matrix is symmetric.
    """
    if groups is None:
        groups = sorted(df[group_col].unique().tolist())

    # Extract data for each group
    group_data: dict[str, np.ndarray] = {}
    for g in groups:
        values = df[df[group_col] == g][value_col].dropna().values
        group_data[g] = values

    n = len(groups)
    matrix = np.zeros((n, n))

    metric_fn = wasserstein if metric == "wasserstein" else jensen_shannon_divergence

    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i == j:
                matrix[i, j] = 0.0
            elif i < j:
                a, b = group_data[g1], group_data[g2]
                if len(a) > 0 and len(b) > 0:
                    dist = metric_fn(a, b)
                    matrix[i, j] = dist
                    matrix[j, i] = dist
                else:
                    matrix[i, j] = np.nan
                    matrix[j, i] = np.nan

    return pd.DataFrame(matrix, index=groups, columns=groups)


def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    datetime_col: str | None = None,
    groups: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    min_count: int = 0,
) -> DistributionComparisonReport:
    """Compare distributions across groups.

    Main entry point that computes all comparison metrics and returns
    a structured report.

    Args:
        df: Input DataFrame with at least group_col and value_col.
        group_col: Column name for grouping.
        value_col: Column name for values to compare.
        datetime_col: Optional datetime column for filtering.
        groups: List of groups to compare. If None, use all groups.
        start: Optional start datetime (inclusive) for filtering.
        end: Optional end datetime (inclusive) for filtering.
        min_count: Minimum sample count per group (drop smaller groups).

    Returns:
        DistributionComparisonReport with all comparison results.
    """
    # Apply datetime filters if specified
    filtered_df = df.copy()
    filters: dict[str, str] = {}

    if datetime_col and start:
        filtered_df = filtered_df[filtered_df[datetime_col] >= start]
        filters["start"] = start

    if datetime_col and end:
        filtered_df = filtered_df[filtered_df[datetime_col] <= end]
        filters["end"] = end

    # Determine groups to use
    if groups is None:
        groups = filtered_df[group_col].unique().tolist()

    # Filter by min_count
    if min_count > 0:
        counts = filtered_df.groupby(group_col)[value_col].count()
        valid_groups = counts[counts >= min_count].index.tolist()
        groups = [g for g in groups if g in valid_groups]

    # Compute summaries
    group_summaries = compute_group_summaries(filtered_df, group_col, value_col)
    group_summaries = group_summaries.loc[group_summaries.index.isin(groups)]

    # Compute pairwise stats
    pairwise_stats = compute_pairwise_stats(filtered_df, group_col, value_col, groups)

    # Compute distance matrices
    distance_matrices = {
        "wasserstein": compute_distance_matrix(
            filtered_df, group_col, value_col, "wasserstein", groups
        ),
        "js_divergence": compute_distance_matrix(
            filtered_df, group_col, value_col, "js_divergence", groups
        ),
    }

    return DistributionComparisonReport(
        group_summaries=group_summaries,
        pairwise_stats=pairwise_stats,
        distance_matrices=distance_matrices,
        value_col=value_col,
        group_col=group_col,
        filters=filters,
    )
