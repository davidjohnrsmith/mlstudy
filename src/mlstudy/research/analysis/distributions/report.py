"""Distribution comparison report dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DistributionComparisonReport:
    """Container for distribution comparison results.

    Attributes:
        group_summaries: DataFrame with summary statistics per group.
            Index: group names.
            Columns: count, missing, mean, std, min, max, q05, q25, q50, q75, q95,
                     skew, kurtosis.
        pairwise_stats: Tidy DataFrame with pairwise comparison metrics.
            Columns: group_a, group_b, ks_stat, ks_pvalue, wasserstein,
                     js_divergence, median_diff, iqr_diff, n_a, n_b.
        distance_matrices: Dict mapping metric name to square DataFrame.
            Keys: "wasserstein", "js_divergence".
            Values: DataFrames with groups as both index and columns.
        value_col: Name of the value column being analyzed.
        group_col: Name of the grouping column.
        filters: Optional dict of filters applied (e.g., start/end dates).
    """

    group_summaries: pd.DataFrame
    pairwise_stats: pd.DataFrame
    distance_matrices: dict[str, pd.DataFrame] = field(default_factory=dict)
    value_col: str = ""
    group_col: str = ""
    filters: dict[str, str] = field(default_factory=dict)

    def to_csv(self, outdir: str) -> dict[str, str]:
        """Write report components to CSV files.

        Args:
            outdir: Output directory path.

        Returns:
            Dict mapping component name to file path.
        """
        from pathlib import Path

        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)

        paths = {}

        summary_path = outpath / "group_summaries.csv"
        self.group_summaries.to_csv(summary_path)
        paths["group_summaries"] = str(summary_path)

        pairwise_path = outpath / "pairwise_stats.csv"
        self.pairwise_stats.to_csv(pairwise_path, index=False)
        paths["pairwise_stats"] = str(pairwise_path)

        for name, df in self.distance_matrices.items():
            matrix_path = outpath / f"distance_matrix_{name}.csv"
            df.to_csv(matrix_path)
            paths[f"distance_matrix_{name}"] = str(matrix_path)

        return paths
