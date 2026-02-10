"""Tests for distribution comparison module."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.research.analysis.distributions.grouped import (
    compare_groups,
    compute_distance_matrix,
    compute_group_summaries,
    compute_pairwise_stats,

)
from mlstudy.research.analysis.distributions.metrics import (

    effect_size_metrics,
    jensen_shannon_divergence,
    ks_test,
    wasserstein,
)
# Fixed random seed for deterministic tests
RNG = np.random.default_rng(42)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create synthetic DataFrame with known distribution properties.

    Groups:
    - "identical_1" and "identical_2": Same distribution (N(0, 1))
    - "shifted": Shifted distribution (N(2, 1))
    - "different_var": Different variance (N(0, 3))
    """
    n = 500

    data = {
        "datetime": pd.date_range("2024-01-01", periods=n * 4, freq="h"),
        "group": (
            ["identical_1"] * n + ["identical_2"] * n + ["shifted"] * n + ["different_var"] * n
        ),
        "value": np.concatenate(
            [
                RNG.normal(0, 1, n),  # identical_1
                RNG.normal(0, 1, n),  # identical_2
                RNG.normal(2, 1, n),  # shifted mean
                RNG.normal(0, 3, n),  # larger variance
            ]
        ),
    }
    return pd.DataFrame(data)


class TestMetrics:
    """Tests for individual metric functions."""

    def test_ks_test_identical(self) -> None:
        """KS test on identical distributions should have high p-value."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(0, 1, 1000)
        result = ks_test(a, b)
        assert result.pvalue > 0.05

    def test_ks_test_different(self) -> None:
        """KS test on different distributions should have low p-value."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(2, 1, 1000)
        result = ks_test(a, b)
        assert result.pvalue < 0.01
        assert result.statistic > 0.3

    def test_wasserstein_identical(self) -> None:
        """Wasserstein distance on identical distributions should be small."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(0, 1, 1000)
        dist = wasserstein(a, b)
        assert dist < 0.1

    def test_wasserstein_shifted(self) -> None:
        """Wasserstein distance on shifted distributions should reflect shift."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(2, 1, 1000)
        dist = wasserstein(a, b)
        # Should be close to the mean shift (2)
        assert 1.5 < dist < 2.5

    def test_js_divergence_identical(self) -> None:
        """JS divergence on identical distributions should be small."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(0, 1, 1000)
        js = jensen_shannon_divergence(a, b)
        assert js < 0.05

    def test_js_divergence_different(self) -> None:
        """JS divergence on different distributions should be larger."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(5, 1, 1000)
        js = jensen_shannon_divergence(a, b)
        assert js > 0.3

    def test_js_divergence_bounded(self) -> None:
        """JS divergence should be in [0, 1]."""
        a = RNG.normal(0, 1, 100)
        b = RNG.normal(100, 1, 100)
        js = jensen_shannon_divergence(a, b)
        assert 0 <= js <= 1

    def test_effect_size_metrics(self) -> None:
        """Effect size metrics should capture median and IQR differences."""
        a = RNG.normal(0, 1, 1000)
        b = RNG.normal(2, 2, 1000)
        result = effect_size_metrics(a, b)
        # Median diff should be around -2
        assert -2.5 < result.median_diff < -1.5
        # IQR of b is larger, so iqr_diff should be negative
        assert result.iqr_diff < 0


class TestGroupSummaries:
    """Tests for group summary statistics."""

    def test_summary_columns(self, synthetic_df: pd.DataFrame) -> None:
        """Summary should have expected columns."""
        summary = compute_group_summaries(synthetic_df, "group", "value")
        expected_cols = [
            "count",
            "missing",
            "mean",
            "std",
            "min",
            "max",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "skew",
            "kurtosis",
        ]
        for col in expected_cols:
            assert col in summary.columns

    def test_summary_counts(self, synthetic_df: pd.DataFrame) -> None:
        """Summary should have correct counts."""
        summary = compute_group_summaries(synthetic_df, "group", "value")
        assert summary.loc["identical_1", "count"] == 500
        assert summary.loc["shifted", "count"] == 500

    def test_summary_means(self, synthetic_df: pd.DataFrame) -> None:
        """Summary means should reflect distribution parameters."""
        summary = compute_group_summaries(synthetic_df, "group", "value")
        # identical_1 and identical_2 should have mean near 0
        assert abs(summary.loc["identical_1", "mean"]) < 0.2
        # shifted should have mean near 2
        assert 1.8 < summary.loc["shifted", "mean"] < 2.2


class TestPairwiseStats:
    """Tests for pairwise comparison statistics."""

    def test_pairwise_columns(self, synthetic_df: pd.DataFrame) -> None:
        """Pairwise stats should have expected columns."""
        pairwise = compute_pairwise_stats(synthetic_df, "group", "value")
        expected_cols = [
            "group_a",
            "group_b",
            "ks_stat",
            "ks_pvalue",
            "wasserstein",
            "js_divergence",
            "median_diff",
            "iqr_diff",
            "n_a",
            "n_b",
        ]
        for col in expected_cols:
            assert col in pairwise.columns

    def test_identical_groups_high_pvalue(self, synthetic_df: pd.DataFrame) -> None:
        """Identical distributions should have high KS p-value."""
        pairwise = compute_pairwise_stats(synthetic_df, "group", "value")
        identical_row = pairwise[
            ((pairwise["group_a"] == "identical_1") & (pairwise["group_b"] == "identical_2"))
            | ((pairwise["group_a"] == "identical_2") & (pairwise["group_b"] == "identical_1"))
        ].iloc[0]
        assert identical_row["ks_pvalue"] > 0.05

    def test_shifted_groups_low_pvalue(self, synthetic_df: pd.DataFrame) -> None:
        """Shifted distributions should have low KS p-value."""
        pairwise = compute_pairwise_stats(synthetic_df, "group", "value")
        shifted_row = pairwise[
            ((pairwise["group_a"] == "identical_1") & (pairwise["group_b"] == "shifted"))
            | ((pairwise["group_a"] == "shifted") & (pairwise["group_b"] == "identical_1"))
        ].iloc[0]
        assert shifted_row["ks_pvalue"] < 0.01
        assert shifted_row["wasserstein"] > 1.5

    def test_different_variance_larger_distances(self, synthetic_df: pd.DataFrame) -> None:
        """Different variance should result in larger distances."""
        pairwise = compute_pairwise_stats(synthetic_df, "group", "value")
        diff_var_row = pairwise[
            ((pairwise["group_a"] == "identical_1") & (pairwise["group_b"] == "different_var"))
            | ((pairwise["group_a"] == "different_var") & (pairwise["group_b"] == "identical_1"))
        ].iloc[0]
        assert diff_var_row["ks_pvalue"] < 0.05
        assert diff_var_row["js_divergence"] > 0.01


class TestDistanceMatrix:
    """Tests for distance matrix computation."""

    def test_matrix_is_square(self, synthetic_df: pd.DataFrame) -> None:
        """Distance matrix should be square."""
        matrix = compute_distance_matrix(synthetic_df, "group", "value")
        assert matrix.shape[0] == matrix.shape[1]

    def test_matrix_diagonal_zero(self, synthetic_df: pd.DataFrame) -> None:
        """Diagonal elements should be zero."""
        matrix = compute_distance_matrix(synthetic_df, "group", "value")
        for i in range(len(matrix)):
            assert matrix.iloc[i, i] == 0.0

    def test_matrix_symmetric(self, synthetic_df: pd.DataFrame) -> None:
        """Matrix should be symmetric."""
        matrix = compute_distance_matrix(synthetic_df, "group", "value")
        pd.testing.assert_frame_equal(matrix, matrix.T)


class TestCompareGroups:
    """Tests for main compare_groups function."""

    def test_report_structure(self, synthetic_df: pd.DataFrame) -> None:
        """Report should have all expected components."""
        report = compare_groups(synthetic_df, "group", "value")
        assert report.group_summaries is not None
        assert report.pairwise_stats is not None
        assert "wasserstein" in report.distance_matrices
        assert "js_divergence" in report.distance_matrices
        assert report.group_col == "group"
        assert report.value_col == "value"

    def test_min_count_filter(self, synthetic_df: pd.DataFrame) -> None:
        """Groups below min_count should be excluded."""
        # Add a small group
        small_group = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=10, freq="h"),
                "group": ["small"] * 10,
                "value": RNG.normal(0, 1, 10),
            }
        )
        df_with_small = pd.concat([synthetic_df, small_group], ignore_index=True)

        report = compare_groups(df_with_small, "group", "value", min_count=100)
        assert "small" not in report.group_summaries.index

    def test_datetime_filter(self, synthetic_df: pd.DataFrame) -> None:
        """Datetime filters should be applied correctly."""
        report = compare_groups(
            synthetic_df,
            "group",
            "value",
            datetime_col="datetime",
            start="2024-01-02",
            end="2024-01-10",
        )
        assert "start" in report.filters
        assert "end" in report.filters

    def test_specific_groups(self, synthetic_df: pd.DataFrame) -> None:
        """Only specified groups should be compared."""
        report = compare_groups(
            synthetic_df,
            "group",
            "value",
            groups=["identical_1", "shifted"],
        )
        assert len(report.group_summaries) == 2
        assert "identical_1" in report.group_summaries.index
        assert "shifted" in report.group_summaries.index
        assert "identical_2" not in report.group_summaries.index
