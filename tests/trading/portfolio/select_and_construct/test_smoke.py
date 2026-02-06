"""Smoke tests for select_and_construct module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# -----------------------------
# Test fixtures
# -----------------------------
@pytest.fixture
def sample_returns():
    """Generate sample return series for testing."""
    idx = pd.date_range("2022-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)

    returns = {
        "strat_A": pd.Series(0.0002 + 0.010 * rng.standard_normal(len(idx)), index=idx),
        "strat_B": pd.Series(0.0001 + 0.011 * rng.standard_normal(len(idx)), index=idx),
        "strat_C": pd.Series(0.00015 + 0.009 * rng.standard_normal(len(idx)), index=idx),
    }
    # Create a highly correlated copy
    returns["strat_A2"] = returns["strat_A"] * 0.9 + 0.001 * rng.standard_normal(len(idx))
    return returns


@pytest.fixture
def panel_with_gaps():
    """Generate returns panel with non-synchronous data (gaps)."""
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    rng = np.random.default_rng(123)

    # Full history
    fly_123 = pd.Series(0.0001 + 0.01 * rng.standard_normal(len(idx)), index=idx)

    # Ends early
    fly_236 = pd.Series(0.0001 + 0.01 * rng.standard_normal(len(idx)), index=idx)
    fly_236.iloc[100:] = np.nan

    # Starts late
    fly_234 = pd.Series(np.nan, index=idx)
    fly_234.iloc[80:] = 0.0001 + 0.01 * rng.standard_normal(len(idx) - 80)

    return {
        "fly_123": fly_123,
        "fly_236": fly_236,
        "fly_234": fly_234,
    }


# -----------------------------
# Metrics tests
# -----------------------------
class TestMetrics:
    def test_sharpe(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import sharpe

        r = sample_returns["strat_A"]
        sr = sharpe(r, ann_factor=252)
        assert np.isfinite(sr)
        # Positive mean strategy should have positive Sharpe
        assert sr > 0

    def test_max_drawdown(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import max_drawdown

        r = sample_returns["strat_A"]
        mdd = max_drawdown(r)
        assert np.isfinite(mdd)
        assert mdd <= 0  # MDD is always non-positive

    def test_newey_west_tstat(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import newey_west_tstat

        r = sample_returns["strat_A"]
        nw = newey_west_tstat(r, lags=5)
        assert np.isfinite(nw)

    def test_worst_rolling_return(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import worst_rolling_return

        r = sample_returns["strat_A"]
        worst = worst_rolling_return(r, window=21)
        assert np.isfinite(worst)
        assert worst <= 0  # Worst return is typically negative


# -----------------------------
# Correlation tests
# -----------------------------
class TestCorrelation:
    def test_corr_matrix(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import corr_matrix

        df = pd.concat(sample_returns, axis=1).dropna()
        corr = corr_matrix(df, method="pearson")

        assert corr.shape == (4, 4)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-10)
        # A and A2 should be highly correlated
        assert corr.loc["strat_A", "strat_A2"] > 0.8

    def test_pairwise_overlap_corr(self, panel_with_gaps):
        from mlstudy.trading.portfolio.select_and_construct import (
            build_returns_panel,
            pairwise_overlap_corr,
        )

        panel = build_returns_panel(panel_with_gaps)
        corr, nobs = pairwise_overlap_corr(panel, min_obs=30)

        assert corr.shape == (3, 3)
        # Diagonal should be 1
        for col in corr.columns:
            assert corr.loc[col, col] == 1.0 or np.isnan(corr.loc[col, col])
        # nobs should reflect actual overlap
        assert nobs.loc["fly_123", "fly_123"] == panel["fly_123"].notna().sum()

    def test_estimate_cov(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import estimate_cov

        df = pd.concat(sample_returns, axis=1).dropna()

        for method in ["sample", "ewma", "diag_shrink"]:
            cov = estimate_cov(df, method=method)
            assert cov.shape == (4, 4)
            # Should be symmetric
            np.testing.assert_allclose(cov, cov.T, atol=1e-10)
            # Diagonal should be positive
            assert np.all(np.diag(cov) > 0)

    def test_make_psd(self):
        from mlstudy.trading.portfolio.select_and_construct import make_psd

        # Create a non-PSD matrix
        A = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
        A[0, 0] = 0.5  # Make it potentially non-PSD

        result = make_psd(A)
        # Check that result is PSD (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues >= 0)


# -----------------------------
# Clustering tests
# -----------------------------
class TestClustering:
    def test_correlation_clusters(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import (
            corr_matrix,
            correlation_clusters,
        )

        df = pd.concat(sample_returns, axis=1).dropna()
        corr = corr_matrix(df)
        clusters = correlation_clusters(corr, threshold=0.8)

        # A and A2 should be in the same cluster
        for cid, names in clusters.items():
            if "strat_A" in names:
                assert "strat_A2" in names
                break


# -----------------------------
# Weights tests
# -----------------------------
class TestWeights:
    def test_equal_weights(self):
        from mlstudy.trading.portfolio.select_and_construct import equal_weights

        w = equal_weights(5)
        np.testing.assert_allclose(w, np.ones(5) / 5)
        np.testing.assert_allclose(np.sum(np.abs(w)), 1.0)

    def test_risk_parity_weights(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import (
            estimate_cov,
            risk_parity_weights,
            risk_contributions,
        )

        df = pd.concat(sample_returns, axis=1).dropna()
        cov = estimate_cov(df, method="sample")
        w = risk_parity_weights(cov, gross_limit=1.0)

        # Check normalization
        np.testing.assert_allclose(np.sum(np.abs(w)), 1.0, atol=1e-6)

        # Check that risk contributions are roughly equal
        rc = risk_contributions(w, cov)
        rc_share = rc / rc.sum()
        expected = 1.0 / len(w)
        np.testing.assert_allclose(rc_share, expected, atol=0.05)

    def test_mean_variance_weights(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import (
            estimate_cov,
            mean_variance_weights,
        )

        df = pd.concat(sample_returns, axis=1).dropna()
        cov = estimate_cov(df, method="sample")
        mu = np.array([0.5, 0.3, 0.4, 0.2])

        w = mean_variance_weights(cov, mu, gross_limit=1.0)

        # Check normalization
        np.testing.assert_allclose(np.sum(np.abs(w)), 1.0, atol=1e-6)


# -----------------------------
# Selector tests
# -----------------------------
class TestSelector:
    def test_static_selection(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import (
            StrategySelector,
            StandaloneGates,
            RedundancyGates,
        )

        selector = StrategySelector(sample_returns, ann_factor=252)
        report = selector.run(
            standalone=StandaloneGates(
                min_obs=100,
                min_sharpe=None,  # Disable for test
                min_nw_tstat=None,
                max_mdd=None,
                max_worst_period=None,
                min_positive_year_frac=None,
            ),
            redundancy=RedundancyGates(
                enabled=True,
                corr_threshold=0.8,
                keep_per_cluster=1,
            ),
        )

        # All should pass standalone gates
        assert len(report.passed_names) == 4

        # After redundancy reduction, A and A2 should collapse
        assert len(report.selected_names) < len(report.passed_names)

    def test_timevarying_selection(self, panel_with_gaps):
        from mlstudy.trading.portfolio.select_and_construct import (
            build_returns_panel,
            StrategySelector,
            UniverseConfig,
            StandaloneGates,
            RedundancyGates,
        )

        panel = build_returns_panel(panel_with_gaps)
        selector = StrategySelector.from_panel(panel, ann_factor=252)

        # Select at a date where all three have some history
        t = panel.index[150]  # After fly_234 starts
        selected, metrics, clusters = selector.select_at(
            t,
            universe_cfg=UniverseConfig(lookback=60, require_active_today=True),
            standalone=StandaloneGates(min_obs=30, min_sharpe=None, min_nw_tstat=None,
                                        max_mdd=None, max_worst_period=None,
                                        min_positive_year_frac=None),
            redundancy=RedundancyGates(enabled=False),
        )

        # fly_123 and fly_234 should be active, fly_236 ends at 100
        assert "fly_123" in selected
        assert "fly_234" in selected
        assert "fly_236" not in selected  # Not active at t


# -----------------------------
# Constructor tests
# -----------------------------
class TestConstructor:
    def test_static_construction(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import (
            PortfolioConstructor,
        )

        constructor = PortfolioConstructor.from_series(sample_returns)
        weights, returns_df, cov = constructor.construct(
            weight_method="risk_parity",
            cov_method="diag_shrink",
        )

        assert len(weights) == 4
        np.testing.assert_allclose(np.sum(np.abs(weights)), 1.0, atol=1e-6)

    def test_static_backtest(self, sample_returns):
        from mlstudy.trading.portfolio.select_and_construct import (
            PortfolioConstructor,
        )

        constructor = PortfolioConstructor.from_series(sample_returns)
        weights, returns_df, cov = constructor.construct(weight_method="equal")
        port_ret = constructor.backtest(returns_df, weights)

        assert isinstance(port_ret, pd.Series)
        assert len(port_ret) == len(returns_df)

    def test_dynamic_backtest(self, panel_with_gaps):
        from mlstudy.trading.portfolio.select_and_construct import (
            build_returns_panel,
            StrategySelector,
            PortfolioConstructor,
            UniverseConfig,
            StandaloneGates,
            RedundancyGates,
        )

        panel = build_returns_panel(panel_with_gaps)
        selector = StrategySelector.from_panel(panel, ann_factor=252)
        constructor = PortfolioConstructor.from_panel(panel)

        port_ret, weights_hist = constructor.run_dynamic_backtest(
            selector=selector,
            universe_cfg=UniverseConfig(lookback=50, rebalance_freq="W-FRI"),
            standalone=StandaloneGates(min_obs=30, min_sharpe=None, min_nw_tstat=None,
                                        max_mdd=None, max_worst_period=None,
                                        min_positive_year_frac=None),
            redundancy=RedundancyGates(enabled=False),
            weight_method="equal",
        )

        assert isinstance(port_ret, pd.Series)
        assert isinstance(weights_hist, pd.DataFrame)
        # Should have some rebalance dates
        assert len(weights_hist) > 0


# -----------------------------
# Integration test
# -----------------------------
class TestIntegration:
    def test_full_workflow_static(self, sample_returns):
        """Test full workflow: select -> construct -> backtest."""
        from mlstudy.trading.portfolio.select_and_construct import (
            StrategySelector,
            PortfolioConstructor,
            StandaloneGates,
            RedundancyGates,
        )

        # 1. Select strategies
        selector = StrategySelector(sample_returns, ann_factor=252)
        report = selector.run(
            standalone=StandaloneGates(min_obs=100, min_sharpe=None, min_nw_tstat=None,
                                        max_mdd=None, max_worst_period=None,
                                        min_positive_year_frac=None),
            redundancy=RedundancyGates(enabled=True, corr_threshold=0.8, keep_per_cluster=1),
        )

        # 2. Construct portfolio with selected strategies
        selected_returns = {name: sample_returns[name] for name in report.selected_names}
        constructor = PortfolioConstructor.from_series(selected_returns)
        weights, returns_df, cov = constructor.construct(weight_method="risk_parity")

        # 3. Backtest
        port_ret = constructor.backtest(returns_df, weights)

        # Verify results
        assert len(report.selected_names) > 0
        assert len(weights) == len(report.selected_names)
        assert len(port_ret) > 0
        # Portfolio should have non-zero returns
        assert port_ret.std() > 0

    def test_full_workflow_dynamic(self, panel_with_gaps):
        """Test full workflow with time-varying universe."""
        from mlstudy.trading.portfolio.select_and_construct import (
            build_returns_panel,
            StrategySelector,
            PortfolioConstructor,
            UniverseConfig,
            StandaloneGates,
            RedundancyGates,
        )

        panel = build_returns_panel(panel_with_gaps)
        selector = StrategySelector.from_panel(panel, ann_factor=252)
        constructor = PortfolioConstructor.from_panel(panel)

        port_ret, weights_hist = constructor.run_dynamic_backtest(
            selector=selector,
            universe_cfg=UniverseConfig(lookback=50, rebalance_freq="W-FRI"),
            standalone=StandaloneGates(min_obs=30, min_sharpe=None, min_nw_tstat=None,
                                        max_mdd=None, max_worst_period=None,
                                        min_positive_year_frac=None),
            redundancy=RedundancyGates(enabled=False),
            weight_method="risk_parity",
        )

        # Verify results
        assert len(port_ret) == len(panel)
        assert len(weights_hist) > 0
        # Portfolio should have non-zero returns
        assert port_ret.dropna().std() > 0
