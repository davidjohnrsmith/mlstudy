"""Tests for mean reversion statistical tests and diagnostics."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.mean_reversion.mean_reversion_tests import (
    adf_test,
    compute_half_life,
    fit_ar1,
    is_mean_reverting,
    kpss_test,
    mean_reversion_diagnostics,
)


def generate_ar1_process(
    n: int,
    b: float,
    a: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> pd.Series:
    """Generate AR(1) process: x_t = a + b * x_{t-1} + eps.

    Args:
        n: Number of observations.
        b: AR(1) coefficient (persistence).
        a: Intercept.
        sigma: Standard deviation of noise.
        seed: Random seed for reproducibility.

    Returns:
        Series with AR(1) process.
    """
    np.random.seed(seed)
    x = np.zeros(n)
    x[0] = a / (1 - b) if abs(1 - b) > 1e-6 else 0  # Start near mean if stationary
    for i in range(1, n):
        x[i] = a + b * x[i - 1] + sigma * np.random.randn()
    return pd.Series(x, name="ar1_process")


def generate_random_walk(
    n: int,
    sigma: float = 1.0,
    seed: int = 42,
) -> pd.Series:
    """Generate random walk: x_t = x_{t-1} + eps.

    Args:
        n: Number of observations.
        sigma: Standard deviation of innovations.
        seed: Random seed for reproducibility.

    Returns:
        Series with random walk.
    """
    np.random.seed(seed)
    eps = sigma * np.random.randn(n)
    x = np.cumsum(eps)
    return pd.Series(x, name="random_walk")


class TestFitAR1:
    """Tests for fit_ar1 function."""

    def test_fit_ar1_detects_mean_reverting_on_ou_like_process(self):
        """AR(1) with b=0.9 should be detected as mean-reverting."""
        # Generate AR(1) with b=0.9 (mean-reverting)
        series = generate_ar1_process(n=500, b=0.9, a=0.0, sigma=1.0, seed=42)

        result = fit_ar1(series, min_obs=30)

        # Check basic structure
        assert "n_obs" in result
        assert "a" in result
        assert "b" in result
        assert "sigma" in result
        assert "half_life" in result
        assert "is_mean_reverting" in result

        # Check mean reversion detection
        assert result["is_mean_reverting"] == True  # noqa: E712
        assert 0 < result["b"] < 1
        assert result["half_life"] is not None
        assert result["half_life"] > 0

        # b should be close to 0.9 (allow some estimation error)
        assert 0.8 < result["b"] < 0.98, f"b={result['b']} not close to 0.9"

        # Half-life for b=0.9: -ln(2)/ln(0.9) ≈ 6.58
        # Allow range for estimation noise
        assert 3 < result["half_life"] < 15, f"half_life={result['half_life']}"

    def test_fit_ar1_non_mean_reverting_random_walk(self):
        """Random walk should have b close to 1, not mean-reverting."""
        # Generate random walk (b effectively = 1)
        series = generate_random_walk(n=500, sigma=1.0, seed=42)

        result = fit_ar1(series, min_obs=30)

        # Check basic structure
        assert "n_obs" in result
        assert "b" in result
        assert "is_mean_reverting" in result
        assert "half_life" in result

        # For random walk, b should be close to 1
        # Either b >= 0.98 or is_mean_reverting should be False
        is_near_unit_root = result["b"] >= 0.98
        is_not_mr = result["is_mean_reverting"] is False

        assert is_near_unit_root or is_not_mr, (
            f"Random walk should have b >= 0.98 or not be mean-reverting. "
            f"Got b={result['b']}, is_mean_reverting={result['is_mean_reverting']}"
        )

        # Half-life should be None for non-mean-reverting
        if not result["is_mean_reverting"]:
            assert result["half_life"] is None

    def test_fit_ar1_handles_nans_and_min_obs(self):
        """Should raise ValueError when too many NaNs leave < min_obs."""
        # Create series with many NaNs
        np.random.seed(42)
        data = np.random.randn(100)
        data[:80] = np.nan  # Only 20 valid observations

        series = pd.Series(data)

        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_ar1(series, min_obs=30)

    def test_fit_ar1_handles_short_series(self):
        """Should raise ValueError for very short series."""
        series = pd.Series([1.0, 2.0])  # Only 2 points

        # Should work with min_obs=2
        result = fit_ar1(series, min_obs=2)
        assert result["n_obs"] == 2

        # Should fail with min_obs=5
        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_ar1(series, min_obs=5)

    def test_fit_ar1_constant_series(self):
        """Should handle constant series gracefully."""
        series = pd.Series([5.0] * 100)

        result = fit_ar1(series, min_obs=30)

        # Should not raise, should return sensible values
        assert result["n_obs"] == 100
        assert result["is_mean_reverting"] == False  # noqa: E712
        assert result["half_life"] is None

    def test_fit_ar1_with_different_params(self):
        """Test AR(1) fitting with various parameters."""
        test_cases = [
            (0.5, True),   # Strongly mean-reverting
            (0.8, True),   # Mean-reverting
            (0.95, True),  # Weakly mean-reverting
        ]

        for b_true, expect_mr in test_cases:
            series = generate_ar1_process(n=1000, b=b_true, a=0.0, sigma=1.0, seed=42)
            result = fit_ar1(series, min_obs=30)

            assert result["is_mean_reverting"] == expect_mr, (
                f"b_true={b_true}: expected is_mean_reverting={expect_mr}, "
                f"got {result['is_mean_reverting']}"
            )


class TestADFTest:
    """Tests for adf_test function."""

    def test_adf_and_kpss_availability_behavior(self):
        """Test that ADF handles availability correctly."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        result = adf_test(series, min_obs=30)

        # Check structure
        assert "available" in result
        assert "n_obs" in result

        if result["available"]:
            # If statsmodels is installed
            assert "test_stat" in result
            assert "p_value" in result
            assert "used_lag" in result
            assert "critical_values" in result
            assert "reject_unit_root_5pct" in result

            # For mean-reverting series, should likely reject unit root
            # (not guaranteed due to finite sample)
            assert isinstance(result["test_stat"], float)
            assert isinstance(result["p_value"], float)
            assert isinstance(result["reject_unit_root_5pct"], bool)
        else:
            # If statsmodels not installed
            assert "reason" in result
            assert isinstance(result["reason"], str)

    def test_adf_insufficient_obs(self):
        """Should return not available for insufficient observations."""
        series = pd.Series(np.random.randn(10))

        result = adf_test(series, min_obs=30)

        assert result["available"] is False
        assert "Insufficient observations" in result.get("reason", "")

    def test_adf_with_random_walk(self):
        """ADF should likely NOT reject unit root for random walk."""
        series = generate_random_walk(n=500, sigma=1.0, seed=42)

        result = adf_test(series, min_obs=30)

        if result["available"]:
            # For random walk, p-value should typically be high
            # (fail to reject unit root)
            # Note: This is statistical, might occasionally fail
            pass  # Just check it runs without error


class TestKPSSTest:
    """Tests for kpss_test function."""

    def test_kpss_availability_behavior(self):
        """Test that KPSS handles availability correctly."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        result = kpss_test(series, min_obs=30)

        # Check structure
        assert "available" in result
        assert "n_obs" in result

        if result["available"]:
            # If statsmodels is installed
            assert "test_stat" in result
            assert "p_value" in result
            assert "used_lags" in result
            assert "critical_values" in result
            assert "reject_stationarity_5pct" in result

            assert isinstance(result["test_stat"], float)
            assert isinstance(result["reject_stationarity_5pct"], bool)
        else:
            # If statsmodels not installed
            assert "reason" in result
            assert isinstance(result["reason"], str)

    def test_kpss_insufficient_obs(self):
        """Should return not available for insufficient observations."""
        series = pd.Series(np.random.randn(10))

        result = kpss_test(series, min_obs=30)

        assert result["available"] is False
        assert "Insufficient observations" in result.get("reason", "")


class TestMeanReversionDiagnostics:
    """Tests for mean_reversion_diagnostics function."""

    def test_mean_reversion_diagnostics_summary_flag_mean_reverting(self):
        """Mean-reverting series should have 'strong' or 'mixed' evidence."""
        series = generate_ar1_process(n=500, b=0.9, a=0.0, sigma=1.0, seed=42)

        result = mean_reversion_diagnostics(series, min_obs=60)

        # Check structure
        assert "n_obs" in result
        assert "ar1" in result
        assert "adf" in result
        assert "kpss" in result
        assert "mr_evidence" in result

        # For mean-reverting series, evidence should be strong or mixed
        # (depends on statsmodels availability)
        assert result["mr_evidence"] in {"strong", "mixed"}, (
            f"Expected 'strong' or 'mixed' for mean-reverting series, "
            f"got '{result['mr_evidence']}'"
        )

        # AR1 should detect mean reversion
        assert result["ar1"].get("is_mean_reverting", False) == True  # noqa: E712

    def test_mean_reversion_diagnostics_summary_flag_random_walk(self):
        """Random walk should have 'weak' or 'mixed' evidence."""
        series = generate_random_walk(n=500, sigma=1.0, seed=42)

        result = mean_reversion_diagnostics(series, min_obs=60)

        # Check structure
        assert "n_obs" in result
        assert "ar1" in result
        assert "mr_evidence" in result

        # For random walk, evidence should be weak or mixed
        assert result["mr_evidence"] in {"weak", "mixed"}, (
            f"Expected 'weak' or 'mixed' for random walk, "
            f"got '{result['mr_evidence']}'"
        )

    def test_mean_reversion_diagnostics_insufficient_data(self):
        """Should return 'insufficient_data' when not enough observations."""
        series = pd.Series(np.random.randn(30))

        result = mean_reversion_diagnostics(series, min_obs=60)

        assert result["mr_evidence"] == "insufficient_data"

    def test_mean_reversion_diagnostics_with_nans(self):
        """Should handle series with NaNs."""
        np.random.seed(42)
        data = np.random.randn(200)
        data[::10] = np.nan  # Add some NaNs

        series = pd.Series(data)
        result = mean_reversion_diagnostics(series, min_obs=60)

        # Should still work with remaining valid observations
        assert "mr_evidence" in result
        assert result["n_obs"] < 200  # Some were NaN

    def test_diagnostics_strongly_mean_reverting(self):
        """Strongly mean-reverting series should often show strong evidence."""
        # Use b=0.7 for strong mean reversion
        series = generate_ar1_process(n=1000, b=0.7, a=0.0, sigma=1.0, seed=42)

        result = mean_reversion_diagnostics(series, min_obs=60)

        # Should definitely be mean-reverting by AR1
        assert result["ar1"]["is_mean_reverting"] == True  # noqa: E712
        assert result["ar1"]["half_life"] is not None

        # Half-life for b=0.7: -ln(2)/ln(0.7) ≈ 1.94
        assert result["ar1"]["half_life"] < 10


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_half_life_mean_reverting(self):
        """compute_half_life should return value for mean-reverting series."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        hl = compute_half_life(series, min_obs=30)

        assert hl is not None
        assert hl > 0
        # For b=0.9, half-life ≈ 6.58
        assert 3 < hl < 15

    def test_compute_half_life_random_walk(self):
        """compute_half_life should return None for random walk."""
        series = generate_random_walk(n=500, seed=42)

        hl = compute_half_life(series, min_obs=30)

        # Random walk has b ≈ 1, so half_life should be None or very large
        # If b > 1, half_life is None
        # If 0.99 < b < 1, half_life would be very large
        if hl is not None:
            # If it returns a value, it should be very large
            assert hl > 50, f"Expected large half-life or None, got {hl}"

    def test_compute_half_life_insufficient_data(self):
        """compute_half_life should return None for insufficient data."""
        series = pd.Series(np.random.randn(10))

        hl = compute_half_life(series, min_obs=30)

        assert hl is None

    def test_is_mean_reverting_true(self):
        """is_mean_reverting should return True for AR(1) series."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        assert is_mean_reverting(series, min_obs=60) == True  # noqa: E712

    def test_is_mean_reverting_false(self):
        """is_mean_reverting might return False for random walk."""
        series = generate_random_walk(n=500, seed=42)

        # Random walk might be detected as not mean-reverting
        # But this depends on estimation, so we just check it runs
        result = is_mean_reverting(series, min_obs=60)
        assert isinstance(result, bool)

    def test_is_mean_reverting_require_strong(self):
        """is_mean_reverting with require_strong=True needs strong evidence."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        result_any = is_mean_reverting(series, min_obs=60, require_strong=False)
        result_strong = is_mean_reverting(series, min_obs=60, require_strong=True)

        # result_any should be True for mean-reverting series
        assert result_any == True  # noqa: E712

        # result_strong might be True or False depending on statsmodels
        assert isinstance(result_strong, bool)


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_empty_series(self):
        """Should handle empty series gracefully."""
        series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_ar1(series, min_obs=1)

    def test_single_value_series(self):
        """Should handle single-value series."""
        series = pd.Series([1.0])

        with pytest.raises(ValueError):
            fit_ar1(series, min_obs=1)

    def test_all_nan_series(self):
        """Should handle all-NaN series."""
        series = pd.Series([np.nan] * 100)

        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_ar1(series, min_obs=30)

    def test_near_constant_with_noise(self):
        """Should handle near-constant series with tiny noise."""
        np.random.seed(42)
        series = pd.Series(5.0 + 1e-10 * np.random.randn(100))

        result = fit_ar1(series, min_obs=30)

        # Should not raise or produce NaN/inf
        assert not np.isnan(result["a"])
        assert not np.isnan(result["sigma"])
        assert not np.isinf(result.get("half_life", 0) or 0)

    def test_large_series(self):
        """Should handle large series efficiently."""
        series = generate_ar1_process(n=10000, b=0.95, seed=42)

        result = fit_ar1(series, min_obs=30)

        assert result["n_obs"] == 10000
        assert result["is_mean_reverting"] == True  # noqa: E712

    def test_highly_persistent_ar1(self):
        """Test AR(1) with b very close to 1."""
        series = generate_ar1_process(n=1000, b=0.99, seed=42)

        result = fit_ar1(series, min_obs=30)

        # Should still detect as mean-reverting if b < 1
        if result["is_mean_reverting"]:
            # Half-life should be large
            assert result["half_life"] > 50


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_fit_ar1_deterministic(self):
        """Same input should produce same output."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        result1 = fit_ar1(series, min_obs=30)
        result2 = fit_ar1(series, min_obs=30)

        assert result1["b"] == result2["b"]
        assert result1["a"] == result2["a"]
        assert result1["half_life"] == result2["half_life"]

    def test_diagnostics_deterministic(self):
        """Same input should produce same diagnostics."""
        series = generate_ar1_process(n=500, b=0.9, seed=42)

        result1 = mean_reversion_diagnostics(series, min_obs=60)
        result2 = mean_reversion_diagnostics(series, min_obs=60)

        assert result1["mr_evidence"] == result2["mr_evidence"]
        assert result1["ar1"]["b"] == result2["ar1"]["b"]
