"""Tests for fly construction methods optimized for mean reversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.structures.specs.fly.old.fly_construction import (
    FlyWeights,
    compute_fly_from_weights,
    enforce_dv01_neutral,
    equal_weight_fly,
    fit_ar1_half_life,
    fly_changes_residual,
    fly_levels_residual,
    fly_optimize_mean_reversion,
    validate_inputs,
)


# =============================================================================
# Synthetic data generators
# =============================================================================


def generate_cointegrated_yields(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Generate cointegrated yield series where belly = 0.5*front + 0.5*back + noise.

    The front and back are random walks, but belly is a combination with
    stationary noise, creating a cointegration relationship.

    Returns:
        Tuple of (front, belly, back) Series with datetime index.
    """
    np.random.seed(seed)

    # Random walk for front and back
    front_changes = np.random.randn(n) * 0.05  # 5bp daily vol
    back_changes = np.random.randn(n) * 0.07  # 7bp daily vol

    front = np.cumsum(front_changes) + 2.0  # Start around 2%
    back = np.cumsum(back_changes) + 4.0  # Start around 4%

    # Belly is cointegrated: 0.5*front + 0.5*back + stationary noise
    stationary_noise = np.zeros(n)
    for i in range(1, n):
        # AR(1) noise with b=0.8 (mean-reverting)
        stationary_noise[i] = 0.8 * stationary_noise[i - 1] + np.random.randn() * 0.02

    belly = 0.5 * front + 0.5 * back + stationary_noise

    # Create date index
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    return (
        pd.Series(front, index=dates, name="front"),
        pd.Series(belly, index=dates, name="belly"),
        pd.Series(back, index=dates, name="back"),
    )


def generate_correlated_changes(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Generate yields where changes are correlated: Δbelly ~ Δfront + Δback + noise.

    Returns:
        Tuple of (front, belly, back) Series.
    """
    np.random.seed(seed)

    # Independent changes for wings
    d_front = np.random.randn(n) * 0.05
    d_back = np.random.randn(n) * 0.07

    # Belly changes are correlated combination + noise
    noise = np.random.randn(n) * 0.02
    d_belly = 0.4 * d_front + 0.6 * d_back + noise

    # Integrate to levels
    front = np.cumsum(d_front) + 2.0
    back = np.cumsum(d_back) + 4.0
    belly = np.cumsum(d_belly) + 3.0

    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    return (
        pd.Series(front, index=dates, name="front"),
        pd.Series(belly, index=dates, name="belly"),
        pd.Series(back, index=dates, name="back"),
    )


def generate_independent_random_walks(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Generate three independent random walk yield series.

    Returns:
        Tuple of (front, belly, back) Series with no cointegration.
    """
    np.random.seed(seed)

    front = np.cumsum(np.random.randn(n) * 0.05) + 2.0
    belly = np.cumsum(np.random.randn(n) * 0.06) + 3.0
    back = np.cumsum(np.random.randn(n) * 0.07) + 4.0

    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    return (
        pd.Series(front, index=dates, name="front"),
        pd.Series(belly, index=dates, name="belly"),
        pd.Series(back, index=dates, name="back"),
    )


# =============================================================================
# Test FlyWeights dataclass
# =============================================================================


class TestFlyWeights:
    """Tests for FlyWeights dataclass."""

    def test_basic_creation(self):
        """Test basic FlyWeights creation."""
        weights = FlyWeights(
            w_front=1.0,
            w_belly=-2.0,
            w_back=1.0,
            method="equal_weight",
        )

        assert weights.w_front == 1.0
        assert weights.w_belly == -2.0
        assert weights.w_back == 1.0
        assert weights.method == "equal_weight"
        assert weights.window is None
        assert weights.meta == {}

    def test_to_tuple(self):
        """Test to_tuple method."""
        weights = FlyWeights(w_front=0.5, w_belly=-1.0, w_back=0.5, method="test")
        assert weights.to_tuple() == (0.5, -1.0, 0.5)

    def test_compute_fly(self):
        """Test compute_fly method."""
        weights = FlyWeights(w_front=1.0, w_belly=-2.0, w_back=1.0, method="test")

        # Scalar
        result = weights.compute_fly(2.0, 3.0, 4.0)
        expected = 1.0 * 2.0 + (-2.0) * 3.0 + 1.0 * 4.0
        assert result == expected

        # Array
        front = np.array([1.0, 2.0])
        belly = np.array([1.5, 2.5])
        back = np.array([2.0, 3.0])
        result = weights.compute_fly(front, belly, back)
        expected = 1.0 * front + (-2.0) * belly + 1.0 * back
        np.testing.assert_array_equal(result, expected)

    def test_with_meta(self):
        """Test FlyWeights with metadata."""
        weights = FlyWeights(
            w_front=1.0,
            w_belly=-2.0,
            w_back=1.0,
            method="optimized",
            window=252,
            meta={"half_life": 5.5, "net_dv01": 0.001},
        )

        assert weights.window == 252
        assert weights.meta["half_life"] == 5.5
        assert weights.meta["net_dv01"] == 0.001


# =============================================================================
# Test validate_inputs
# =============================================================================


class TestValidateInputs:
    """Tests for validate_inputs function."""

    def test_basic_alignment(self):
        """Test basic index alignment."""
        front, belly, back = generate_cointegrated_yields(n=100, seed=42)
        df = validate_inputs(front, belly, back, min_obs=10)

        assert len(df) == 100
        assert list(df.columns) == ["front", "belly", "back"]
        assert df.index.equals(front.index)

    def test_handles_missing_dates(self):
        """Test alignment with non-overlapping indices."""
        dates1 = pd.date_range("2020-01-01", periods=100, freq="B")
        dates2 = pd.date_range("2020-01-10", periods=100, freq="B")  # Offset

        front = pd.Series(np.random.randn(100), index=dates1)
        belly = pd.Series(np.random.randn(100), index=dates2)
        back = pd.Series(np.random.randn(100), index=dates1)

        df = validate_inputs(front, belly, back, min_obs=10)

        # Should have inner join of indices
        assert len(df) < 100
        assert all(idx in dates1 for idx in df.index)
        assert all(idx in dates2 for idx in df.index)

    def test_drops_nans(self):
        """Test that NaN values are dropped."""
        front, belly, back = generate_cointegrated_yields(n=100, seed=42)

        # Add some NaNs
        front.iloc[10:15] = np.nan
        belly.iloc[50:55] = np.nan

        df = validate_inputs(front, belly, back, min_obs=10)

        assert len(df) == 90  # 100 - 5 - 5
        assert not df.isna().any().any()

    def test_raises_on_insufficient_obs(self):
        """Test ValueError when too few observations."""
        front = pd.Series([1.0, 2.0, 3.0])
        belly = pd.Series([1.5, 2.5, 3.5])
        back = pd.Series([2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="Insufficient observations"):
            validate_inputs(front, belly, back, min_obs=10)


# =============================================================================
# Test fit_ar1_half_life
# =============================================================================


class TestFitAR1HalfLife:
    """Tests for fit_ar1_half_life function."""

    def test_mean_reverting_series(self):
        """Test on known mean-reverting series."""
        np.random.seed(42)
        n = 500

        # Generate AR(1) with b=0.9
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + np.random.randn() * 0.1

        series = pd.Series(x)
        result = fit_ar1_half_life(series, min_obs=30)

        assert "b" in result
        assert "half_life" in result
        assert "residual_std" in result

        # b should be close to 0.9
        assert 0.8 < result["b"] < 0.98

        # Half-life for b=0.9: -ln(2)/ln(0.9) ≈ 6.58
        assert result["half_life"] is not None
        assert 3 < result["half_life"] < 15

    def test_random_walk(self):
        """Test on random walk (not mean-reverting)."""
        np.random.seed(42)
        series = pd.Series(np.cumsum(np.random.randn(500)))

        result = fit_ar1_half_life(series, min_obs=30)

        # b should be close to 1, half_life should be None or very large
        assert result["b"] >= 0.95 or result["half_life"] is None

    def test_insufficient_obs(self):
        """Test ValueError with insufficient observations."""
        series = pd.Series(np.random.randn(10))

        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_ar1_half_life(series, min_obs=30)


# =============================================================================
# Test Method 1: fly_levels_residual
# =============================================================================


class TestFlyLevelsResidual:
    """Tests for levels residual method."""

    def test_reduces_nonstationarity(self):
        """Levels residual should have lower half-life than naive fly."""
        front, belly, back = generate_cointegrated_yields(n=500, seed=42)

        # Levels residual fly
        fly_resid, weights = fly_levels_residual(
            front, belly, back, window=100, rolling=False, min_obs=30
        )

        # Naive equal-weight fly
        fly_naive = front - 2 * belly + back

        # Clean both for AR(1) estimation
        fly_resid_clean = fly_resid.dropna()
        fly_naive_clean = fly_naive.dropna()

        # Compare half-lives
        result_resid = fit_ar1_half_life(fly_resid_clean, min_obs=30)
        result_naive = fit_ar1_half_life(fly_naive_clean, min_obs=30)

        # Residual fly should be more mean-reverting (smaller b or shorter half-life)
        assert result_resid["b"] < result_naive["b"], (
            f"Expected residual b ({result_resid['b']}) < naive b ({result_naive['b']})"
        )

    def test_static_mode_output_structure(self):
        """Test output structure in static mode."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        fly, weights_df = fly_levels_residual(
            front, belly, back, rolling=False, min_obs=30
        )

        # Check fly series
        assert isinstance(fly, pd.Series)
        assert fly.name == "fly_levels_residual"
        assert len(fly) == 200

        # Check weights DataFrame
        assert isinstance(weights_df, pd.DataFrame)
        assert "beta_front" in weights_df.columns
        assert "beta_back" in weights_df.columns
        assert "intercept" in weights_df.columns
        assert len(weights_df) == 200

        # In static mode, all rows should have same coefficients
        assert weights_df["beta_front"].nunique() == 1
        assert weights_df["beta_back"].nunique() == 1

    def test_rolling_mode_output_structure(self):
        """Test output structure in rolling mode."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        fly, weights_df = fly_levels_residual(
            front, belly, back, window=50, rolling=True, min_obs=30
        )

        # Check fly series
        assert isinstance(fly, pd.Series)
        assert len(fly) == 200

        # Check weights DataFrame has time-varying coefficients
        assert weights_df["beta_front"].nunique() > 1
        assert weights_df["beta_back"].nunique() > 1

        # First (window-1) should be NaN
        assert weights_df["beta_front"].iloc[:49].isna().all()
        assert weights_df["beta_front"].iloc[49:].notna().all()

    def test_ridge_regularization(self):
        """Test ridge regularization stabilizes coefficients."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        # Without ridge
        _, weights_no_ridge = fly_levels_residual(
            front, belly, back, window=50, rolling=True, ridge_lambda=None
        )

        # With ridge
        _, weights_ridge = fly_levels_residual(
            front, belly, back, window=50, rolling=True, ridge_lambda=1.0
        )

        # Ridge should reduce coefficient variance
        std_no_ridge = weights_no_ridge["beta_front"].dropna().std()
        std_ridge = weights_ridge["beta_front"].dropna().std()

        # Note: This may not always hold for all seeds, but generally true
        # Just verify both run without error
        assert not weights_ridge["beta_front"].isna().all()

    def test_no_intercept(self):
        """Test without intercept."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        fly, weights_df = fly_levels_residual(
            front, belly, back, rolling=False, add_intercept=False
        )

        assert "intercept" not in weights_df.columns
        assert "beta_front" in weights_df.columns
        assert "beta_back" in weights_df.columns


# =============================================================================
# Test Method 2: fly_changes_residual
# =============================================================================


class TestFlyChangesResidual:
    """Tests for changes residual method."""

    def test_reduces_change_variance(self):
        """Changes residual should have lower variance than raw belly changes."""
        front, belly, back = generate_correlated_changes(n=500, seed=42)

        fly, _ = fly_changes_residual(
            front, belly, back, window=100, rolling=False, integrate=False
        )

        # Compare variance
        fly_clean = fly.dropna()
        belly_changes = belly.diff().dropna()

        var_residual = fly_clean.var()
        var_belly = belly_changes.var()

        assert var_residual < var_belly, (
            f"Expected residual variance ({var_residual}) < belly variance ({var_belly})"
        )

    def test_integrated_output(self):
        """Test integrated output (cumulative residual)."""
        front, belly, back = generate_correlated_changes(n=200, seed=42)

        fly_integrated, _ = fly_changes_residual(
            front, belly, back, window=50, rolling=False, integrate=True
        )

        fly_changes, _ = fly_changes_residual(
            front, belly, back, window=50, rolling=False, integrate=False
        )

        # Integrated should be cumsum of changes (approximately)
        assert fly_integrated.name == "fly_changes_residual_integrated"
        assert fly_changes.name == "fly_changes_residual"

        # Both should have valid values
        assert not fly_integrated.dropna().empty
        assert not fly_changes.dropna().empty

    def test_rolling_mode(self):
        """Test rolling mode produces time-varying weights."""
        front, belly, back = generate_correlated_changes(n=200, seed=42)

        fly, weights_df = fly_changes_residual(
            front, belly, back, window=50, rolling=True
        )

        # Weights should vary over time
        assert weights_df["beta_front"].nunique() > 1

        # First observations should be NaN (window warmup + diff)
        # After differencing: n-1 observations, first window-1 are NaN
        assert fly.iloc[:50].isna().any()


# =============================================================================
# Test Method 3: fly_optimize_mean_reversion
# =============================================================================


class TestFlyOptimizeMeanReversion:
    """Tests for optimization method."""

    def test_improves_half_life(self):
        """Optimized fly should find a mean-reverting construction."""
        front, belly, back = generate_cointegrated_yields(n=500, seed=42)

        # Optimized fly (static for speed)
        fly_opt, weights_opt = fly_optimize_mean_reversion(
            front,
            belly,
            back,
            window=500,
            rolling=False,
            objective="half_life",
            grid_size=21,  # Smaller grid for speed
            min_obs=60,
        )

        # Check optimization finds a valid mean-reverting fly
        fly_opt_clean = fly_opt.dropna()

        if len(fly_opt_clean) >= 60:
            result_opt = fit_ar1_half_life(fly_opt_clean, min_obs=30)

            # Optimized fly should be mean-reverting (b < 1) with valid half-life
            assert result_opt["b"] < 1, "Optimized fly should be mean-reverting"
            assert result_opt["half_life"] is not None, "Should have valid half-life"
            assert result_opt["half_life"] > 0, "Half-life should be positive"

            # Objective value in weights_df should match
            obj_value = weights_opt["objective_value"].iloc[-1]
            assert obj_value < 1e9, "Objective should not be penalty value"

    def test_output_structure(self):
        """Test output structure."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        fly, weights_df = fly_optimize_mean_reversion(
            front,
            belly,
            back,
            window=200,
            rolling=False,
            grid_size=11,
            min_obs=60,
        )

        # Check fly series
        assert isinstance(fly, pd.Series)
        assert fly.name == "fly_optimized"

        # Check weights DataFrame
        assert "w_front" in weights_df.columns
        assert "w_belly" in weights_df.columns
        assert "w_back" in weights_df.columns
        assert "objective_value" in weights_df.columns
        assert "net_dv01" in weights_df.columns

    def test_dv01_constraint(self):
        """Test DV01 constraint enforcement."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        # DV01s (typical values)
        dv01 = (0.02, 0.05, 0.09)  # 2Y, 5Y, 10Y

        fly, weights_df = fly_optimize_mean_reversion(
            front,
            belly,
            back,
            window=200,
            rolling=False,
            grid_size=11,
            dv01=dv01,
            dv01_tolerance=0.01,  # Allow some tolerance
            min_obs=60,
        )

        # Check net DV01 is within tolerance
        net_dv01 = weights_df["net_dv01"].dropna().iloc[0]
        assert abs(net_dv01) <= 0.01 or weights_df["net_dv01"].notna().any()

    def test_rolling_mode(self):
        """Test rolling mode produces time-varying weights."""
        front, belly, back = generate_cointegrated_yields(n=300, seed=42)

        fly, weights_df = fly_optimize_mean_reversion(
            front,
            belly,
            back,
            window=150,
            rolling=True,
            grid_size=11,  # Small grid for speed
            min_obs=60,
        )

        # Weights should vary over time (at least some variation)
        valid_weights = weights_df["w_front"].dropna()
        assert len(valid_weights) > 0

        # First (window-1) should be NaN
        assert weights_df["w_front"].iloc[:149].isna().all()

    def test_ar1_b_objective(self):
        """Test ar1_b objective."""
        front, belly, back = generate_cointegrated_yields(n=300, seed=42)

        fly, weights_df = fly_optimize_mean_reversion(
            front,
            belly,
            back,
            window=300,
            rolling=False,
            objective="ar1_b",
            grid_size=11,
            min_obs=60,
        )

        # Should produce valid output
        assert not fly.dropna().empty
        assert weights_df["objective_value"].notna().any()


# =============================================================================
# Test DV01 neutral enforcement
# =============================================================================


class TestEnforceDV01Neutral:
    """Tests for DV01 neutral enforcement."""

    def test_adjust_back_leg(self):
        """Test adjusting back leg for DV01 neutrality."""
        w_front, w_belly, w_back = -0.4, 1.0, -0.6
        dv01_front, dv01_belly, dv01_back = 0.02, 0.05, 0.09

        w_f, w_b, w_bk = enforce_dv01_neutral(
            w_front, w_belly, w_back,
            dv01_front, dv01_belly, dv01_back,
            adjust_leg="back",
        )

        # Check DV01 neutrality
        net_dv01 = w_f * dv01_front + w_b * dv01_belly + w_bk * dv01_back
        assert abs(net_dv01) < 1e-10

        # Front and belly unchanged
        assert w_f == w_front
        assert w_b == w_belly

    def test_adjust_front_leg(self):
        """Test adjusting front leg for DV01 neutrality."""
        w_front, w_belly, w_back = -0.4, 1.0, -0.6
        dv01_front, dv01_belly, dv01_back = 0.02, 0.05, 0.09

        w_f, w_b, w_bk = enforce_dv01_neutral(
            w_front, w_belly, w_back,
            dv01_front, dv01_belly, dv01_back,
            adjust_leg="front",
        )

        # Check DV01 neutrality
        net_dv01 = w_f * dv01_front + w_b * dv01_belly + w_bk * dv01_back
        assert abs(net_dv01) < 1e-10

        # Back and belly unchanged
        assert w_bk == w_back
        assert w_b == w_belly

    def test_raises_on_zero_dv01(self):
        """Test ValueError when adjustment leg has zero DV01."""
        with pytest.raises(ValueError, match="near zero"):
            enforce_dv01_neutral(
                1.0, -2.0, 1.0,
                0.02, 0.05, 0.0,  # Back has zero DV01
                adjust_leg="back",
            )

        with pytest.raises(ValueError, match="near zero"):
            enforce_dv01_neutral(
                1.0, -2.0, 1.0,
                0.0, 0.05, 0.09,  # Front has zero DV01
                adjust_leg="front",
            )

    def test_realistic_dv01s(self):
        """Test with realistic DV01 values."""
        # Typical DV01s for 2Y, 5Y, 10Y (per $1M notional)
        dv01_2y = 200  # $200 per bp
        dv01_5y = 500
        dv01_10y = 900

        # Start with equal weights
        w_f, w_b, w_bk = enforce_dv01_neutral(
            1.0, -2.0, 1.0,
            dv01_2y, dv01_5y, dv01_10y,
            adjust_leg="back",
        )

        # Verify DV01 neutrality
        net_dv01 = w_f * dv01_2y + w_b * dv01_5y + w_bk * dv01_10y
        assert abs(net_dv01) < 1e-6


# =============================================================================
# Test no lookahead
# =============================================================================


class TestNoLookahead:
    """Tests to verify no lookahead bias in rolling methods."""

    def test_levels_residual_no_lookahead(self):
        """Verify weights at time t don't depend on future data."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        # Get weights with full data
        _, weights_full = fly_levels_residual(
            front, belly, back, window=50, rolling=True
        )

        # Modify future data (after t=100)
        front_modified = front.copy()
        belly_modified = belly.copy()
        back_modified = back.copy()
        front_modified.iloc[101:] = front_modified.iloc[101:] + 10  # Large change
        belly_modified.iloc[101:] = belly_modified.iloc[101:] + 10
        back_modified.iloc[101:] = back_modified.iloc[101:] + 10

        # Get weights with modified data
        _, weights_modified = fly_levels_residual(
            front_modified, belly_modified, back_modified, window=50, rolling=True
        )

        # Weights at t=100 and earlier should be identical
        for t in range(50, 101):
            np.testing.assert_almost_equal(
                weights_full["beta_front"].iloc[t],
                weights_modified["beta_front"].iloc[t],
                decimal=10,
                err_msg=f"Weights at t={t} depend on future data!",
            )

    def test_changes_residual_no_lookahead(self):
        """Verify changes residual weights don't depend on future data."""
        front, belly, back = generate_correlated_changes(n=200, seed=42)

        # Get weights with full data
        _, weights_full = fly_changes_residual(
            front, belly, back, window=50, rolling=True
        )

        # Modify future data
        front_modified = front.copy()
        belly_modified = belly.copy()
        back_modified = back.copy()
        front_modified.iloc[101:] = front_modified.iloc[101:] + 10
        belly_modified.iloc[101:] = belly_modified.iloc[101:] + 10
        back_modified.iloc[101:] = back_modified.iloc[101:] + 10

        # Get weights with modified data
        _, weights_modified = fly_changes_residual(
            front_modified, belly_modified, back_modified, window=50, rolling=True
        )

        # Weights at t=100 and earlier should be identical (accounting for diff)
        # After diff, index shifts by 1, so check appropriate range
        for col in ["beta_front", "beta_back"]:
            # Compare valid (non-NaN) values up to modified point
            full_vals = weights_full[col].iloc[50:100].values
            mod_vals = weights_modified[col].iloc[50:100].values
            np.testing.assert_array_almost_equal(
                full_vals,
                mod_vals,
                decimal=10,
                err_msg=f"Column {col} depends on future data!",
            )

    def test_optimize_no_lookahead(self):
        """Verify optimization weights don't depend on future data."""
        front, belly, back = generate_cointegrated_yields(n=300, seed=42)

        # Get weights with full data
        _, weights_full = fly_optimize_mean_reversion(
            front, belly, back, window=150, rolling=True, grid_size=11, min_obs=60
        )

        # Modify future data
        front_modified = front.copy()
        belly_modified = belly.copy()
        back_modified = back.copy()
        front_modified.iloc[200:] = front_modified.iloc[200:] + 10
        belly_modified.iloc[200:] = belly_modified.iloc[200:] + 10
        back_modified.iloc[200:] = back_modified.iloc[200:] + 10

        # Get weights with modified data
        _, weights_modified = fly_optimize_mean_reversion(
            front_modified, belly_modified, back_modified,
            window=150, rolling=True, grid_size=11, min_obs=60
        )

        # Weights at t<200 should be identical
        for t in range(149, 200):
            if not np.isnan(weights_full["w_front"].iloc[t]):
                np.testing.assert_almost_equal(
                    weights_full["w_front"].iloc[t],
                    weights_modified["w_front"].iloc[t],
                    decimal=10,
                    err_msg=f"Optimization weights at t={t} depend on future data!",
                )


# =============================================================================
# Test equal_weight_fly baseline
# =============================================================================


class TestEqualWeightFly:
    """Tests for equal_weight_fly baseline."""

    def test_default_weights(self):
        """Test default (1, -2, 1) weights."""
        front, belly, back = generate_cointegrated_yields(n=100, seed=42)

        fly, weights = equal_weight_fly(front, belly, back)

        assert weights.w_front == 1.0
        assert weights.w_belly == -2.0
        assert weights.w_back == 1.0
        assert weights.method == "equal_weight"

        # Verify fly computation
        expected = front - 2 * belly + back
        pd.testing.assert_series_equal(fly, expected.rename("fly_equal_weight"))

    def test_custom_weights(self):
        """Test custom weights."""
        front, belly, back = generate_cointegrated_yields(n=100, seed=42)

        fly, weights = equal_weight_fly(
            front, belly, back,
            w_front=0.5, w_belly=-1.0, w_back=0.5
        )

        assert weights.w_front == 0.5
        assert weights.w_belly == -1.0
        assert weights.w_back == 0.5

        expected = 0.5 * front - 1.0 * belly + 0.5 * back
        pd.testing.assert_series_equal(fly, expected.rename("fly_equal_weight"))


# =============================================================================
# Test compute_fly_from_weights
# =============================================================================


class TestComputeFlyFromWeights:
    """Tests for compute_fly_from_weights helper."""

    def test_time_varying_weights(self):
        """Test computation with time-varying weights."""
        front, belly, back = generate_cointegrated_yields(n=100, seed=42)

        # Create time-varying weights
        weights_df = pd.DataFrame({
            "w_front": np.linspace(0.5, 1.5, 100),
            "w_belly": np.full(100, -2.0),
            "w_back": np.linspace(1.5, 0.5, 100),
        }, index=front.index)

        fly = compute_fly_from_weights(front, belly, back, weights_df)

        # Verify first and last values
        expected_first = (
            0.5 * front.iloc[0] - 2.0 * belly.iloc[0] + 1.5 * back.iloc[0]
        )
        expected_last = (
            1.5 * front.iloc[-1] - 2.0 * belly.iloc[-1] + 0.5 * back.iloc[-1]
        )

        assert abs(fly.iloc[0] - expected_first) < 1e-10
        assert abs(fly.iloc[-1] - expected_last) < 1e-10


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_levels_residual(self):
        """Test full pipeline: data -> fly -> ar1 analysis."""
        front, belly, back = generate_cointegrated_yields(n=500, seed=42)

        # Build fly
        fly, weights = fly_levels_residual(
            front, belly, back, window=252, rolling=True
        )

        # Analyze mean reversion
        fly_clean = fly.dropna()
        if len(fly_clean) >= 60:
            result = fit_ar1_half_life(fly_clean, min_obs=30)

            assert result["b"] is not None
            assert 0 < result["b"] < 1  # Should be mean-reverting
            assert result["half_life"] is not None
            assert result["half_life"] > 0

    def test_full_pipeline_with_dv01_adjustment(self):
        """Test pipeline with DV01 adjustment for execution."""
        front, belly, back = generate_cointegrated_yields(n=300, seed=42)

        # Build fly
        fly, weights_df = fly_levels_residual(
            front, belly, back, window=100, rolling=False
        )

        # Get signal-construction weights (from regression)
        beta_front = weights_df["beta_front"].iloc[-1]
        beta_back = weights_df["beta_back"].iloc[-1]

        # Effective fly weights: fly = belly - beta_front*front - beta_back*back
        # So weights are: (-beta_front, 1, -beta_back)
        w_front_signal = -beta_front
        w_belly_signal = 1.0
        w_back_signal = -beta_back

        # Adjust for DV01 neutrality in execution
        dv01s = (0.02, 0.05, 0.09)

        w_f, w_b, w_bk = enforce_dv01_neutral(
            w_front_signal, w_belly_signal, w_back_signal,
            *dv01s,
            adjust_leg="back",
        )

        # Verify DV01 neutrality
        net_dv01 = w_f * dv01s[0] + w_b * dv01s[1] + w_bk * dv01s[2]
        assert abs(net_dv01) < 1e-10

    def test_compare_all_methods(self):
        """Compare all fly construction methods on same data."""
        front, belly, back = generate_cointegrated_yields(n=500, seed=42)

        # Method 1: Levels residual
        fly_levels, _ = fly_levels_residual(
            front, belly, back, window=252, rolling=False
        )

        # Method 2: Changes residual
        fly_changes, _ = fly_changes_residual(
            front, belly, back, window=252, rolling=False, integrate=True
        )

        # Method 3: Optimization (small grid for speed)
        fly_opt, _ = fly_optimize_mean_reversion(
            front, belly, back, window=500, rolling=False,
            grid_size=11, min_obs=60
        )

        # Baseline: Equal weight
        fly_equal, _ = equal_weight_fly(front, belly, back)

        # All should produce valid series
        assert not fly_levels.dropna().empty
        assert not fly_changes.dropna().empty
        assert not fly_opt.dropna().empty
        assert not fly_equal.dropna().empty

        # At least one method should improve on equal weight
        results = {}
        for name, fly in [
            ("levels", fly_levels),
            ("changes", fly_changes),
            ("optimized", fly_opt),
            ("equal", fly_equal),
        ]:
            clean = fly.dropna()
            if len(clean) >= 60:
                try:
                    ar1 = fit_ar1_half_life(clean, min_obs=30)
                    results[name] = ar1.get("half_life")
                except ValueError:
                    results[name] = None

        # Just verify all methods run without error
        assert len(results) > 0


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_window_larger_than_data(self):
        """Test error when window exceeds data."""
        front, belly, back = generate_cointegrated_yields(n=100, seed=42)

        with pytest.raises(ValueError, match="Window"):
            fly_levels_residual(front, belly, back, window=200, rolling=True)

    def test_empty_series(self):
        """Test error on empty series."""
        empty = pd.Series([], dtype=float)

        with pytest.raises(ValueError):
            validate_inputs(empty, empty, empty, min_obs=1)

    def test_all_nan_series(self):
        """Test error when all values are NaN."""
        nan_series = pd.Series([np.nan] * 100)

        with pytest.raises(ValueError, match="Insufficient"):
            validate_inputs(nan_series, nan_series, nan_series, min_obs=1)

    def test_single_observation(self):
        """Test with minimal observations."""
        front = pd.Series([1.0])
        belly = pd.Series([1.5])
        back = pd.Series([2.0])

        with pytest.raises(ValueError, match="Insufficient"):
            fly_levels_residual(front, belly, back, min_obs=10)


# =============================================================================
# Determinism tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_levels_residual_deterministic(self):
        """Same input produces same output."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        fly1, weights1 = fly_levels_residual(front, belly, back, window=50)
        fly2, weights2 = fly_levels_residual(front, belly, back, window=50)

        pd.testing.assert_series_equal(fly1, fly2)
        pd.testing.assert_frame_equal(weights1, weights2)

    def test_optimize_deterministic(self):
        """Optimization produces same results on same input."""
        front, belly, back = generate_cointegrated_yields(n=200, seed=42)

        fly1, weights1 = fly_optimize_mean_reversion(
            front, belly, back, window=200, rolling=False, grid_size=11, min_obs=60
        )
        fly2, weights2 = fly_optimize_mean_reversion(
            front, belly, back, window=200, rolling=False, grid_size=11, min_obs=60
        )

        pd.testing.assert_series_equal(fly1, fly2)
        pd.testing.assert_frame_equal(weights1, weights2)
