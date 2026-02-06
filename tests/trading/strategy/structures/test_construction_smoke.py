"""Smoke tests for shared construction primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.structures.construction.construction import (
    changes_residual_3leg,
    enforce_dv01_neutral,
    fit_ar1_half_life,
    levels_residual_3leg,
    optimize_stationarity_3leg,
)


def generate_synthetic_yields(n: int = 200, seed: int = 42):
    """Generate synthetic yield series for testing."""
    np.random.seed(seed)

    # Random walks with some cointegration structure
    front_changes = np.random.randn(n) * 0.05
    back_changes = np.random.randn(n) * 0.07

    front = np.cumsum(front_changes) + 2.0
    back = np.cumsum(back_changes) + 4.0

    # Belly is combination + noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + np.random.randn() * 0.02
    belly = 0.5 * front + 0.5 * back + noise

    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    return (
        pd.Series(front, index=dates, name="front"),
        pd.Series(belly, index=dates, name="belly"),
        pd.Series(back, index=dates, name="back"),
    )


class TestLevelsResidual:
    """Smoke tests for levels_residual_3leg."""

    def test_returns_correct_shapes(self):
        """Test output shapes are correct."""
        front, belly, back = generate_synthetic_yields(n=200)

        fly, weights_df = levels_residual_3leg(
            front, belly, back, window=50, rolling=True, min_obs=30
        )

        # Check fly series
        assert isinstance(fly, pd.Series)
        assert len(fly) == 200

        # Check weights DataFrame
        assert isinstance(weights_df, pd.DataFrame)
        assert len(weights_df) == 200
        assert "beta_front" in weights_df.columns
        assert "beta_back" in weights_df.columns
        assert "intercept" in weights_df.columns

    def test_weights_are_finite(self):
        """Test weights are finite after warmup."""
        front, belly, back = generate_synthetic_yields(n=200)

        fly, weights_df = levels_residual_3leg(
            front, belly, back, window=50, rolling=True, min_obs=30
        )

        # After warmup, weights should be finite
        valid_weights = weights_df.iloc[49:].dropna()
        assert len(valid_weights) > 0
        assert np.isfinite(valid_weights["beta_front"]).all()
        assert np.isfinite(valid_weights["beta_back"]).all()

    def test_static_mode(self):
        """Test static (non-rolling) mode."""
        front, belly, back = generate_synthetic_yields(n=200)

        fly, weights_df = levels_residual_3leg(
            front, belly, back, rolling=False, min_obs=30
        )

        # All weights should be same in static mode
        assert weights_df["beta_front"].nunique() == 1
        assert weights_df["beta_back"].nunique() == 1


class TestChangesResidual:
    """Smoke tests for changes_residual_3leg."""

    def test_returns_correct_shapes(self):
        """Test output shapes are correct."""
        front, belly, back = generate_synthetic_yields(n=200)

        fly, weights_df = changes_residual_3leg(
            front, belly, back, window=50, rolling=True, min_obs=30
        )

        # Fly series should exist
        assert isinstance(fly, pd.Series)
        assert len(fly) > 0

        # Weights DataFrame should have expected columns
        assert "beta_front" in weights_df.columns
        assert "beta_back" in weights_df.columns


class TestEnforceDV01Neutral:
    """Smoke tests for enforce_dv01_neutral."""

    def test_net_dv01_is_zero(self):
        """Test that net DV01 is approximately zero after adjustment."""
        w_front, w_belly, w_back = 1.0, -2.0, 1.0
        dv01_front, dv01_belly, dv01_back = 0.02, 0.05, 0.09

        w_f, w_b, w_bk = enforce_dv01_neutral(
            w_front, w_belly, w_back,
            dv01_front, dv01_belly, dv01_back,
            adjust_leg="back",
        )

        net_dv01 = w_f * dv01_front + w_b * dv01_belly + w_bk * dv01_back
        assert abs(net_dv01) < 1e-10

    def test_adjust_front_leg(self):
        """Test adjusting front leg."""
        w_front, w_belly, w_back = 1.0, -2.0, 1.0
        dv01_front, dv01_belly, dv01_back = 0.02, 0.05, 0.09

        w_f, w_b, w_bk = enforce_dv01_neutral(
            w_front, w_belly, w_back,
            dv01_front, dv01_belly, dv01_back,
            adjust_leg="front",
        )

        net_dv01 = w_f * dv01_front + w_b * dv01_belly + w_bk * dv01_back
        assert abs(net_dv01) < 1e-10

    def test_raises_on_zero_dv01(self):
        """Test ValueError when adjustment leg has zero DV01."""
        with pytest.raises(ValueError, match="near zero"):
            enforce_dv01_neutral(
                1.0, -2.0, 1.0,
                0.02, 0.05, 0.0,  # Back has zero DV01
                adjust_leg="back",
            )


class TestOptimizeStationarity:
    """Smoke tests for optimize_stationarity_3leg."""

    def test_returns_weights(self):
        """Test optimization returns valid weights."""
        front, belly, back = generate_synthetic_yields(n=200)

        fly, weights_df = optimize_stationarity_3leg(
            front, belly, back,
            window=200,
            rolling=False,
            grid_size=11,  # Small grid for speed
            min_obs=60,
        )

        # Check output structure
        assert isinstance(fly, pd.Series)
        assert isinstance(weights_df, pd.DataFrame)

        # Check weights columns
        assert "w_front" in weights_df.columns
        assert "w_belly" in weights_df.columns
        assert "w_back" in weights_df.columns
        assert "objective_value" in weights_df.columns

        # Weights should be finite
        assert np.isfinite(weights_df["w_front"].iloc[-1])
        assert np.isfinite(weights_df["w_back"].iloc[-1])

    def test_objective_value_is_reasonable(self):
        """Test objective value is not a penalty value."""
        front, belly, back = generate_synthetic_yields(n=200)

        fly, weights_df = optimize_stationarity_3leg(
            front, belly, back,
            window=200,
            rolling=False,
            grid_size=11,
            min_obs=60,
        )

        obj_value = weights_df["objective_value"].iloc[-1]
        assert obj_value < 1e9  # Not a penalty value


class TestFitAR1HalfLife:
    """Smoke tests for fit_ar1_half_life."""

    def test_mean_reverting_series(self):
        """Test on a mean-reverting series."""
        np.random.seed(42)
        n = 200

        # Generate AR(1) with b=0.9
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + np.random.randn() * 0.1

        series = pd.Series(x)
        result = fit_ar1_half_life(series, min_obs=30)

        assert "b" in result
        assert "half_life" in result
        assert "residual_std" in result

        # Should be mean-reverting
        assert 0 < result["b"] < 1
        assert result["half_life"] is not None
        assert result["half_life"] > 0

    def test_raises_on_insufficient_obs(self):
        """Test ValueError with insufficient observations."""
        series = pd.Series(np.random.randn(10))

        with pytest.raises(ValueError, match="Insufficient"):
            fit_ar1_half_life(series, min_obs=30)
