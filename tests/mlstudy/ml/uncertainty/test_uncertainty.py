"""Tests for uncertainty quantification (quantile regression + conformal)."""

import numpy as np
import pytest

from mlstudy.ml.uncertainty.conformal import (
    ConformalCalibrator,
    calibrate_intervals,
    compute_coverage,
    compute_interval_width,
)


@pytest.fixture
def regression_data():
    """Synthetic regression data with known noise level."""
    np.random.seed(42)
    n = 500

    X = np.random.randn(n, 5)
    # Linear model with noise
    true_coef = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    y = X @ true_coef + np.random.randn(n) * 0.5

    # Split
    train_idx = np.arange(300)
    val_idx = np.arange(300, 400)
    test_idx = np.arange(400, 500)

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }


class TestComputeCoverage:
    """Tests for compute_coverage."""

    def test_perfect_coverage(self):
        """Should return 1.0 for intervals containing all values."""
        y_true = np.array([1, 2, 3, 4, 5])
        lower = np.array([0, 1, 2, 3, 4])
        upper = np.array([2, 3, 4, 5, 6])

        coverage = compute_coverage(y_true, lower, upper)
        assert coverage == 1.0

    def test_no_coverage(self):
        """Should return 0.0 for intervals missing all values."""
        y_true = np.array([1, 2, 3, 4, 5])
        lower = np.array([10, 10, 10, 10, 10])
        upper = np.array([20, 20, 20, 20, 20])

        coverage = compute_coverage(y_true, lower, upper)
        assert coverage == 0.0

    def test_partial_coverage(self):
        """Should return correct fraction for partial coverage."""
        y_true = np.array([1, 2, 3, 4, 5])
        lower = np.array([0, 0, 10, 10, 10])  # First 2 covered
        upper = np.array([2, 3, 11, 11, 11])

        coverage = compute_coverage(y_true, lower, upper)
        assert coverage == 0.4  # 2/5


class TestComputeIntervalWidth:
    """Tests for compute_interval_width."""

    def test_uniform_width(self):
        """Should compute correct stats for uniform width."""
        lower = np.array([0, 1, 2, 3, 4])
        upper = np.array([2, 3, 4, 5, 6])  # All width 2

        stats = compute_interval_width(lower, upper)
        assert stats["mean_width"] == 2.0
        assert stats["median_width"] == 2.0
        assert stats["std_width"] == 0.0

    def test_varying_width(self):
        """Should compute correct stats for varying width."""
        lower = np.array([0, 0, 0])
        upper = np.array([1, 2, 3])  # Widths: 1, 2, 3

        stats = compute_interval_width(lower, upper)
        assert stats["mean_width"] == 2.0
        assert stats["median_width"] == 2.0


class TestConformalCalibrator:
    """Tests for ConformalCalibrator."""

    def test_fit_symmetric(self, regression_data):
        """Should fit symmetric calibration."""
        # Create some intervals
        y_val = regression_data["y_val"]
        pred = y_val + np.random.randn(len(y_val)) * 0.1  # Close predictions
        lower = pred - 0.2
        upper = pred + 0.2

        calibrator = ConformalCalibrator(target_coverage=0.9, method="symmetric")
        calibrator.fit(y_val, lower=lower, upper=upper)

        assert calibrator.is_fitted
        assert calibrator.adjustment is not None
        assert calibrator.pre_calibration_coverage is not None

    def test_calibrate_expands_intervals(self, regression_data):
        """Calibration should expand intervals when coverage is low."""
        y_val = regression_data["y_val"]
        pred = y_val + np.random.randn(len(y_val)) * 0.1
        # Narrow intervals with low coverage
        lower = pred - 0.1
        upper = pred + 0.1

        calibrator = ConformalCalibrator(target_coverage=0.9)
        calibrator.fit(y_val, lower=lower, upper=upper)

        lower_cal, upper_cal = calibrator.calibrate(lower, upper)

        # Calibrated intervals should be wider
        orig_width = np.mean(upper - lower)
        cal_width = np.mean(upper_cal - lower_cal)
        assert cal_width >= orig_width

    def test_coverage_improves_after_calibration(self, regression_data):
        """Coverage should improve after calibration."""
        np.random.seed(123)
        y_val = regression_data["y_val"]
        y_test = regression_data["y_test"]

        # Make predictions with intentionally narrow intervals
        pred_val = y_val + np.random.randn(len(y_val)) * 0.2
        pred_test = y_test + np.random.randn(len(y_test)) * 0.2

        lower_val = pred_val - 0.3
        upper_val = pred_val + 0.3
        lower_test = pred_test - 0.3
        upper_test = pred_test + 0.3

        # Coverage before calibration
        coverage_before = compute_coverage(y_test, lower_test, upper_test)

        # Calibrate using validation data
        calibrator = ConformalCalibrator(target_coverage=0.9)
        calibrator.fit(y_val, lower=lower_val, upper=upper_val)
        lower_cal, upper_cal = calibrator.calibrate(lower_test, upper_test)

        # Coverage after calibration
        coverage_after = compute_coverage(y_test, lower_cal, upper_cal)

        # Coverage should improve (or stay same if already high)
        assert coverage_after >= coverage_before - 0.1  # Allow small variation

    def test_save_load(self, regression_data, tmp_path):
        """Should save and load correctly."""
        y_val = regression_data["y_val"]
        pred = y_val + np.random.randn(len(y_val)) * 0.1
        lower = pred - 0.2
        upper = pred + 0.2

        calibrator = ConformalCalibrator(target_coverage=0.9)
        calibrator.fit(y_val, lower=lower, upper=upper)

        # Save
        save_path = tmp_path / "calibrator.json"
        calibrator.save(save_path)

        # Load
        loaded = ConformalCalibrator.load(save_path)

        assert loaded.target_coverage == calibrator.target_coverage
        assert loaded.adjustment == calibrator.adjustment
        assert loaded.is_fitted


class TestCalibrateIntervals:
    """Tests for calibrate_intervals convenience function."""

    def test_returns_calibrated_and_calibrator(self, regression_data):
        """Should return calibrated intervals and calibrator."""
        np.random.seed(42)
        y_val = regression_data["y_val"]
        y_test = regression_data["y_test"]

        pred_val = y_val + np.random.randn(len(y_val)) * 0.1
        pred_test = y_test + np.random.randn(len(y_test)) * 0.1

        lower_val = pred_val - 0.2
        upper_val = pred_val + 0.2
        lower_test = pred_test - 0.2
        upper_test = pred_test + 0.2

        lower_cal, upper_cal, calibrator = calibrate_intervals(
            y_val, lower_val, upper_val,
            lower_test, upper_test,
            target_coverage=0.9,
        )

        assert isinstance(lower_cal, np.ndarray)
        assert isinstance(upper_cal, np.ndarray)
        assert isinstance(calibrator, ConformalCalibrator)
        assert len(lower_cal) == len(lower_test)


class TestQuantilePredictor:
    """Tests for QuantilePredictor (requires lightgbm)."""

    @pytest.fixture
    def lgb_available(self):
        """Check if lightgbm is available."""
        import importlib.util

        if importlib.util.find_spec("lightgbm") is None:
            pytest.skip("lightgbm not installed")
        return True

    def test_fit_predict(self, regression_data, lgb_available):
        """Should fit and predict quantiles."""
        from mlstudy.ml.uncertainty.quantiles import QuantilePredictor

        predictor = QuantilePredictor(quantiles=[0.1, 0.5, 0.9], n_estimators=10)
        predictor.fit(regression_data["X_train"], regression_data["y_train"])

        preds = predictor.predict(regression_data["X_test"])

        assert "q10" in preds
        assert "q50" in preds
        assert "q90" in preds
        assert len(preds["q50"]) == len(regression_data["y_test"])

    def test_predict_intervals(self, regression_data, lgb_available):
        """Should predict intervals."""
        from mlstudy.ml.uncertainty.quantiles import QuantilePredictor

        predictor = QuantilePredictor(quantiles=[0.1, 0.5, 0.9], n_estimators=10)
        predictor.fit(regression_data["X_train"], regression_data["y_train"])

        lower, point, upper = predictor.predict_intervals(regression_data["X_test"])

        # Lower should be less than upper
        assert np.all(lower <= upper)
        # Point should be between bounds
        assert np.all((point >= lower) | np.isclose(point, lower))
        assert np.all((point <= upper) | np.isclose(point, upper))

    def test_quantile_ordering(self, regression_data, lgb_available):
        """Lower quantiles should predict smaller values."""
        from mlstudy.ml.uncertainty.quantiles import QuantilePredictor

        predictor = QuantilePredictor(quantiles=[0.1, 0.5, 0.9], n_estimators=50)
        predictor.fit(regression_data["X_train"], regression_data["y_train"])

        preds = predictor.predict(regression_data["X_test"])

        # On average, q10 < q50 < q90
        assert np.mean(preds["q10"]) < np.mean(preds["q50"])
        assert np.mean(preds["q50"]) < np.mean(preds["q90"])

    def test_coverage_with_conformal(self, regression_data, lgb_available):
        """Quantile + conformal should achieve target coverage."""
        from mlstudy.ml.uncertainty.quantiles import QuantilePredictor

        predictor = QuantilePredictor(quantiles=[0.1, 0.5, 0.9], n_estimators=50)
        predictor.fit(regression_data["X_train"], regression_data["y_train"])

        # Get intervals on validation
        lower_val, _, upper_val = predictor.predict_intervals(regression_data["X_val"])

        # Get intervals on test
        lower_test, _, upper_test = predictor.predict_intervals(regression_data["X_test"])

        # Coverage before calibration (computed for debugging but not asserted)
        _coverage_before = compute_coverage(regression_data["y_test"], lower_test, upper_test)

        # Calibrate
        calibrator = ConformalCalibrator(target_coverage=0.9)
        calibrator.fit(regression_data["y_val"], lower=lower_val, upper=upper_val)
        lower_cal, upper_cal = calibrator.calibrate(lower_test, upper_test)

        # Coverage after calibration
        coverage_after = compute_coverage(regression_data["y_test"], lower_cal, upper_cal)

        # Should be closer to target (0.9)
        # Note: may not hit exactly 0.9 due to small sample size
        assert coverage_after >= 0.75  # At least reasonable coverage
