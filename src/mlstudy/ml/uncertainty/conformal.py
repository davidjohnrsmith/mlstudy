"""Conformal prediction for calibrated prediction intervals.

Adjusts prediction intervals using validation residuals to achieve
guaranteed coverage at target confidence levels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_coverage(
    y_true: NDArray,
    lower: NDArray,
    upper: NDArray,
) -> float:
    """Compute empirical coverage of prediction intervals.

    Args:
        y_true: True values.
        lower: Lower bounds of intervals.
        upper: Upper bounds of intervals.

    Returns:
        Fraction of true values within intervals.
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_interval_width(lower: NDArray, upper: NDArray) -> dict[str, float]:
    """Compute interval width statistics.

    Args:
        lower: Lower bounds.
        upper: Upper bounds.

    Returns:
        Dict with mean, median, std of interval widths.
    """
    widths = upper - lower
    return {
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "std_width": float(np.std(widths)),
    }


class ConformalCalibrator:
    """Calibrate prediction intervals using conformal prediction.

    Uses validation residuals to compute quantile adjustments that
    guarantee (asymptotically) the target coverage level.

    Two calibration methods:
    1. Symmetric: Single adjustment applied to both bounds
    2. Asymmetric: Separate adjustments for lower and upper bounds

    Example:
        >>> calibrator = ConformalCalibrator(target_coverage=0.9)
        >>> calibrator.fit(y_val, pred_val, lower_val, upper_val)
        >>> lower_cal, upper_cal = calibrator.calibrate(lower_test, upper_test)
        >>> # Calibrated intervals have ~90% coverage
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        method: str = "symmetric",
    ):
        """Initialize conformal calibrator.

        Args:
            target_coverage: Target coverage probability (e.g., 0.9 for 90%).
            method: "symmetric" for single adjustment, "asymmetric" for separate.
        """
        self.target_coverage = target_coverage
        self.method = method

        # Fitted parameters
        self.adjustment: float | None = None
        self.lower_adjustment: float | None = None
        self.upper_adjustment: float | None = None
        self._fitted = False

        # Diagnostics
        self.pre_calibration_coverage: float | None = None
        self.residuals: NDArray | None = None

    def fit(
        self,
        y_true: NDArray,
        y_pred: NDArray | None = None,
        lower: NDArray | None = None,
        upper: NDArray | None = None,
    ) -> ConformalCalibrator:
        """Fit calibration parameters from validation data.

        Args:
            y_true: True target values.
            y_pred: Point predictions (used if lower/upper not provided).
            lower: Lower bounds of prediction intervals.
            upper: Upper bounds of prediction intervals.

        Returns:
            Self for chaining.
        """
        y_true = np.asarray(y_true)

        # If only point predictions provided, create symmetric intervals
        if lower is None or upper is None:
            if y_pred is None:
                raise ValueError("Must provide either y_pred or (lower, upper)")
            y_pred = np.asarray(y_pred)
            residuals = y_true - y_pred
            # Use residuals for calibration
            lower = y_pred + np.percentile(residuals, 5)  # placeholder
            upper = y_pred + np.percentile(residuals, 95)

        lower = np.asarray(lower)
        upper = np.asarray(upper)

        # Store pre-calibration coverage
        self.pre_calibration_coverage = compute_coverage(y_true, lower, upper)

        if self.method == "symmetric":
            # Compute conformity scores: how far outside the interval
            scores = np.maximum(lower - y_true, y_true - upper)
            # Positive score = outside interval, negative = inside

            # Find adjustment at target quantile
            # For coverage p, we need the (1-p) quantile of scores
            alpha = 1 - self.target_coverage
            # Use finite-sample correction
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)

            self.adjustment = float(np.quantile(scores, q_level))
            self.residuals = scores

        else:  # asymmetric
            # Separate adjustments for lower and upper
            lower_scores = lower - y_true  # Positive if lower too high
            upper_scores = y_true - upper  # Positive if upper too low

            alpha = 1 - self.target_coverage
            n = len(y_true)
            q_level = np.ceil((n + 1) * (1 - alpha / 2)) / n
            q_level = min(q_level, 1.0)

            self.lower_adjustment = float(np.quantile(lower_scores, q_level))
            self.upper_adjustment = float(np.quantile(upper_scores, q_level))
            self.residuals = np.column_stack([lower_scores, upper_scores])

        self._fitted = True
        return self

    def calibrate(
        self,
        lower: NDArray,
        upper: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Apply calibration to prediction intervals.

        Args:
            lower: Lower bounds to calibrate.
            upper: Upper bounds to calibrate.

        Returns:
            Tuple of (calibrated_lower, calibrated_upper).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")

        lower = np.asarray(lower)
        upper = np.asarray(upper)

        if self.method == "symmetric":
            # Expand intervals symmetrically
            calibrated_lower = lower - self.adjustment
            calibrated_upper = upper + self.adjustment
        else:
            # Asymmetric adjustments
            calibrated_lower = lower - self.lower_adjustment
            calibrated_upper = upper + self.upper_adjustment

        return calibrated_lower, calibrated_upper

    def save(self, path: str | Path) -> str:
        """Save calibration parameters to JSON.

        Args:
            path: File path to save parameters.

        Returns:
            Path to saved file.
        """
        path = Path(path)

        params = {
            "target_coverage": self.target_coverage,
            "method": self.method,
            "pre_calibration_coverage": self.pre_calibration_coverage,
        }

        if self.method == "symmetric":
            params["adjustment"] = self.adjustment
        else:
            params["lower_adjustment"] = self.lower_adjustment
            params["upper_adjustment"] = self.upper_adjustment

        with open(path, "w") as f:
            json.dump(params, f, indent=2)

        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> ConformalCalibrator:
        """Load calibrator from saved parameters.

        Args:
            path: Path to saved JSON file.

        Returns:
            Loaded ConformalCalibrator.
        """
        path = Path(path)

        with open(path) as f:
            params = json.load(f)

        calibrator = cls(
            target_coverage=params["target_coverage"],
            method=params["method"],
        )
        calibrator.pre_calibration_coverage = params.get("pre_calibration_coverage")

        if calibrator.method == "symmetric":
            calibrator.adjustment = params["adjustment"]
        else:
            calibrator.lower_adjustment = params["lower_adjustment"]
            calibrator.upper_adjustment = params["upper_adjustment"]

        calibrator._fitted = True
        return calibrator

    @property
    def is_fitted(self) -> bool:
        """Whether calibrator has been fitted."""
        return self._fitted

    def get_params(self) -> dict[str, Any]:
        """Get calibration parameters."""
        params = {
            "target_coverage": self.target_coverage,
            "method": self.method,
            "pre_calibration_coverage": self.pre_calibration_coverage,
        }

        if self.method == "symmetric":
            params["adjustment"] = self.adjustment
        else:
            params["lower_adjustment"] = self.lower_adjustment
            params["upper_adjustment"] = self.upper_adjustment

        return params


def calibrate_intervals(
    y_val: NDArray,
    lower_val: NDArray,
    upper_val: NDArray,
    lower_test: NDArray,
    upper_test: NDArray,
    target_coverage: float = 0.9,
    method: str = "symmetric",
) -> tuple[NDArray, NDArray, ConformalCalibrator]:
    """Calibrate test intervals using validation data.

    Convenience function for one-shot calibration.

    Args:
        y_val: Validation true values.
        lower_val: Validation lower bounds.
        upper_val: Validation upper bounds.
        lower_test: Test lower bounds to calibrate.
        upper_test: Test upper bounds to calibrate.
        target_coverage: Target coverage probability.
        method: Calibration method ("symmetric" or "asymmetric").

    Returns:
        Tuple of (calibrated_lower, calibrated_upper, calibrator).

    Example:
        >>> lower_cal, upper_cal, cal = calibrate_intervals(
        ...     y_val, lower_val, upper_val,
        ...     lower_test, upper_test,
        ...     target_coverage=0.9
        ... )
    """
    calibrator = ConformalCalibrator(target_coverage=target_coverage, method=method)
    calibrator.fit(y_val, lower=lower_val, upper=upper_val)
    lower_cal, upper_cal = calibrator.calibrate(lower_test, upper_test)

    return lower_cal, upper_cal, calibrator
