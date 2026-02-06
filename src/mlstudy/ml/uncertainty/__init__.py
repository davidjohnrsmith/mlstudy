"""Uncertainty quantification for predictions.

Provides:
- Quantile regression for prediction intervals
- Conformal prediction for calibrated coverage guarantees
"""

from __future__ import annotations

from mlstudy.ml.uncertainty.conformal import (
    ConformalCalibrator,
    calibrate_intervals,
    compute_coverage,
)

__all__ = [
    "QuantilePredictor",
    "build_quantile_lgbm",
    "predict_intervals",
    "ConformalCalibrator",
    "calibrate_intervals",
    "compute_coverage",
]
