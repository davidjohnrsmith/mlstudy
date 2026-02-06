"""Training utilities and experiment runner."""

from __future__ import annotations

from mlstudy.ml.train.metrics import (
    classification_metrics,
    regression_metrics,
    spearman_ic,
)

__all__ = [
    "classification_metrics",
    "ExperimentConfig",
    "ExperimentResult",
    "HorizonResult",
    "MultiHorizonConfig",
    "MultiHorizonResult",
    "regression_metrics",
    "run_experiment",
    "save_multi_horizon_result",
    "spearman_ic",
    "train_horizon_as_feature",
    "train_multi_horizon",
    "train_one_model_per_horizon",
]
