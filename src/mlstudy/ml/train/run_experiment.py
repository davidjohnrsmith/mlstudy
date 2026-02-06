"""Experiment runner for ML training pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from typing_extensions import Literal  # noqa: UP035

from mlstudy.ml.models.registry import build_model
from mlstudy.core.preprocess import PreprocessConfig, Preprocessor
from mlstudy.core.splitters.time import time_train_val_test_split
from mlstudy.core.splitters.walk_forward import walk_forward_splits
from mlstudy.ml.train.metrics import classification_metrics, regression_metrics


@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment.

    Attributes:
        task: "regression" or "classification".
        model_name: Name of model to use (see models.registry).
        split_strategy: "time" or "walk_forward".
        datetime_col: Column name for datetime.
        target_col: Column name for target variable.
        feature_cols: List of feature column names.
        group_col: Optional grouping column.
        preprocess: Preprocessing configuration.
        random_state: Random seed for reproducibility.

        # Time split params (used when split_strategy="time")
        train_end: End date for training set.
        val_end: End date for validation set.
        test_end: End date for test set.

        # Walk-forward params (used when split_strategy="walk_forward")
        train_days: Number of training days per fold.
        val_days: Number of validation days per fold.
        test_days: Number of test days per fold.
        step_days: Days to step between folds.
        expanding: Use expanding (True) or rolling (False) window.

        min_count: Minimum samples per group.
    """

    task: Literal["regression", "classification"]
    model_name: str
    split_strategy: Literal["time", "walk_forward"]
    datetime_col: str
    target_col: str
    feature_cols: list[str]
    group_col: str | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    random_state: int = 42

    # Time split params
    train_end: str | None = None
    val_end: str | None = None
    test_end: str | None = None

    # Walk-forward params
    train_days: int = 252
    val_days: int = 63
    test_days: int = 63
    step_days: int = 21
    expanding: bool = True

    min_count: int = 0


@dataclass
class ExperimentResult:
    """Results from an experiment run.

    Attributes:
        config: The experiment configuration used.
        train_metrics: Metrics on training set.
        val_metrics: Metrics on validation set.
        test_metrics: Metrics on test set.
        fold_metrics: Per-fold metrics (for walk-forward only).
        predictions: DataFrame with predictions.
        output_dir: Directory where artifacts are saved.
    """

    config: ExperimentConfig
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    predictions: pd.DataFrame | None = None
    output_dir: str | None = None


def _evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
) -> dict[str, float]:
    """Evaluate model on data."""
    y_pred = model.predict(X)

    if task == "regression":
        return regression_metrics(y, y_pred)
    else:
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
        return classification_metrics(y, y_pred, y_prob)


def _run_time_split(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run experiment with time-based split."""
    if config.train_end is None or config.val_end is None:
        raise ValueError("train_end and val_end required for time split")

    train_df, val_df, test_df = time_train_val_test_split(
        df=df,
        datetime_col=config.datetime_col,
        train_end=config.train_end,
        val_end=config.val_end,
        test_end=config.test_end,
        group_col=config.group_col,
        min_count=config.min_count,
    )

    # Extract features and targets
    X_train = train_df[config.feature_cols].values
    y_train = train_df[config.target_col].values
    X_val = val_df[config.feature_cols].values
    y_val = val_df[config.target_col].values
    X_test = test_df[config.feature_cols].values
    y_test = test_df[config.target_col].values

    # Fit preprocessor on train only
    preprocessor = Preprocessor(config.preprocess)
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    # Build and train model
    model = build_model(
        config.model_name,
        config.task,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_metrics = _evaluate(model, X_train, y_train, config.task)
    val_metrics = _evaluate(model, X_val, y_val, config.task)
    test_metrics = _evaluate(model, X_test, y_test, config.task)

    # Create predictions DataFrame
    predictions = _create_predictions_df(
        test_df, config, model, X_test, preprocessor
    )

    return ExperimentResult(
        config=config,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions=predictions,
    )


def _run_walk_forward(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run experiment with walk-forward cross-validation."""
    fold_metrics = []
    all_predictions = []

    # Aggregate metrics across folds
    all_train_metrics: list[dict] = []
    all_val_metrics: list[dict] = []
    all_test_metrics: list[dict] = []

    for fold in walk_forward_splits(
        df=df,
        datetime_col=config.datetime_col,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        step_days=config.step_days,
        expanding=config.expanding,
        group_col=config.group_col,
        min_count=config.min_count,
    ):
        # Extract features and targets
        X_train = fold.train_df[config.feature_cols].values
        y_train = fold.train_df[config.target_col].values
        X_val = fold.val_df[config.feature_cols].values
        y_val = fold.val_df[config.target_col].values
        X_test = fold.test_df[config.feature_cols].values
        y_test = fold.test_df[config.target_col].values

        # Fit preprocessor on train only
        preprocessor = Preprocessor(config.preprocess)
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)

        # Build and train model
        model = build_model(
            config.model_name,
            config.task,
            random_state=config.random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_m = _evaluate(model, X_train, y_train, config.task)
        val_m = _evaluate(model, X_val, y_val, config.task)
        test_m = _evaluate(model, X_test, y_test, config.task)

        all_train_metrics.append(train_m)
        all_val_metrics.append(val_m)
        all_test_metrics.append(test_m)

        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "train_metrics": train_m,
                "val_metrics": val_m,
                "test_metrics": test_m,
            }
        )

        # Collect predictions
        preds_df = fold.test_df.copy()
        preds_df["fold_id"] = fold.fold_id
        preds_df["y_pred"] = model.predict(X_test)
        all_predictions.append(preds_df)

    # Average metrics across folds
    train_metrics = _average_metrics(all_train_metrics)
    val_metrics = _average_metrics(all_val_metrics)
    test_metrics = _average_metrics(all_test_metrics)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else None

    return ExperimentResult(
        config=config,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )


def _average_metrics(metrics_list: list[dict]) -> dict[str, float]:
    """Average metrics across folds."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def _create_predictions_df(
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    model: Any,
    X_test: np.ndarray,
    preprocessor: Preprocessor,
) -> pd.DataFrame:
    """Create predictions DataFrame."""
    preds_df = test_df[[config.datetime_col, config.target_col]].copy()
    if config.group_col:
        preds_df[config.group_col] = test_df[config.group_col]
    preds_df["y_pred"] = model.predict(X_test)
    return preds_df


def run_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | None = None,
) -> ExperimentResult:
    """Run a complete ML experiment.

    Args:
        df: Input DataFrame with features and target.
        config: Experiment configuration.
        output_dir: Optional directory to save outputs. If None, uses
            outputs/experiments/<timestamp>_<model>/

    Returns:
        ExperimentResult with metrics and predictions.
    """
    # Run experiment based on split strategy
    if config.split_strategy == "time":
        result = _run_time_split(df, config)
    elif config.split_strategy == "walk_forward":
        result = _run_walk_forward(df, config)
    else:
        raise ValueError(f"Unknown split strategy: {config.split_strategy}")

    # Save outputs if output_dir specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/experiments/{timestamp}_{config.model_name}"

    _save_outputs(result, output_dir, df, config)
    result.output_dir = output_dir

    return result


def _save_outputs(
    result: ExperimentResult,
    output_dir: str,
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Save experiment outputs to disk."""
    outpath = Path(output_dir)
    outpath.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "train": result.train_metrics,
        "val": result.val_metrics,
        "test": result.test_metrics,
        "fold_metrics": result.fold_metrics,
        "config": {
            "task": config.task,
            "model_name": config.model_name,
            "split_strategy": config.split_strategy,
            "feature_cols": config.feature_cols,
            "target_col": config.target_col,
        },
    }
    with open(outpath / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save predictions
    if result.predictions is not None:
        result.predictions.to_csv(outpath / "predictions.csv", index=False)

    # Re-fit model on full train+val for saving (time split only)
    if config.split_strategy == "time" and config.train_end and config.val_end:
        train_df, val_df, _ = time_train_val_test_split(
            df=df,
            datetime_col=config.datetime_col,
            train_end=config.train_end,
            val_end=config.val_end,
            test_end=config.test_end,
            group_col=config.group_col,
            min_count=config.min_count,
        )
        combined = pd.concat([train_df, val_df])
        X = combined[config.feature_cols].values
        y = combined[config.target_col].values

        preprocessor = Preprocessor(config.preprocess)
        X = preprocessor.fit_transform(X)

        model = build_model(
            config.model_name,
            config.task,
            random_state=config.random_state,
        )
        model.fit(X, y)

        joblib.dump(model, outpath / "model.joblib")
        joblib.dump(preprocessor, outpath / "preprocessor.joblib")
