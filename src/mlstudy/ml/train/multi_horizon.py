"""Multi-horizon training strategies.

Supports:
- Strategy A: One model per horizon (loop over horizons)
- Strategy B: Horizon as feature (stacked dataset)
- Uncertainty: Quantile regression + conformal calibration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlstudy.core.preprocess import Preprocessor
from mlstudy.ml.targets.horizons import MultiHorizonTargetGenerator
from mlstudy.ml.uncertainty.conformal import (
    ConformalCalibrator,
    compute_coverage,
    compute_interval_width,
)
from mlstudy.ml.uncertainty.quantiles import QuantilePredictor


@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon training."""

    horizons: list[int]
    strategy: str = "per_horizon"  # "per_horizon" or "horizon_feature"
    uncertainty: str = "none"  # "none", "quantile", "quantile+conformal"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    target_coverage: float = 0.9
    model_type: str = "lgbm"
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class HorizonResult:
    """Results for a single horizon."""

    horizon: int
    metrics: dict[str, float]
    predictions: NDArray
    intervals: tuple[NDArray, NDArray] | None = None
    calibrated_intervals: tuple[NDArray, NDArray] | None = None
    coverage_stats: dict[str, float] | None = None


@dataclass
class MultiHorizonResult:
    """Results for all horizons."""

    config: MultiHorizonConfig
    horizon_results: dict[int, HorizonResult]
    models: dict[int, Any]
    quantile_models: dict[int, QuantilePredictor] | None = None
    calibrators: dict[int, ConformalCalibrator] | None = None
    preprocessor: Preprocessor | None = None


def _build_lgbm_regressor(params: dict[str, Any]) -> Any:
    """Build LightGBM regressor."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm required for multi-horizon training") from e

    default_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
    }
    default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def train_one_model_per_horizon(
    X_train: NDArray,
    y_train_dict: dict[int, NDArray],
    X_val: NDArray,
    y_val_dict: dict[int, NDArray],
    config: MultiHorizonConfig,
) -> MultiHorizonResult:
    """Strategy A: Train one model per horizon.

    Args:
        X_train: Training features.
        y_train_dict: Dict mapping horizon to training targets.
        X_val: Validation features.
        y_val_dict: Dict mapping horizon to validation targets.
        config: Training configuration.

    Returns:
        MultiHorizonResult with models and metrics per horizon.
    """
    models = {}
    quantile_models = {} if config.uncertainty != "none" else None
    calibrators = {} if "conformal" in config.uncertainty else None
    horizon_results = {}

    for horizon in config.horizons:
        y_train = y_train_dict[horizon]
        y_val = y_val_dict[horizon]

        # Train point prediction model
        model = _build_lgbm_regressor(config.model_params)
        model.fit(X_train, y_train)
        models[horizon] = model

        # Predictions
        preds_val = model.predict(X_val)

        # Metrics
        mse = float(np.mean((y_val - preds_val) ** 2))
        mae = float(np.mean(np.abs(y_val - preds_val)))
        r2 = float(1 - np.sum((y_val - preds_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

        metrics = {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "r2": r2}

        intervals = None
        calibrated_intervals = None
        coverage_stats = None

        # Uncertainty quantification
        if config.uncertainty != "none":
            # Train quantile models
            qp = QuantilePredictor(quantiles=config.quantiles, **config.model_params)
            qp.fit(X_train, y_train)
            quantile_models[horizon] = qp

            # Get intervals
            lower, point, upper = qp.predict_intervals(X_val)
            intervals = (lower, upper)

            # Coverage before calibration
            coverage_pre = compute_coverage(y_val, lower, upper)
            width_stats = compute_interval_width(lower, upper)

            coverage_stats = {
                "coverage_pre_calibration": coverage_pre,
                "target_coverage": config.target_coverage,
                **width_stats,
            }

            # Conformal calibration
            if "conformal" in config.uncertainty:
                calibrator = ConformalCalibrator(target_coverage=config.target_coverage)
                calibrator.fit(y_val, lower=lower, upper=upper)
                calibrators[horizon] = calibrator

                lower_cal, upper_cal = calibrator.calibrate(lower, upper)
                calibrated_intervals = (lower_cal, upper_cal)

                coverage_post = compute_coverage(y_val, lower_cal, upper_cal)
                width_stats_cal = compute_interval_width(lower_cal, upper_cal)

                coverage_stats["coverage_post_calibration"] = coverage_post
                coverage_stats["calibrated_mean_width"] = width_stats_cal["mean_width"]
                coverage_stats["adjustment"] = calibrator.adjustment

        horizon_results[horizon] = HorizonResult(
            horizon=horizon,
            metrics=metrics,
            predictions=preds_val,
            intervals=intervals,
            calibrated_intervals=calibrated_intervals,
            coverage_stats=coverage_stats,
        )

    return MultiHorizonResult(
        config=config,
        horizon_results=horizon_results,
        models=models,
        quantile_models=quantile_models,
        calibrators=calibrators,
    )


def train_horizon_as_feature(
    X_train: NDArray,
    y_train_dict: dict[int, NDArray],
    X_val: NDArray,
    y_val_dict: dict[int, NDArray],
    config: MultiHorizonConfig,
) -> MultiHorizonResult:
    """Strategy B: Train single model with horizon as feature.

    Stacks data across horizons and adds horizon as an additional feature.

    Args:
        X_train: Training features.
        y_train_dict: Dict mapping horizon to training targets.
        X_val: Validation features.
        y_val_dict: Dict mapping horizon to validation targets.
        config: Training configuration.

    Returns:
        MultiHorizonResult with single model and metrics per horizon.
    """
    horizons = config.horizons

    # Stack training data
    X_train_list = []
    y_train_list = []
    for h in horizons:
        n = len(y_train_dict[h])
        horizon_feature = np.full((n, 1), h)
        X_train_list.append(np.hstack([X_train, horizon_feature]))
        y_train_list.append(y_train_dict[h])

    X_train_stacked = np.vstack(X_train_list)
    y_train_stacked = np.concatenate(y_train_list)

    # Train single model
    model = _build_lgbm_regressor(config.model_params)
    model.fit(X_train_stacked, y_train_stacked)

    # Single model stored with key 0
    models = {0: model}

    # For uncertainty, also train quantile models on stacked data
    quantile_models = None
    calibrators = {} if "conformal" in config.uncertainty else None

    if config.uncertainty != "none":
        qp = QuantilePredictor(quantiles=config.quantiles, **config.model_params)
        qp.fit(X_train_stacked, y_train_stacked)
        quantile_models = {0: qp}  # Single model

    # Evaluate per horizon
    horizon_results = {}

    for horizon in horizons:
        y_val = y_val_dict[horizon]
        n_val = len(y_val)

        # Add horizon feature
        horizon_feature = np.full((n_val, 1), horizon)
        X_val_h = np.hstack([X_val, horizon_feature])

        # Predictions
        preds_val = model.predict(X_val_h)

        # Metrics
        mse = float(np.mean((y_val - preds_val) ** 2))
        mae = float(np.mean(np.abs(y_val - preds_val)))
        r2 = float(1 - np.sum((y_val - preds_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

        metrics = {"mse": mse, "rmse": np.sqrt(mse), "mae": mae, "r2": r2}

        intervals = None
        calibrated_intervals = None
        coverage_stats = None

        if config.uncertainty != "none":
            qp = quantile_models[0]
            lower, point, upper = qp.predict_intervals(X_val_h)
            intervals = (lower, upper)

            coverage_pre = compute_coverage(y_val, lower, upper)
            width_stats = compute_interval_width(lower, upper)

            coverage_stats = {
                "coverage_pre_calibration": coverage_pre,
                "target_coverage": config.target_coverage,
                **width_stats,
            }

            if "conformal" in config.uncertainty:
                calibrator = ConformalCalibrator(target_coverage=config.target_coverage)
                calibrator.fit(y_val, lower=lower, upper=upper)
                calibrators[horizon] = calibrator

                lower_cal, upper_cal = calibrator.calibrate(lower, upper)
                calibrated_intervals = (lower_cal, upper_cal)

                coverage_post = compute_coverage(y_val, lower_cal, upper_cal)
                width_stats_cal = compute_interval_width(lower_cal, upper_cal)

                coverage_stats["coverage_post_calibration"] = coverage_post
                coverage_stats["calibrated_mean_width"] = width_stats_cal["mean_width"]
                coverage_stats["adjustment"] = calibrator.adjustment

        horizon_results[horizon] = HorizonResult(
            horizon=horizon,
            metrics=metrics,
            predictions=preds_val,
            intervals=intervals,
            calibrated_intervals=calibrated_intervals,
            coverage_stats=coverage_stats,
        )

    return MultiHorizonResult(
        config=config,
        horizon_results=horizon_results,
        models=models,
        quantile_models=quantile_models,
        calibrators=calibrators,
    )


def train_multi_horizon(
    df: pd.DataFrame,
    feature_cols: list[str],
    value_col: str,
    datetime_col: str,
    group_col: str | None,
    train_end: str,
    val_end: str,
    config: MultiHorizonConfig,
    preprocessor: Preprocessor | None = None,
) -> MultiHorizonResult:
    """Train multi-horizon models on panel data.

    Args:
        df: DataFrame with features, value column, datetime, and optional group.
        feature_cols: Feature column names.
        value_col: Value column for target generation.
        datetime_col: Datetime column for splitting.
        group_col: Group column for panel data.
        train_end: End date for training data (exclusive).
        val_end: End date for validation data (exclusive).
        config: Multi-horizon training configuration.
        preprocessor: Optional preprocessor (will be fit on training data).

    Returns:
        MultiHorizonResult with models, metrics, and uncertainty estimates.
    """
    # Generate multi-horizon targets
    target_gen = MultiHorizonTargetGenerator(
        horizons=config.horizons,
        value_col=value_col,
        datetime_col=datetime_col,
        group_col=group_col,
    )
    df = target_gen.fit_transform(df)

    # Split by date
    train_mask = df[datetime_col] < pd.Timestamp(train_end)
    val_mask = (df[datetime_col] >= pd.Timestamp(train_end)) & (df[datetime_col] < pd.Timestamp(val_end))

    df_train = df[train_mask]
    df_val = df[val_mask]

    # Get X, y
    X_train, y_train_dict = target_gen.get_X_y(df_train, feature_cols, dropna=True)
    X_val, y_val_dict = target_gen.get_X_y(df_val, feature_cols, dropna=True)

    # Preprocess
    if preprocessor is not None:
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_val = preprocessor.transform(X_val)

    # Train based on strategy
    if config.strategy == "per_horizon":
        result = train_one_model_per_horizon(X_train, y_train_dict, X_val, y_val_dict, config)
    elif config.strategy == "horizon_feature":
        result = train_horizon_as_feature(X_train, y_train_dict, X_val, y_val_dict, config)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    result.preprocessor = preprocessor
    return result


def save_multi_horizon_result(
    result: MultiHorizonResult,
    path: str | Path,
) -> dict[str, str]:
    """Save multi-horizon training results.

    Saves:
    - manifest.json with configuration and metrics
    - model files (LightGBM text format)
    - quantile model files (if uncertainty enabled)
    - conformal calibrator parameters (if enabled)

    Args:
        result: Training result to save.
        path: Output directory.

    Returns:
        Dict mapping artifact name to file path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Save models
    model_dir = path / "models"
    model_dir.mkdir(exist_ok=True)

    if result.config.strategy == "per_horizon":
        for horizon, model in result.models.items():
            model_path = model_dir / f"model_h{horizon}.txt"
            model.booster_.save_model(str(model_path))
            artifacts[f"model_h{horizon}"] = str(model_path)
    else:
        model_path = model_dir / "model_stacked.txt"
        result.models[0].booster_.save_model(str(model_path))
        artifacts["model_stacked"] = str(model_path)

    # Save quantile models
    if result.quantile_models:
        quantile_dir = path / "quantile_models"
        quantile_dir.mkdir(exist_ok=True)

        if result.config.strategy == "per_horizon":
            for horizon, qp in result.quantile_models.items():
                qp_artifacts = qp.save(quantile_dir / f"h{horizon}")
                for k, v in qp_artifacts.items():
                    artifacts[f"quantile_h{horizon}_{k}"] = v
        else:
            qp_artifacts = result.quantile_models[0].save(quantile_dir / "stacked")
            for k, v in qp_artifacts.items():
                artifacts[f"quantile_stacked_{k}"] = v

    # Save conformal calibrators
    if result.calibrators:
        cal_dir = path / "calibrators"
        cal_dir.mkdir(exist_ok=True)

        for horizon, calibrator in result.calibrators.items():
            cal_path = cal_dir / f"calibrator_h{horizon}.json"
            calibrator.save(cal_path)
            artifacts[f"calibrator_h{horizon}"] = str(cal_path)

    # Save preprocessor params
    if result.preprocessor:
        import numpy as np

        params = result.preprocessor.get_params()
        arrays = {}
        for key in ["impute_values", "scale_center", "scale_scale", "winsorize_low", "winsorize_high"]:
            if params.get(key) is not None:
                arrays[key] = params[key]
        if params.get("config"):
            arrays["config_json"] = np.array(json.dumps(params["config"]))

        prep_path = path / "preprocess_params.npz"
        np.savez(prep_path, **arrays)
        artifacts["preprocess_params"] = str(prep_path)

    # Save manifest
    manifest = {
        "config": {
            "horizons": result.config.horizons,
            "strategy": result.config.strategy,
            "uncertainty": result.config.uncertainty,
            "quantiles": result.config.quantiles,
            "target_coverage": result.config.target_coverage,
            "model_type": result.config.model_type,
        },
        "metrics_per_horizon": {
            h: hr.metrics for h, hr in result.horizon_results.items()
        },
        "coverage_stats_per_horizon": {
            h: hr.coverage_stats for h, hr in result.horizon_results.items() if hr.coverage_stats
        },
        "artifacts": {k: Path(v).name for k, v in artifacts.items()},
    }

    manifest_path = path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    artifacts["manifest"] = str(manifest_path)

    return artifacts
