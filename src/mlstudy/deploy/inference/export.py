"""Export models to deployable inference artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlstudy.deploy.inference.linear import LinearInferenceModel


def get_model_type(model: Any) -> str:
    """Determine model type from sklearn or other model object.

    Args:
        model: Trained model object.

    Returns:
        Model type string: "linear", "xgboost", "lightgbm", or "unknown".
    """
    model_class = type(model).__name__

    # Linear models
    linear_classes = {
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LogisticRegression",
        "SGDClassifier",
        "SGDRegressor",
    }
    if model_class in linear_classes:
        return "linear"

    # XGBoost
    if model_class in {"XGBRegressor", "XGBClassifier", "Booster"}:
        return "xgboost"

    # LightGBM
    if model_class in {"LGBMRegressor", "LGBMClassifier", "Booster"}:
        # Check if it's actually LightGBM Booster
        module = type(model).__module__
        if "lightgbm" in module:
            return "lightgbm"
        if "xgboost" in module:
            return "xgboost"

    return "unknown"


def export_model(
    model: Any,
    path: str | Path,
    task: str = "regression",
    model_type: str | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Export model to deployable inference artifacts.

    Args:
        model: Trained model object (sklearn, xgboost, or lightgbm).
        path: Directory to save artifacts.
        task: "regression" or "classification".
        model_type: Override auto-detected model type.
        **kwargs: Additional arguments passed to inference model.

    Returns:
        Dict mapping artifact name to file path.

    Raises:
        ValueError: If model type is unknown or unsupported.

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> model = Ridge().fit(X_train, y_train)
        >>> artifacts = export_model(model, "outputs/model", task="regression")
    """
    if model_type is None:
        model_type = get_model_type(model)

    if model_type == "linear":
        inference_model = LinearInferenceModel.from_sklearn(model, task=task)
        return inference_model.save(path)

    elif model_type == "xgboost":
        from mlstudy.deploy.inference.xgboost_inf import XGBoostInferenceModel

        save_format = kwargs.get("save_format", "json")
        inference_model = XGBoostInferenceModel.from_sklearn(
            model, task=task, save_format=save_format
        )
        return inference_model.save(path)

    elif model_type == "lightgbm":
        from mlstudy.deploy.inference.lightgbm_inf import LightGBMInferenceModel

        inference_model = LightGBMInferenceModel.from_sklearn(model, task=task)
        return inference_model.save(path)

    else:
        raise ValueError(
            f"Unknown or unsupported model type: {model_type}. "
            f"Detected from: {type(model).__name__}"
        )


def export_preprocessor(
    preprocessor: Any,
    path: str | Path,
) -> dict[str, str]:
    """Export preprocessor for deployment.

    Args:
        preprocessor: Fitted Preprocessor object.
        path: Directory to save artifacts.

    Returns:
        Dict mapping artifact name to file path.
    """
    import json

    import numpy as np

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    params = preprocessor.get_params()

    # Save numpy arrays
    for key in ["impute_values", "scale_center", "scale_scale", "winsorize_low", "winsorize_high"]:
        if params.get(key) is not None:
            arr_path = path / f"{key}.npy"
            np.save(arr_path, params[key])
            artifacts[key] = str(arr_path)

    # Save config as JSON
    config = params.get("config", {})
    config_path = path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    artifacts["config"] = str(config_path)

    return artifacts
