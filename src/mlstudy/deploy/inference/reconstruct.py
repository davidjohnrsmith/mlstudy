"""Reconstruct inference models from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from mlstudy.deploy.inference.base import InferenceModel
from mlstudy.deploy.inference.linear import LinearInferenceModel


def load_inference_model(path: str | Path) -> InferenceModel:
    """Load inference model from saved artifacts.

    Automatically detects model type from metadata and loads
    the appropriate inference model.

    Args:
        path: Directory containing saved artifacts.

    Returns:
        Loaded InferenceModel instance.

    Raises:
        FileNotFoundError: If metadata.json not found.
        ValueError: If model type is unknown.

    Example:
        >>> model = load_inference_model("outputs/model")
        >>> predictions = model.predict(X_test)
    """
    path = Path(path)

    # Load metadata to determine model type
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type", "unknown")

    if model_type == "linear":
        return LinearInferenceModel.load(path)

    elif model_type == "xgboost":
        from mlstudy.deploy.inference.xgboost_inf import XGBoostInferenceModel

        return XGBoostInferenceModel.load(path)

    elif model_type == "lightgbm":
        from mlstudy.deploy.inference.lightgbm_inf import LightGBMInferenceModel

        return LightGBMInferenceModel.load(path)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_preprocessor_params(path: str | Path) -> dict:
    """Load preprocessor parameters from saved artifacts.

    Args:
        path: Directory containing saved preprocessor artifacts.

    Returns:
        Dict with preprocessor parameters that can be used to
        reconstruct preprocessing in pure numpy.
    """
    import numpy as np

    path = Path(path)

    # Load config
    with open(path / "config.json") as f:
        config = json.load(f)

    params = {"config": config}

    # Load numpy arrays
    for key in ["impute_values", "scale_center", "scale_scale", "winsorize_low", "winsorize_high"]:
        arr_path = path / f"{key}.npy"
        if arr_path.exists():
            params[key] = np.load(arr_path)

    return params


class NumpyPreprocessor:
    """Pure numpy preprocessor reconstructed from saved params.

    For deployment without sklearn dependency.
    """

    def __init__(self, params: dict):
        """Initialize from loaded params.

        Args:
            params: Dict from load_preprocessor_params().
        """
        self.config = params.get("config", {})
        self.impute_values = params.get("impute_values")
        self.scale_center = params.get("scale_center")
        self.scale_scale = params.get("scale_scale")
        self.winsorize_low = params.get("winsorize_low")
        self.winsorize_high = params.get("winsorize_high")

    def transform(self, X):
        """Transform features using saved parameters.

        Args:
            X: Feature matrix.

        Returns:
            Transformed feature matrix.
        """
        import numpy as np

        X = np.asarray(X, dtype=np.float64).copy()

        # Impute
        if self.config.get("impute") != "none" and self.impute_values is not None:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                X[mask, i] = self.impute_values[i]

        # Winsorize
        if self.config.get("winsorize") and self.winsorize_low is not None:
            X = np.clip(X, self.winsorize_low, self.winsorize_high)

        # Scale
        if self.config.get("scale") != "none" and self.scale_center is not None:
            X = (X - self.scale_center) / self.scale_scale

        return X
