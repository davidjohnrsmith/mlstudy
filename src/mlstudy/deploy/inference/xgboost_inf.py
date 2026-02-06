"""XGBoost Booster inference with JSON/UBJ export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlstudy.deploy.inference.base import InferenceModel


class XGBoostInferenceModel(InferenceModel):
    """XGBoost Booster inference model.

    Saves/loads using XGBoost's native JSON or UBJ (universal binary JSON)
    formats for portable deployment.
    """

    model_type: str = "xgboost"

    def __init__(
        self,
        booster: Any,
        task: str = "regression",
        classes: NDArray | None = None,
        save_format: str = "json",
    ):
        """Initialize XGBoost inference model.

        Args:
            booster: XGBoost Booster object.
            task: "regression" or "classification".
            classes: Class labels for classification.
            save_format: "json" or "ubj" (universal binary JSON).
        """
        self.booster = booster
        self.task = task
        self.classes = np.asarray(classes) if classes is not None else None
        self.save_format = save_format

    def predict(self, X: NDArray) -> NDArray:
        """Make predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required for XGBoostInferenceModel") from e

        X = np.asarray(X)
        dmatrix = xgb.DMatrix(X)
        preds = self.booster.predict(dmatrix)

        if self.task == "classification":
            if preds.ndim == 1:
                # Binary classification (probabilities)
                class_preds = (preds >= 0.5).astype(int)
            else:
                # Multiclass (probability matrix)
                class_preds = np.argmax(preds, axis=1)

            if self.classes is not None:
                return self.classes[class_preds]
            return class_preds

        return preds

    def predict_proba(self, X: NDArray) -> NDArray | None:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix or None for regression.
        """
        if self.task != "classification":
            return None

        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required") from e

        X = np.asarray(X)
        dmatrix = xgb.DMatrix(X)
        preds = self.booster.predict(dmatrix)

        if preds.ndim == 1:
            # Binary classification
            return np.column_stack([1 - preds, preds])
        return preds

    def save(self, path: str | Path) -> dict[str, str]:
        """Save model artifacts.

        Saves:
        - model.json or model.ubj: XGBoost Booster
        - metadata.json: Model metadata

        Args:
            path: Directory to save artifacts.

        Returns:
            Dict mapping artifact name to file path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Save booster
        if self.save_format == "ubj":
            model_path = path / "model.ubj"
            self.booster.save_model(str(model_path))
        else:
            model_path = path / "model.json"
            self.booster.save_model(str(model_path))
        artifacts["model"] = str(model_path)

        # Save classes if present
        if self.classes is not None:
            classes_path = path / "classes.npy"
            np.save(classes_path, self.classes)
            artifacts["classes"] = str(classes_path)

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "task": self.task,
            "save_format": self.save_format,
            "has_classes": self.classes is not None,
        }
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = str(metadata_path)

        return artifacts

    @classmethod
    def load(cls, path: str | Path) -> XGBoostInferenceModel:
        """Load model from saved artifacts.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            Loaded XGBoostInferenceModel.
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required to load XGBoost models") from e

        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load booster
        save_format = metadata.get("save_format", "json")
        model_path = path / "model.ubj" if save_format == "ubj" else path / "model.json"

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        # Load classes
        classes = None
        if metadata.get("has_classes", False):
            classes = np.load(path / "classes.npy")

        return cls(
            booster=booster,
            task=metadata["task"],
            classes=classes,
            save_format=save_format,
        )

    @classmethod
    def from_sklearn(
        cls,
        model: Any,
        task: str = "regression",
        save_format: str = "json",
    ) -> XGBoostInferenceModel:
        """Create from XGBoost sklearn wrapper.

        Args:
            model: XGBRegressor or XGBClassifier.
            task: "regression" or "classification".
            save_format: "json" or "ubj".

        Returns:
            XGBoostInferenceModel instance.
        """
        booster = model.get_booster()

        classes = None
        if hasattr(model, "classes_"):
            classes = model.classes_

        return cls(
            booster=booster,
            task=task,
            classes=classes,
            save_format=save_format,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_type": self.model_type,
            "task": self.task,
            "save_format": self.save_format,
        }
