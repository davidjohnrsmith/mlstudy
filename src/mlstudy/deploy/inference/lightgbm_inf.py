"""LightGBM Booster inference with text file export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlstudy.deploy.inference.base import InferenceModel


class LightGBMInferenceModel(InferenceModel):
    """LightGBM Booster inference model.

    Saves/loads using LightGBM's native text format for
    portable deployment.
    """

    model_type: str = "lightgbm"

    def __init__(
        self,
        booster: Any,
        task: str = "regression",
        classes: NDArray | None = None,
        num_classes: int | None = None,
    ):
        """Initialize LightGBM inference model.

        Args:
            booster: LightGBM Booster object.
            task: "regression" or "classification".
            classes: Class labels for classification.
            num_classes: Number of classes (for multiclass).
        """
        self.booster = booster
        self.task = task
        self.classes = np.asarray(classes) if classes is not None else None
        self.num_classes = num_classes

    def predict(self, X: NDArray) -> NDArray:
        """Make predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        X = np.asarray(X)
        preds = self.booster.predict(X)

        if self.task == "classification":
            if preds.ndim == 1:
                # Binary classification (probabilities or raw scores)
                # LightGBM returns probabilities for binary with objective=binary
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

        X = np.asarray(X)
        preds = self.booster.predict(X)

        if preds.ndim == 1:
            # Binary classification
            return np.column_stack([1 - preds, preds])
        return preds

    def save(self, path: str | Path) -> dict[str, str]:
        """Save model artifacts.

        Saves:
        - model.txt: LightGBM Booster in text format
        - metadata.json: Model metadata

        Args:
            path: Directory to save artifacts.

        Returns:
            Dict mapping artifact name to file path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Save booster as text file
        model_path = path / "model.txt"
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
            "has_classes": self.classes is not None,
            "num_classes": self.num_classes,
        }
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = str(metadata_path)

        return artifacts

    @classmethod
    def load(cls, path: str | Path) -> LightGBMInferenceModel:
        """Load model from saved artifacts.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            Loaded LightGBMInferenceModel.
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm is required to load LightGBM models") from e

        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load booster from text file
        model_path = path / "model.txt"
        booster = lgb.Booster(model_file=str(model_path))

        # Load classes
        classes = None
        if metadata.get("has_classes", False):
            classes = np.load(path / "classes.npy")

        return cls(
            booster=booster,
            task=metadata["task"],
            classes=classes,
            num_classes=metadata.get("num_classes"),
        )

    @classmethod
    def from_sklearn(cls, model: Any, task: str = "regression") -> LightGBMInferenceModel:
        """Create from LightGBM sklearn wrapper.

        Args:
            model: LGBMRegressor or LGBMClassifier.
            task: "regression" or "classification".

        Returns:
            LightGBMInferenceModel instance.
        """
        booster = model.booster_

        classes = None
        num_classes = None
        if hasattr(model, "classes_"):
            classes = model.classes_
            num_classes = len(classes)

        return cls(
            booster=booster,
            task=task,
            classes=classes,
            num_classes=num_classes,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_type": self.model_type,
            "task": self.task,
            "num_classes": self.num_classes,
        }
