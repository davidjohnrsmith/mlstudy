"""Linear model inference with pure numpy (no sklearn required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlstudy.deploy.inference.base import InferenceModel


class LinearInferenceModel(InferenceModel):
    """Pure numpy inference for linear models.

    Supports:
    - LinearRegression
    - Ridge
    - LogisticRegression (binary and multiclass)

    Exports coefficients and intercept as numpy arrays for
    deployment without sklearn dependency.
    """

    model_type: str = "linear"

    def __init__(
        self,
        coef: NDArray,
        intercept: NDArray | float,
        task: str = "regression",
        classes: NDArray | None = None,
    ):
        """Initialize linear inference model.

        Args:
            coef: Coefficient array of shape (n_features,) for regression
                  or (n_classes, n_features) for multiclass.
            intercept: Intercept value(s).
            task: "regression" or "classification".
            classes: Class labels for classification.
        """
        self.coef = np.asarray(coef)
        self.intercept = np.asarray(intercept)
        self.task = task
        self.classes = np.asarray(classes) if classes is not None else None

        # Ensure proper shapes
        if self.coef.ndim == 1:
            self.coef = self.coef.reshape(1, -1)
        if self.intercept.ndim == 0:
            self.intercept = self.intercept.reshape(1)

    def predict(self, X: NDArray) -> NDArray:
        """Make predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        X = np.asarray(X)

        # Linear combination: X @ coef.T + intercept
        scores = X @ self.coef.T + self.intercept

        if self.task == "regression":
            return scores.ravel()
        else:
            # Classification
            if self.coef.shape[0] == 1:
                # Binary classification
                probs = self._sigmoid(scores.ravel())
                return (probs >= 0.5).astype(int)
            else:
                # Multiclass (softmax)
                probs = self._softmax(scores)
                class_indices = np.argmax(probs, axis=1)
                if self.classes is not None:
                    return self.classes[class_indices]
                return class_indices

    def predict_proba(self, X: NDArray) -> NDArray | None:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        if self.task != "classification":
            return None

        X = np.asarray(X)
        scores = X @ self.coef.T + self.intercept

        if self.coef.shape[0] == 1:
            # Binary classification
            prob_pos = self._sigmoid(scores.ravel())
            return np.column_stack([1 - prob_pos, prob_pos])
        else:
            # Multiclass
            return self._softmax(scores)

    @staticmethod
    def _sigmoid(x: NDArray) -> NDArray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    @staticmethod
    def _softmax(x: NDArray) -> NDArray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> dict[str, str]:
        """Save model artifacts.

        Saves:
        - coef.npy: Coefficient array
        - intercept.npy: Intercept array
        - metadata.json: Model metadata

        Args:
            path: Directory to save artifacts.

        Returns:
            Dict mapping artifact name to file path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Save coefficients
        coef_path = path / "coef.npy"
        np.save(coef_path, self.coef)
        artifacts["coef"] = str(coef_path)

        # Save intercept
        intercept_path = path / "intercept.npy"
        np.save(intercept_path, self.intercept)
        artifacts["intercept"] = str(intercept_path)

        # Save classes if present
        if self.classes is not None:
            classes_path = path / "classes.npy"
            np.save(classes_path, self.classes)
            artifacts["classes"] = str(classes_path)

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "task": self.task,
            "n_features": int(self.coef.shape[1]),
            "n_classes": int(self.coef.shape[0]),
            "has_classes": self.classes is not None,
        }
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = str(metadata_path)

        return artifacts

    @classmethod
    def load(cls, path: str | Path) -> LinearInferenceModel:
        """Load model from saved artifacts.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            Loaded LinearInferenceModel.
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load arrays
        coef = np.load(path / "coef.npy")
        intercept = np.load(path / "intercept.npy")

        classes = None
        if metadata.get("has_classes", False):
            classes = np.load(path / "classes.npy")

        return cls(
            coef=coef,
            intercept=intercept,
            task=metadata["task"],
            classes=classes,
        )

    @classmethod
    def from_sklearn(cls, model: Any, task: str = "regression") -> LinearInferenceModel:
        """Create from sklearn linear model.

        Args:
            model: Sklearn LinearRegression, Ridge, or LogisticRegression.
            task: "regression" or "classification".

        Returns:
            LinearInferenceModel instance.
        """
        coef = model.coef_
        intercept = model.intercept_

        classes = None
        if hasattr(model, "classes_"):
            classes = model.classes_

        return cls(coef=coef, intercept=intercept, task=task, classes=classes)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_type": self.model_type,
            "task": self.task,
            "n_features": int(self.coef.shape[1]),
            "n_classes": int(self.coef.shape[0]),
        }
