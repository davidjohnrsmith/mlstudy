"""Base classes for inference models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from numpy.typing import NDArray


class InferenceModel(ABC):
    """Abstract base class for inference models.

    All inference models must implement predict() and provide
    save/load functionality for portable deployment.
    """

    model_type: str = "base"

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Make predictions on input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs).
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> dict[str, str]:
        """Save model artifacts to disk.

        Args:
            path: Directory to save artifacts.

        Returns:
            Dict mapping artifact name to file path.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> InferenceModel:
        """Load model from saved artifacts.

        Args:
            path: Directory containing saved artifacts.

        Returns:
            Loaded inference model.
        """
        pass

    def predict_proba(self, X: NDArray) -> NDArray | None:
        """Predict class probabilities (for classifiers).

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix or None if not applicable.
        """
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {"model_type": self.model_type}
