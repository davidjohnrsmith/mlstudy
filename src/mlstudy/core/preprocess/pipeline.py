"""Preprocessing pipeline with train-only fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Literal  # noqa: UP035


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline.

    Attributes:
        impute: Imputation strategy. "none" keeps NaN, "median" uses column
            median, "mean" uses column mean.
        scale: Scaling strategy. "none" keeps original scale, "standard"
            uses z-score normalization, "robust" uses median/IQR scaling.
        winsorize: Optional percentile for winsorization. If provided, clips
            values outside [winsorize, 100-winsorize] percentiles.
            E.g., winsorize=0.01 clips at 1st and 99th percentiles.
    """

    impute: Literal["none", "median", "mean"] = "none"
    scale: Literal["none", "standard", "robust"] = "none"
    winsorize: float | None = None


class Preprocessor:
    """Preprocessing transformer that fits on train data only.

    Computes statistics (mean, median, std, etc.) from training data
    and applies the same transformation to validation/test data.

    Example:
        >>> config = PreprocessConfig(impute="median", scale="standard")
        >>> preprocessor = Preprocessor(config)
        >>> preprocessor.fit(X_train)
        >>> X_train_transformed = preprocessor.transform(X_train)
        >>> X_val_transformed = preprocessor.transform(X_val)
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """Initialize preprocessor with configuration.

        Args:
            config: PreprocessConfig specifying transformation options.
        """
        self.config = config
        self._fitted = False

        # Statistics computed during fit
        self._impute_values: NDArray | None = None
        self._scale_center: NDArray | None = None
        self._scale_scale: NDArray | None = None
        self._winsorize_low: NDArray | None = None
        self._winsorize_high: NDArray | None = None

    def fit(self, X: NDArray | pd.DataFrame) -> Preprocessor:
        """Fit preprocessor on training data.

        Computes all necessary statistics from the training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).

        Returns:
            Self for method chaining.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]

        # Compute imputation values
        if self.config.impute == "median":
            self._impute_values = np.nanmedian(X, axis=0)
        elif self.config.impute == "mean":
            self._impute_values = np.nanmean(X, axis=0)
        else:
            self._impute_values = np.zeros(n_features)

        # For scaling, we need imputed data
        X_imputed = X.copy()
        if self.config.impute != "none":
            for i in range(n_features):
                mask = np.isnan(X_imputed[:, i])
                X_imputed[mask, i] = self._impute_values[i]

        # Compute scaling parameters
        if self.config.scale == "standard":
            self._scale_center = np.nanmean(X_imputed, axis=0)
            self._scale_scale = np.nanstd(X_imputed, axis=0)
            # Avoid division by zero
            self._scale_scale[self._scale_scale == 0] = 1.0
        elif self.config.scale == "robust":
            self._scale_center = np.nanmedian(X_imputed, axis=0)
            q75 = np.nanpercentile(X_imputed, 75, axis=0)
            q25 = np.nanpercentile(X_imputed, 25, axis=0)
            self._scale_scale = q75 - q25
            # Avoid division by zero
            self._scale_scale[self._scale_scale == 0] = 1.0
        else:
            self._scale_center = np.zeros(n_features)
            self._scale_scale = np.ones(n_features)

        # Compute winsorization bounds
        if self.config.winsorize is not None:
            pct = self.config.winsorize * 100
            self._winsorize_low = np.nanpercentile(X_imputed, pct, axis=0)
            self._winsorize_high = np.nanpercentile(X_imputed, 100 - pct, axis=0)

        self._fitted = True
        return self

    def transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """Transform data using fitted statistics.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Transformed feature matrix.

        Raises:
            RuntimeError: If transform is called before fit.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=np.float64).copy()
        n_features = X.shape[1]

        # Apply imputation
        if self.config.impute != "none" and self._impute_values is not None:
            for i in range(n_features):
                mask = np.isnan(X[:, i])
                X[mask, i] = self._impute_values[i]

        # Apply winsorization (before scaling)
        if (
            self.config.winsorize is not None
            and self._winsorize_low is not None
            and self._winsorize_high is not None
        ):
            X = np.clip(X, self._winsorize_low, self._winsorize_high)

        # Apply scaling
        if (
            self.config.scale != "none"
            and self._scale_center is not None
            and self._scale_scale is not None
        ):
            X = (X - self._scale_center) / self._scale_scale

        return X

    def fit_transform(self, X: NDArray | pd.DataFrame) -> NDArray:
        """Fit and transform in one step.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).

        Returns:
            Transformed feature matrix.
        """
        return self.fit(X).transform(X)

    @property
    def is_fitted(self) -> bool:
        """Whether the preprocessor has been fitted."""
        return self._fitted

    def get_params(self) -> dict:
        """Get fitted parameters for serialization."""
        return {
            "config": {
                "impute": self.config.impute,
                "scale": self.config.scale,
                "winsorize": self.config.winsorize,
            },
            "impute_values": self._impute_values,
            "scale_center": self._scale_center,
            "scale_scale": self._scale_scale,
            "winsorize_low": self._winsorize_low,
            "winsorize_high": self._winsorize_high,
        }
