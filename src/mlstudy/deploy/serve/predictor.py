"""Artifact-based predictor for deployment.

Loads exported artifacts and provides inference without
sklearn or training dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ArtifactPredictor:
    """Load and serve predictions from exported artifacts.

    Supports linear, XGBoost, and LightGBM models with
    optional preprocessing.

    Example:
        >>> predictor = ArtifactPredictor.load("artifacts/run_001")
        >>> predictions = predictor.predict(X_new)
        >>> probas = predictor.predict_proba(X_new)  # For classification
    """

    def __init__(
        self,
        model_type: str,
        task: str,
        model_data: dict[str, Any],
        feature_names: list[str] | None = None,
        preprocess_params: dict[str, Any] | None = None,
        classes: NDArray | None = None,
    ):
        """Initialize predictor.

        Args:
            model_type: "linear", "xgboost", or "lightgbm".
            task: "regression" or "classification".
            model_data: Model parameters/artifacts.
            feature_names: Expected feature names in order.
            preprocess_params: Preprocessor parameters.
            classes: Class labels for classification.
        """
        self.model_type = model_type
        self.task = task
        self.model_data = model_data
        self.feature_names = feature_names or []
        self.preprocess_params = preprocess_params or {}
        self.classes = classes

        # Initialize model
        self._init_model()

    def _init_model(self) -> None:
        """Initialize model from loaded data."""
        if self.model_type == "linear":
            self._coef = np.array(self.model_data["coef"])
            self._intercept = np.array(self.model_data["intercept"])
        elif self.model_type in ("xgboost", "lightgbm"):
            self._booster = self.model_data["booster"]

    def _preprocess(self, X: NDArray) -> NDArray:
        """Apply preprocessing if configured."""
        X = np.asarray(X, dtype=np.float64).copy()

        if not self.preprocess_params:
            return X

        config = self.preprocess_params.get("config", {})

        # Impute missing values
        impute_values = self.preprocess_params.get("impute_values")
        if config.get("impute") != "none" and impute_values is not None:
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                if mask.any():
                    X[mask, i] = impute_values[i]

        # Winsorize
        winsorize_low = self.preprocess_params.get("winsorize_low")
        winsorize_high = self.preprocess_params.get("winsorize_high")
        if config.get("winsorize") and winsorize_low is not None:
            X = np.clip(X, winsorize_low, winsorize_high)

        # Scale
        scale_center = self.preprocess_params.get("scale_center")
        scale_scale = self.preprocess_params.get("scale_scale")
        if config.get("scale") != "none" and scale_center is not None:
            X = (X - scale_center) / scale_scale

        return X

    def predict(self, X: NDArray) -> NDArray:
        """Make predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        X = self._preprocess(X)

        if self.model_type == "linear":
            return self._predict_linear(X)
        elif self.model_type == "xgboost":
            return self._predict_xgboost(X)
        elif self.model_type == "lightgbm":
            return self._predict_lightgbm(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _predict_linear(self, X: NDArray) -> NDArray:
        """Predict with linear model."""
        # Compute raw scores: X @ coef.T + intercept
        raw = X @ self._coef.T + self._intercept

        if self.task == "classification":
            if raw.shape[1] == 1:
                # Binary classification
                proba = self._sigmoid(raw.ravel())
                class_preds = (proba >= 0.5).astype(int)
            else:
                # Multiclass
                proba = self._softmax(raw)
                class_preds = np.argmax(proba, axis=1)

            if self.classes is not None:
                return self.classes[class_preds]
            return class_preds

        # Regression
        return raw.ravel() if raw.shape[1] == 1 else raw

    def _predict_xgboost(self, X: NDArray) -> NDArray:
        """Predict with XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost required for XGBoost inference") from e

        dmatrix = xgb.DMatrix(X)
        preds = self._booster.predict(dmatrix)

        if self.task == "classification":
            if preds.ndim == 1:
                class_preds = (preds >= 0.5).astype(int)
            else:
                class_preds = np.argmax(preds, axis=1)

            if self.classes is not None:
                return self.classes[class_preds]
            return class_preds

        return preds

    def _predict_lightgbm(self, X: NDArray) -> NDArray:
        """Predict with LightGBM model."""
        preds = self._booster.predict(X)

        if self.task == "classification":
            if preds.ndim == 1:
                class_preds = (preds >= 0.5).astype(int)
            else:
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
            Probability matrix of shape (n_samples, n_classes),
            or None for regression.
        """
        if self.task != "classification":
            return None

        X = self._preprocess(X)

        if self.model_type == "linear":
            return self._predict_proba_linear(X)
        elif self.model_type == "xgboost":
            return self._predict_proba_xgboost(X)
        elif self.model_type == "lightgbm":
            return self._predict_proba_lightgbm(X)

        return None

    def _predict_proba_linear(self, X: NDArray) -> NDArray:
        """Predict probabilities with linear model."""
        raw = X @ self._coef.T + self._intercept

        if raw.shape[1] == 1:
            # Binary
            p1 = self._sigmoid(raw.ravel())
            return np.column_stack([1 - p1, p1])
        else:
            # Multiclass
            return self._softmax(raw)

    def _predict_proba_xgboost(self, X: NDArray) -> NDArray:
        """Predict probabilities with XGBoost."""
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X)
        preds = self._booster.predict(dmatrix)

        if preds.ndim == 1:
            return np.column_stack([1 - preds, preds])
        return preds

    def _predict_proba_lightgbm(self, X: NDArray) -> NDArray:
        """Predict probabilities with LightGBM."""
        preds = self._booster.predict(X)

        if preds.ndim == 1:
            return np.column_stack([1 - preds, preds])
        return preds

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
        x_max = x.max(axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    @classmethod
    def load(cls, path: str | Path) -> ArtifactPredictor:
        """Load predictor from artifact directory.

        Args:
            path: Artifact directory containing manifest.json.

        Returns:
            Loaded ArtifactPredictor.

        Raises:
            FileNotFoundError: If required files not found.
            ValueError: If model type unknown.
        """
        path = Path(path)

        # Load manifest
        with open(path / "manifest.json") as f:
            manifest = json.load(f)

        model_type = manifest["model_type"]
        task = manifest["task"]

        # Load feature schema
        feature_names = None
        schema_path = path / "feature_schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            feature_names = schema.get("feature_names", [])

        # Load preprocessor params
        preprocess_params = None
        npz_path = path / "preprocess_params.npz"
        if npz_path.exists():
            preprocess_params = cls._load_preprocess_params(npz_path)

        # Load model
        model_data, classes = cls._load_model(path / "model", model_type)

        return cls(
            model_type=model_type,
            task=task,
            model_data=model_data,
            feature_names=feature_names,
            preprocess_params=preprocess_params,
            classes=classes,
        )

    @classmethod
    def _load_preprocess_params(cls, npz_path: Path) -> dict[str, Any]:
        """Load preprocessor params from npz file."""
        data = np.load(npz_path, allow_pickle=True)
        params = {}

        for key in ["impute_values", "scale_center", "scale_scale", "winsorize_low", "winsorize_high"]:
            if key in data:
                params[key] = data[key]

        if "config_json" in data:
            params["config"] = json.loads(str(data["config_json"]))

        return params

    @classmethod
    def _load_model(
        cls, model_dir: Path, model_type: str
    ) -> tuple[dict[str, Any], NDArray | None]:
        """Load model from model directory.

        Returns:
            Tuple of (model_data dict, classes array or None).
        """
        classes = None
        classes_path = model_dir / "classes.npy"
        if classes_path.exists():
            classes = np.load(classes_path)

        if model_type == "linear":
            with open(model_dir / "linear.json") as f:
                model_data = json.load(f)
            if "classes" in model_data:
                classes = np.array(model_data["classes"])
            return model_data, classes

        elif model_type == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as e:
                raise ImportError("xgboost required to load XGBoost model") from e

            booster = xgb.Booster()
            model_path = model_dir / "xgb.json"
            if not model_path.exists():
                model_path = model_dir / "xgb.ubj"
            booster.load_model(str(model_path))
            return {"booster": booster}, classes

        elif model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as e:
                raise ImportError("lightgbm required to load LightGBM model") from e

            booster = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
            return {"booster": booster}, classes

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @property
    def n_features(self) -> int:
        """Number of expected features."""
        return len(self.feature_names) if self.feature_names else 0

    @property
    def metadata(self) -> dict[str, Any]:
        """Return predictor metadata."""
        return {
            "model_type": self.model_type,
            "task": self.task,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "has_preprocessor": bool(self.preprocess_params),
        }
