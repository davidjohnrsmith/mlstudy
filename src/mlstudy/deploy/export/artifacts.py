"""Export models to deployable artifact bundles.

Artifact format:
    artifacts/<run_id>/
        manifest.json          # Metadata, model type, feature schema path, etc.
        feature_schema.json    # Feature names, dtypes, order
        preprocess_params.npz  # Preprocessor parameters (impute values, scales, etc.)
        model/
            linear.json        # Linear model coefficients (if linear)
            xgb.json           # XGBoost model (if xgboost)
            lgbm.txt           # LightGBM model (if lightgbm)
        code_version.txt       # Git commit hash or version string

No pickled Python objects are used - all artifacts are portable.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def get_code_version() -> str:
    """Get current git commit hash or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


class ArtifactExporter:
    """Export trained models to deployable artifact bundles.

    Creates a self-contained directory with all artifacts needed
    for inference without sklearn or training dependencies.

    Example:
        >>> exporter = ArtifactExporter(
        ...     model=trained_model,
        ...     preprocessor=fitted_preprocessor,
        ...     feature_names=feature_cols,
        ...     task="regression",
        ... )
        >>> exporter.export("artifacts/run_001")
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any | None = None,
        feature_names: list[str] | None = None,
        feature_dtypes: dict[str, str] | None = None,
        task: str = "regression",
        model_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize artifact exporter.

        Args:
            model: Trained model (sklearn, xgboost, or lightgbm).
            preprocessor: Fitted Preprocessor object (optional).
            feature_names: List of feature names in order.
            feature_dtypes: Dict mapping feature name to dtype string.
            task: "regression" or "classification".
            model_type: Override auto-detected model type.
            metadata: Additional metadata to include in manifest.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names or []
        self.feature_dtypes = feature_dtypes or {}
        self.task = task
        self.model_type = model_type or self._detect_model_type()
        self.metadata = metadata or {}

    def _detect_model_type(self) -> str:
        """Detect model type from model object."""
        model_class = type(self.model).__name__

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

        if model_class in {"XGBRegressor", "XGBClassifier", "Booster"}:
            module = type(self.model).__module__
            if "xgboost" in module:
                return "xgboost"

        if model_class in {"LGBMRegressor", "LGBMClassifier", "Booster"}:
            module = type(self.model).__module__
            if "lightgbm" in module:
                return "lightgbm"

        return "unknown"

    def export(self, path: str | Path) -> dict[str, str]:
        """Export all artifacts to directory.

        Args:
            path: Directory to save artifacts.

        Returns:
            Dict mapping artifact name to file path.

        Raises:
            ValueError: If model type is unknown.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifacts = {}

        # Export model
        model_artifacts = self._export_model(path / "model")
        artifacts.update(model_artifacts)

        # Export feature schema
        schema_path = self._export_feature_schema(path)
        artifacts["feature_schema"] = str(schema_path)

        # Export preprocessor params
        if self.preprocessor is not None:
            preprocess_path = self._export_preprocessor(path)
            artifacts["preprocess_params"] = str(preprocess_path)

        # Export code version
        version_path = path / "code_version.txt"
        version_path.write_text(get_code_version())
        artifacts["code_version"] = str(version_path)

        # Export manifest
        manifest_path = self._export_manifest(path, artifacts)
        artifacts["manifest"] = str(manifest_path)

        return artifacts

    def _export_model(self, model_dir: Path) -> dict[str, str]:
        """Export model to model/ subdirectory."""
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type == "linear":
            return self._export_linear_model(model_dir)
        elif self.model_type == "xgboost":
            return self._export_xgboost_model(model_dir)
        elif self.model_type == "lightgbm":
            return self._export_lightgbm_model(model_dir)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _export_linear_model(self, model_dir: Path) -> dict[str, str]:
        """Export linear model coefficients to JSON."""
        coef = self.model.coef_
        intercept = self.model.intercept_

        # Handle multiclass (2D coef)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if np.isscalar(intercept):
            intercept = np.array([intercept])

        model_data = {
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
            "task": self.task,
        }

        # Save classes for classification
        if hasattr(self.model, "classes_"):
            model_data["classes"] = self.model.classes_.tolist()

        model_path = model_dir / "linear.json"
        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)

        return {"model": str(model_path)}

    def _export_xgboost_model(self, model_dir: Path) -> dict[str, str]:
        """Export XGBoost model to JSON."""
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model

        model_path = model_dir / "xgb.json"
        booster.save_model(str(model_path))

        artifacts = {"model": str(model_path)}

        # Save classes
        if hasattr(self.model, "classes_"):
            classes_path = model_dir / "classes.npy"
            np.save(classes_path, self.model.classes_)
            artifacts["classes"] = str(classes_path)

        return artifacts

    def _export_lightgbm_model(self, model_dir: Path) -> dict[str, str]:
        """Export LightGBM model to text file."""
        booster = self.model.booster_ if hasattr(self.model, "booster_") else self.model

        model_path = model_dir / "lgbm.txt"
        booster.save_model(str(model_path))

        artifacts = {"model": str(model_path)}

        # Save classes
        if hasattr(self.model, "classes_"):
            classes_path = model_dir / "classes.npy"
            np.save(classes_path, self.model.classes_)
            artifacts["classes"] = str(classes_path)

        return artifacts

    def _export_feature_schema(self, path: Path) -> Path:
        """Export feature schema to JSON."""
        schema = {
            "feature_names": self.feature_names,
            "feature_dtypes": self.feature_dtypes,
            "n_features": len(self.feature_names),
        }

        schema_path = path / "feature_schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        return schema_path

    def _export_preprocessor(self, path: Path) -> Path:
        """Export preprocessor parameters to npz."""
        params = self.preprocessor.get_params()

        # Collect numpy arrays and config
        arrays = {}
        for key in ["impute_values", "scale_center", "scale_scale", "winsorize_low", "winsorize_high"]:
            if params.get(key) is not None:
                arrays[key] = params[key]

        # Save config as JSON string in npz
        config = params.get("config", {})
        arrays["config_json"] = np.array(json.dumps(config))

        preprocess_path = path / "preprocess_params.npz"
        np.savez(preprocess_path, **arrays)

        return preprocess_path

    def _export_manifest(self, path: Path, artifacts: dict[str, str]) -> Path:
        """Export manifest with metadata."""
        manifest = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "model_type": self.model_type,
            "task": self.task,
            "n_features": len(self.feature_names),
            "has_preprocessor": self.preprocessor is not None,
            "artifacts": {k: Path(v).name for k, v in artifacts.items()},
            "code_version": get_code_version(),
            **self.metadata,
        }

        manifest_path = path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest_path


def export_artifact(
    model: Any,
    path: str | Path,
    preprocessor: Any | None = None,
    feature_names: list[str] | None = None,
    feature_dtypes: dict[str, str] | None = None,
    task: str = "regression",
    model_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Export model to deployable artifact bundle.

    Convenience function wrapping ArtifactExporter.

    Args:
        model: Trained model.
        path: Directory to save artifacts.
        preprocessor: Fitted preprocessor (optional).
        feature_names: Feature names in order.
        feature_dtypes: Feature dtype mapping.
        task: "regression" or "classification".
        model_type: Override auto-detected model type.
        metadata: Additional manifest metadata.

    Returns:
        Dict mapping artifact name to file path.

    Example:
        >>> artifacts = export_artifact(
        ...     model=ridge_model,
        ...     path="artifacts/run_001",
        ...     preprocessor=prep,
        ...     feature_names=feature_cols,
        ...     task="regression",
        ... )
    """
    exporter = ArtifactExporter(
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        task=task,
        model_type=model_type,
        metadata=metadata,
    )
    return exporter.export(path)


def load_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Load manifest from artifact directory.

    Args:
        path: Artifact directory.

    Returns:
        Manifest dict.

    Raises:
        FileNotFoundError: If manifest.json not found.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {path}")

    with open(manifest_path) as f:
        return json.load(f)


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    """Load feature schema from artifact directory.

    Args:
        path: Artifact directory.

    Returns:
        Feature schema dict with feature_names, feature_dtypes, n_features.
    """
    path = Path(path)
    schema_path = path / "feature_schema.json"

    with open(schema_path) as f:
        return json.load(f)


def load_preprocess_params(path: str | Path) -> dict[str, Any]:
    """Load preprocessor parameters from artifact directory.

    Args:
        path: Artifact directory.

    Returns:
        Dict with preprocessor arrays and config.
    """
    path = Path(path)
    npz_path = path / "preprocess_params.npz"

    if not npz_path.exists():
        return {}

    data = np.load(npz_path, allow_pickle=True)
    params = {}

    for key in ["impute_values", "scale_center", "scale_scale", "winsorize_low", "winsorize_high"]:
        if key in data:
            params[key] = data[key]

    if "config_json" in data:
        params["config"] = json.loads(str(data["config_json"]))

    return params
