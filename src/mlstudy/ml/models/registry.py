"""Model registry for building sklearn estimators."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from typing_extensions import Literal  # noqa: UP035

# Type aliases
Task = Literal["regression", "classification"]
ModelName = Literal[
    "linear",
    "ridge",
    "rf",
    "random_forest",
    "hgb",
    "hist_gradient_boosting",
    "logistic",
    "lgbm",
    "lightgbm",
]

# Model registry mapping
_REGRESSION_MODELS: dict[str, type] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "rf": RandomForestRegressor,
    "random_forest": RandomForestRegressor,
    "hgb": HistGradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}

_CLASSIFICATION_MODELS: dict[str, type] = {
    "logistic": LogisticRegression,
    "rf": RandomForestClassifier,
    "random_forest": RandomForestClassifier,
    "hgb": HistGradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

# Default hyperparameters per model
_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 1.0},
    "rf": {"n_estimators": 100, "max_depth": 10, "n_jobs": -1},
    "random_forest": {"n_estimators": 100, "max_depth": 10, "n_jobs": -1},
    "hgb": {"max_iter": 100, "max_depth": 6, "learning_rate": 0.1},
    "hist_gradient_boosting": {"max_iter": 100, "max_depth": 6, "learning_rate": 0.1},
    "logistic": {"max_iter": 1000, "solver": "lbfgs"},
    "lgbm": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "verbose": -1},
    "lightgbm": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "verbose": -1},
}


def _get_lgbm_class(task: Task) -> type:
    """Get LightGBM class, importing lazily."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm is required for lgbm models. "
            "Install with: pip install lightgbm"
        ) from e

    if task == "regression":
        return lgb.LGBMRegressor
    else:
        return lgb.LGBMClassifier


def build_model(
    model_name: str,
    task: Task,
    random_state: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build a sklearn estimator by name.

    Args:
        model_name: Name of the model. Options:
            - Regression: "linear", "ridge", "rf"/"random_forest",
              "hgb"/"hist_gradient_boosting", "lgbm"/"lightgbm"
            - Classification: "logistic", "rf"/"random_forest",
              "hgb"/"hist_gradient_boosting", "lgbm"/"lightgbm"
        task: Either "regression" or "classification".
        random_state: Random seed for reproducibility.
        **kwargs: Additional hyperparameters to override defaults.

    Returns:
        Instantiated sklearn estimator.

    Raises:
        ValueError: If model_name is not valid for the given task.
    """
    model_name = model_name.lower()

    # Handle LightGBM specially (lazy import)
    if model_name in ("lgbm", "lightgbm"):
        model_cls = _get_lgbm_class(task)
    elif task == "regression":
        if model_name not in _REGRESSION_MODELS:
            valid = list(_REGRESSION_MODELS.keys()) + ["lgbm", "lightgbm"]
            raise ValueError(
                f"Unknown regression model: {model_name}. Valid options: {valid}"
            )
        model_cls = _REGRESSION_MODELS[model_name]
    elif task == "classification":
        if model_name not in _CLASSIFICATION_MODELS:
            valid = list(_CLASSIFICATION_MODELS.keys()) + ["lgbm", "lightgbm"]
            raise ValueError(
                f"Unknown classification model: {model_name}. Valid options: {valid}"
            )
        model_cls = _CLASSIFICATION_MODELS[model_name]
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")

    # Build kwargs with defaults
    model_kwargs = _MODEL_DEFAULTS.get(model_name, {}).copy()
    model_kwargs.update(kwargs)

    # Add random_state if the model supports it
    if random_state is not None and model_name not in ("linear",):
        model_kwargs["random_state"] = random_state

    return model_cls(**model_kwargs)


def build_quantile_model(
    quantile: float,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build LightGBM model for quantile regression.

    Args:
        quantile: Target quantile (0-1), e.g., 0.1, 0.5, 0.9.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        random_state: Random seed.
        **kwargs: Additional LightGBM parameters.

    Returns:
        LGBMRegressor configured for quantile regression.

    Example:
        >>> model_q10 = build_quantile_model(0.1)
        >>> model_q90 = build_quantile_model(0.9)
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm required for quantile regression") from e

    model_kwargs = {
        "objective": "quantile",
        "alpha": quantile,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "verbose": -1,
    }
    model_kwargs.update(kwargs)

    if random_state is not None:
        model_kwargs["random_state"] = random_state

    return lgb.LGBMRegressor(**model_kwargs)


def list_models(task: Task) -> list[str]:
    """List available model names for a task.

    Args:
        task: Either "regression" or "classification".

    Returns:
        List of valid model names.
    """
    if task == "regression":
        return list(_REGRESSION_MODELS.keys()) + ["lgbm", "lightgbm"]
    elif task == "classification":
        return list(_CLASSIFICATION_MODELS.keys()) + ["lgbm", "lightgbm"]
    else:
        raise ValueError(f"Unknown task: {task}")
