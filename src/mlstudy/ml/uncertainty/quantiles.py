"""Quantile regression for prediction intervals.

Provides LightGBM-based quantile regression for generating
prediction intervals (e.g., q10/q50/q90).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from numpy.typing import NDArray


def build_quantile_lgbm(
    quantile: float,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_samples: int = 20,
    **kwargs: Any,
) -> Any:
    """Build LightGBM model for quantile regression.

    Args:
        quantile: Target quantile (0-1), e.g., 0.1, 0.5, 0.9.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        min_child_samples: Minimum samples per leaf.
        **kwargs: Additional LightGBM parameters.

    Returns:
        LGBMRegressor configured for quantile regression.

    Example:
        >>> model_q10 = build_quantile_lgbm(0.1)
        >>> model_q90 = build_quantile_lgbm(0.9)
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm required for quantile regression") from e

    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        verbose=-1,
        **kwargs,
    )


class QuantilePredictor:
    """Train and predict with multiple quantile models.

    Trains separate LightGBM models for each quantile to produce
    prediction intervals.

    Example:
        >>> predictor = QuantilePredictor(quantiles=[0.1, 0.5, 0.9])
        >>> predictor.fit(X_train, y_train)
        >>> intervals = predictor.predict(X_test)
        >>> # intervals["q50"] is median prediction
        >>> # intervals["q10"], intervals["q90"] form 80% interval
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
        **lgbm_kwargs: Any,
    ):
        """Initialize quantile predictor.

        Args:
            quantiles: List of quantiles to predict. Default [0.1, 0.5, 0.9].
            n_estimators: Number of boosting rounds per model.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            min_child_samples: Minimum samples per leaf.
            **lgbm_kwargs: Additional LightGBM parameters.
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.lgbm_kwargs = lgbm_kwargs

        self.models: dict[float, Any] = {}
        self._fitted = False

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        eval_set: tuple[NDArray, NDArray] | None = None,
    ) -> QuantilePredictor:
        """Fit quantile models.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,).
            eval_set: Optional (X_val, y_val) for early stopping.

        Returns:
            Self for chaining.
        """
        for q in self.quantiles:
            model = build_quantile_lgbm(
                quantile=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_samples=self.min_child_samples,
                **self.lgbm_kwargs,
            )

            fit_kwargs: dict[str, Any] = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [eval_set]

            model.fit(X, y, **fit_kwargs)
            self.models[q] = model

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> dict[str, NDArray]:
        """Predict all quantiles.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Dict mapping quantile names (e.g., "q10") to predictions.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        results = {}
        for q, model in self.models.items():
            key = f"q{int(q * 100)}"
            results[key] = model.predict(X)

        return results

    def predict_intervals(
        self,
        X: NDArray,
        lower_q: float = 0.1,
        upper_q: float = 0.9,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Predict with intervals.

        Args:
            X: Features.
            lower_q: Lower quantile for interval.
            upper_q: Upper quantile for interval.

        Returns:
            Tuple of (lower_bound, point_estimate, upper_bound).
        """
        preds = self.predict(X)

        lower_key = f"q{int(lower_q * 100)}"
        upper_key = f"q{int(upper_q * 100)}"
        median_key = "q50"

        lower = preds.get(lower_key, preds[list(preds.keys())[0]])
        upper = preds.get(upper_key, preds[list(preds.keys())[-1]])
        point = preds.get(median_key, (lower + upper) / 2)

        return lower, point, upper

    def save(self, path: str | Path, prefix: str = "quantile") -> dict[str, str]:
        """Save all quantile models.

        Args:
            path: Directory to save models.
            prefix: Filename prefix.

        Returns:
            Dict mapping quantile to file path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifacts = {}
        for q, model in self.models.items():
            q_name = f"q{int(q * 100)}"
            model_path = path / f"{prefix}_{q_name}.txt"
            model.booster_.save_model(str(model_path))
            artifacts[q_name] = str(model_path)

        return artifacts

    @classmethod
    def load(cls, path: str | Path, quantiles: list[float], prefix: str = "quantile") -> QuantilePredictor:
        """Load quantile models from saved files.

        Args:
            path: Directory containing saved models.
            quantiles: Quantiles that were trained.
            prefix: Filename prefix used when saving.

        Returns:
            Loaded QuantilePredictor.
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm required to load quantile models") from e

        path = Path(path)
        predictor = cls(quantiles=quantiles)

        for q in quantiles:
            q_name = f"q{int(q * 100)}"
            model_path = path / f"{prefix}_{q_name}.txt"
            booster = lgb.Booster(model_file=str(model_path))
            predictor.models[q] = booster

        predictor._fitted = True
        return predictor

    @property
    def is_fitted(self) -> bool:
        """Whether models have been fitted."""
        return self._fitted


def predict_intervals(
    models: dict[str, Any],
    X: NDArray,
    lower_key: str = "q10",
    upper_key: str = "q90",
    point_key: str = "q50",
) -> tuple[NDArray, NDArray, NDArray]:
    """Predict intervals from pre-trained quantile models.

    Args:
        models: Dict mapping quantile key to model.
        X: Features.
        lower_key: Key for lower bound model.
        upper_key: Key for upper bound model.
        point_key: Key for point estimate model.

    Returns:
        Tuple of (lower_bound, point_estimate, upper_bound).
    """
    lower = models[lower_key].predict(X)
    upper = models[upper_key].predict(X)
    point = models[point_key].predict(X)

    return lower, point, upper
