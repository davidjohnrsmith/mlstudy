"""Evaluation metrics for regression and classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def spearman_ic(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Spearman Information Coefficient (rank correlation).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Spearman correlation coefficient.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr)


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dict with keys: mae, mse, rmse, r2, spearman_ic
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    ic = spearman_ic(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman_ic": float(ic),
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute classification evaluation metrics.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        y_prob: Optional predicted probabilities for positive class (for AUC).

    Returns:
        Dict with keys: accuracy, f1, roc_auc (if y_prob provided)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    acc = accuracy_score(y_true, y_pred)

    # Use weighted F1 for multiclass, binary for binary
    n_classes = len(np.unique(y_true))
    if n_classes > 2:
        f1 = f1_score(y_true, y_pred, average="weighted")
    else:
        f1 = f1_score(y_true, y_pred, average="binary")

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
    }

    # Compute ROC-AUC if probabilities provided
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if n_classes == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
                auc = roc_auc_score(y_true, y_prob)
            else:
                # Multiclass: use OvR
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            metrics["roc_auc"] = float(auc)
        except ValueError:
            # AUC undefined (e.g., only one class in y_true)
            metrics["roc_auc"] = np.nan

    return metrics
