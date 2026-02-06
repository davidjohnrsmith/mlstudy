"""Feature registry with decorator-based registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlstudy.core.features.base import FeatureInfo

# Global registry
_FEATURE_REGISTRY: dict[str, FeatureInfo] = {}


def register_feature(
    name: str,
    required_cols: list[str],
    output_cols_fn: Callable[[dict[str, Any]], list[str]] | None = None,
    description: str = "",
) -> Callable:
    """Decorator to register a feature function.

    Args:
        name: Unique name for the feature.
        required_cols: List of required input column names. Use placeholders
            like "{col}" for parameterized column names.
        output_cols_fn: Function that takes params dict and returns output
            column names. If None, uses [name].
        description: Description of what the feature computes.

    Returns:
        Decorator function.

    Example:
        @register_feature(
            name="returns",
            required_cols=["{price_col}"],
            output_cols_fn=lambda p: [f"{p.get('price_col', 'close')}_return"],
            description="Compute period returns"
        )
        def compute_returns(df, price_col="close", horizon=1):
            ...
    """

    def decorator(func: Callable) -> Callable:
        _FEATURE_REGISTRY[name] = FeatureInfo(
            name=name,
            func=func,
            required_cols=required_cols,
            output_cols_fn=output_cols_fn,
            description=description,
        )
        return func

    return decorator


def get_feature(name: str) -> FeatureInfo:
    """Get a registered feature by name.

    Args:
        name: Feature name.

    Returns:
        FeatureInfo for the feature.

    Raises:
        KeyError: If feature not found.
    """
    if name not in _FEATURE_REGISTRY:
        available = list(_FEATURE_REGISTRY.keys())
        raise KeyError(f"Feature '{name}' not found. Available: {available}")
    return _FEATURE_REGISTRY[name]


def list_features() -> list[str]:
    """List all registered feature names."""
    return list(_FEATURE_REGISTRY.keys())


def get_feature_info(name: str) -> dict[str, Any]:
    """Get detailed info about a feature.

    Args:
        name: Feature name.

    Returns:
        Dict with name, required_cols, description.
    """
    info = get_feature(name)
    return {
        "name": info.name,
        "required_cols": info.required_cols,
        "description": info.description,
    }


def clear_registry() -> None:
    """Clear the feature registry. Mainly for testing."""
    _FEATURE_REGISTRY.clear()
