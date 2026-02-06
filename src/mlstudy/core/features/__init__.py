"""Feature engineering module for trading research.

This module provides:
- Feature specifications and registry
- Low-level operations (rolling, lag, groupby-aware)
- Time-series features (price, volume, flow)
- Cross-sectional features (ranking, z-score)
- Calendar features (day of week, month, etc.)
- Pipeline for building features from specs
"""

from __future__ import annotations

# Import feature modules to register features (noqa for side-effect imports)
from mlstudy.core.features import calendar as calendar  # noqa: F401
from mlstudy.core.features import cross_sectional as cross_sectional  # noqa: F401
from mlstudy.core.features import time_series as time_series  # noqa: F401
from mlstudy.core.features.pipeline import build_features, quick_features
from mlstudy.core.features.base import FeatureSpec, FeatureResult, FeatureReport, FeatureInfo
from mlstudy.core.features.registry import register_feature, get_feature, list_features
__all__ = [
    # Base classes
    "FeatureSpec",
    "FeatureResult",
    "FeatureReport",
    "FeatureInfo",
    # Registry
    "register_feature",
    "get_feature",
    "list_features",
    # Pipeline
    "build_features",
    "quick_features",
]

