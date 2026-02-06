"""Target variable generation for ML pipelines."""

from __future__ import annotations

from mlstudy.ml.targets.horizons import (
    MultiHorizonTargetGenerator,
    extract_horizon,
    get_horizon_columns,
    make_forward_change_target,
    make_multi_horizon_targets,
)

__all__ = [
    "make_forward_return_target",
    "make_forward_change_target",
    "make_multi_horizon_targets",
    "get_horizon_columns",
    "extract_horizon",
    "MultiHorizonTargetGenerator",
]
