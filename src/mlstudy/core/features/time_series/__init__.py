"""Time-series feature families for trading research."""

from __future__ import annotations

from mlstudy.core.features.time_series.flow import (
    compute_flow_imbalance,
    compute_signed_volume,
)
from mlstudy.core.features.time_series.price import (
    compute_log_returns,
    compute_momentum,
    compute_returns,
    compute_rolling_volatility,
)
from mlstudy.core.features.time_series.volume import (
    compute_dollar_volume,
    compute_rolling_volume,
    compute_volume_shock,
)

__all__ = [
    # Price features
    "compute_returns",
    "compute_log_returns",
    "compute_momentum",
    "compute_rolling_volatility",
    # Volume features
    "compute_volume_shock",
    "compute_rolling_volume",
    "compute_dollar_volume",
    # Flow features
    "compute_flow_imbalance",
    "compute_signed_volume",
]
