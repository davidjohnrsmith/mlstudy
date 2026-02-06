"""Data loading interfaces.

This module contains:
- Dataset abstractions for ML training
- Price data loaders (OHLCV)
- Feature data loaders
- External signal/factor loaders
- Data validation utilities
- Panel data utilities
- Session/trading hours utilities
"""

from __future__ import annotations

from mlstudy.core.data.dataset import MLDataFrameDataset
from mlstudy.core.data.panel import (
    PanelValidationResult,
    align_panels,
    fill_panel_gaps,
    get_panel_summary,
    pivot_by_bond,
    unpivot_to_panel,
    validate_panel,
)
from mlstudy.core.data.session import (
    add_session_flags,
    compute_trading_date,
    filter_session,
    get_session_boundaries,
    is_rebalance_bar,
    parse_time,
)

__all__ = [
    # Dataset
    "MLDataFrameDataset",
    # Panel
    "PanelValidationResult",
    "align_panels",
    "fill_panel_gaps",
    "get_panel_summary",
    "pivot_by_bond",
    "unpivot_to_panel",
    "validate_panel",
    # Session
    "add_session_flags",
    "compute_trading_date",
    "filter_session",
    "get_session_boundaries",
    "is_rebalance_bar",
    "parse_time",
]
