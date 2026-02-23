"""Strategy-agnostic sweep data types.

These types use ``Any`` for config and results so they work with both
mean-reversion and portfolio backtests (or any future strategy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics


@dataclass(frozen=True)
class SweepScenario:
    """One parameter combination to evaluate."""

    name: str
    cfg: Any  # strategy-specific config dataclass
    tags: dict[str, Any]


@dataclass
class SweepResultLight:
    """Metrics-only result (no full backtest arrays)."""

    scenario: SweepScenario
    metrics: BacktestMetrics
    scenario_idx: int = -1


@dataclass
class SweepResult:
    """Full result including backtest arrays."""

    scenario: SweepScenario
    results: Any  # strategy-specific results object
    metrics: BacktestMetrics
    scenario_idx: int = -1


@dataclass
class SweepSummary:
    """Combined output: all scenarios ranked + top-k full results."""

    all_metrics: list[SweepResultLight]
    top_full: list[SweepResult]


class SweepError(Exception):
    """Raised when a scenario fails during sweep execution."""

    def __init__(self, scenario_idx: int, scenario_cfg: Any):
        self.scenario_idx = scenario_idx
        self.scenario_cfg = scenario_cfg
        super().__init__(f"Scenario {scenario_idx} failed")
