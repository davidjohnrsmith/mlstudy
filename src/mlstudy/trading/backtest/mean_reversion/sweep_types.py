from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..metrics import BacktestMetrics
from .engine import MRBacktestConfig
from .results import MRBacktestResults


@dataclass(frozen=True)
class SweepScenario:
    name: str
    cfg: MRBacktestConfig
    tags: dict[str, Any]


@dataclass(frozen=True)
class MetricsOnlyResult:
    scenario_idx: int
    scenario: SweepScenario
    total_pnl: float
    final_equity: float
    n_trades: int
    max_drawdown: float
    sharpe_ratio: float
    code_counts: dict[str, int]


@dataclass
class SweepResult:
    scenario: SweepScenario
    results: MRBacktestResults
    metrics: BacktestMetrics
    scenario_idx: int = -1


@dataclass
class SweepSummary:
    all_metrics: list[MetricsOnlyResult]
    top_full: list[SweepResult]


class SweepError(Exception):
    def __init__(self, scenario_idx: int, scenario_cfg: MRBacktestConfig):
        self.scenario_idx = scenario_idx
        self.scenario_cfg = scenario_cfg
        super().__init__(f"Scenario {scenario_idx} failed")
