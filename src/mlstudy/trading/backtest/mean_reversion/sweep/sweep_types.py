from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults
from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics



@dataclass(frozen=True)
class SweepScenario:
    name: str
    cfg: MRBacktestConfig
    tags: dict[str, Any]


@dataclass
class SweepResultLight:
    scenario: SweepScenario
    metrics: BacktestMetrics
    scenario_idx: int = -1


@dataclass
class SweepResult:
    scenario: SweepScenario
    results: MRBacktestResults
    metrics: BacktestMetrics
    scenario_idx: int = -1


@dataclass
class SweepSummary:
    all_metrics: list[SweepResultLight]
    top_full: list[SweepResult]


class SweepError(Exception):
    def __init__(self, scenario_idx: int, scenario_cfg: MRBacktestConfig):
        self.scenario_idx = scenario_idx
        self.scenario_cfg = scenario_cfg
        super().__init__(f"Scenario {scenario_idx} failed")
