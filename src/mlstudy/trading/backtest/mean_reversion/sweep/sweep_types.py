from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults
from mlstudy.trading.backtest.metrics import BacktestMetrics


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

    def metrics_dict(self) -> dict[str, float]:
        return {
            "total_pnl": self.total_pnl,
            "final_equity": self.final_equity,
            "n_trades": float(self.n_trades),
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
        }


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
