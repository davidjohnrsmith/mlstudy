from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Sequence

from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import SweepScenario


class ScenarioBuilder:
    @staticmethod
    def make_scenarios(
        base_cfg: MRBacktestConfig,
        grid: dict[str, Sequence],
        *,
        name_prefix: str = "sweep",
    ) -> list[SweepScenario]:
        keys = list(grid.keys())
        combos = list(product(*[grid[k] for k in keys]))
        n_digits = max(len(str(len(combos) - 1)), 4)

        scenarios: list[SweepScenario] = []
        for idx, combo in enumerate(combos):
            patch = dict(zip(keys, combo))
            cfg = replace(base_cfg, **patch)
            name = f"{name_prefix}_{idx:0{n_digits}d}"
            scenarios.append(SweepScenario(name=name, cfg=cfg, tags=patch))

        return scenarios
