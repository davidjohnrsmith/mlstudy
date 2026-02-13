from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Sequence

from .engine import MRBacktestConfig
from .sweep_types import SweepScenario


def make_scenarios(
    base_cfg: MRBacktestConfig,
    grid: dict[str, Sequence],
    *,
    name_prefix: str = "sweep",
) -> list[SweepScenario]:
    keys = list(grid.keys())
    scenarios: list[SweepScenario] = []

    for combo in product(*[grid[k] for k in keys]):
        patch = dict(zip(keys, combo))
        cfg = replace(base_cfg, **patch)
        suffix = ",".join(f"{k}={v}" for k, v in patch.items())
        name = f"{name_prefix}:{suffix}"
        scenarios.append(SweepScenario(name=name, cfg=cfg, tags=patch))

    return scenarios
