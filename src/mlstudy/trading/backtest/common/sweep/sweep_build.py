"""Strategy-agnostic scenario builder.

Works with any frozen dataclass config via ``dataclasses.replace``.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Sequence

from .sweep_types import SweepScenario


class ScenarioBuilder:
    @staticmethod
    def make_scenarios(
        base_cfg: Any,
        grid: dict[str, Sequence],
        *,
        name_prefix: str = "sweep",
    ) -> list[SweepScenario]:
        """Build all combinations from a base config and parameter grid.

        Parameters
        ----------
        base_cfg : frozen dataclass
            Base configuration.  Must support ``dataclasses.replace``.
        grid : dict[str, list]
            Parameter name → list of values to sweep.
        name_prefix : str
            Prefix for scenario names.

        Returns
        -------
        list[SweepScenario]
        """
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
