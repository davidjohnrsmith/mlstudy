"""Load sweep configuration from YAML files.

A YAML config fully specifies everything needed to call ``run_sweep``:
base config, parameter grid, sweep execution options, and ranking plan.

Multiple named configs are supported via a config map — a YAML file that
maps short names to YAML file paths.  This lets you keep one map and
switch between tuning runs by name.

Typical usage::

    cfg = load_sweep_config("configs/mr_grid_v1.yaml")
    scenarios = make_scenarios(cfg.base_config, cfg.grid, name_prefix=cfg.grid_name)
    results = run_sweep(scenarios, **market_data, **cfg.sweep_kwargs)

Or with the config map::

    cfg = load_sweep_config_by_name("mr_grid_v1")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.data.data_loader import BacktestDataLoader
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan
from mlstudy.trading.backtest.mean_reversion.parameters.parameter import MRParameter
from mlstudy.trading.backtest.parameters.parameters_registry import ParameterPreferenceRegistry

_MR_REGISTRY = ParameterPreferenceRegistry(MRParameter)


@dataclass(frozen=True)
class SweepConfig:
    """Parsed sweep configuration from a YAML file."""

    grid_name: str
    base_config: MRBacktestConfig
    grid: dict[str, Sequence]
    sweep_kwargs: dict[str, Any]
    ranking_plan: RankingPlan | None
    data_loader: BacktestDataLoader | None = None  # BacktestDataLoader, if ``data`` section present


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_base_config(raw: dict[str, Any]) -> MRBacktestConfig:
    """Construct ``MRBacktestConfig`` from the ``base_config`` YAML section."""
    return MRBacktestConfig(**raw)


def _build_grid(raw: dict[str, list]) -> dict[str, Sequence]:
    """Validate and return the parameter grid."""
    for key, values in raw.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"grid[{key!r}] must be a non-empty list, got {values!r}")
    return raw


def _build_ranking_plan(raw: dict[str, Any] | None) -> RankingPlan | None:
    """Construct ``RankingPlan`` from the ``rank`` YAML section.

    Supports two styles:

    1. Explicit weights (recommended)::

        rank:
          primary_metrics:
            - ["total_pnl", 1.0]
          tie_metrics:
            - ["max_drawdown", 0.5]

    2. Simple list (weight defaults to 1.0)::

        rank:
          primary_metrics:
            - "total_pnl"
    """
    if raw is None:
        return None

    def _parse_features(items: list | None) -> tuple[tuple[str, float], ...]:
        if not items:
            return ()
        result = []
        for item in items:
            if isinstance(item, str):
                result.append((item, 1.0))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                result.append((str(item[0]), float(item[1])))
            else:
                raise ValueError(
                    f"Each ranking entry must be a string or [name, weight], got {item!r}"
                )
        return tuple(result)

    return RankingPlan(
        primary_metrics=_parse_features(raw.get("primary_metrics")),
        tie_metrics=_parse_features(raw.get("tie_metrics")),
        primary_params=_parse_features(raw.get("primary_params")),
        tie_params=_parse_features(raw.get("tie_params")),
        param_registry=_MR_REGISTRY,
    )


def _build_sweep_kwargs(
    raw: dict[str, Any] | None,
    ranking_plan: RankingPlan | None,
) -> dict[str, Any]:
    """Build keyword arguments for ``run_sweep`` from the ``sweep`` YAML section."""
    kwargs: dict[str, Any] = {}
    if raw is None:
        return kwargs

    key_map = {
        "backend": "backend",
        "n_workers": "n_workers",
        "chunk_size": "chunk_size",
        "mode": "mode",
        "keep_top_k_full": "keep_top_k_full",
        "save_top_full_dir": "save_top_full_dir",
    }
    for yaml_key, kwarg_key in key_map.items():
        if yaml_key in raw:
            kwargs[kwarg_key] = raw[yaml_key]

    if ranking_plan is not None:
        kwargs["ranking_plan"] = ranking_plan

    return kwargs


def _build_data_loader(raw: dict[str, Any] | None) -> Any | None:
    """Construct ``BacktestDataLoader`` from the ``data`` YAML section.

    Returns *None* when no ``data`` section is present.
    """
    if raw is None:
        return None



    return BacktestDataLoader(**raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sweep_config(path: str | Path) -> SweepConfig:
    """Load a sweep configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SweepConfig
        Parsed configuration ready for use with ``make_scenarios`` and ``run_sweep``.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    grid_name = raw.get("grid_name", path.stem)
    base_config = _build_base_config(raw.get("base_config", {}))
    grid = _build_grid(raw.get("grid", {}))
    ranking_plan = _build_ranking_plan(raw.get("rank"))
    sweep_kwargs = _build_sweep_kwargs(raw.get("sweep"), ranking_plan)

    data_loader = _build_data_loader(raw.get("data"))

    return SweepConfig(
        grid_name=grid_name,
        base_config=base_config,
        grid=grid,
        sweep_kwargs=sweep_kwargs,
        ranking_plan=ranking_plan,
        data_loader=data_loader,
    )






