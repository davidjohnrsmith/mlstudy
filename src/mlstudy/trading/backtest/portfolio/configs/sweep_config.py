"""Load sweep configuration from YAML files for portfolio backtests.

A YAML config fully specifies everything needed to call a portfolio sweep:
base config, parameter grid, sweep execution options, and ranking plan.

Multiple named configs are supported via a config map — a YAML file that
maps short names to YAML file paths.

Typical usage::

    cfg = load_sweep_config("configs/portfolio_grid_v1.yaml")
    # use cfg.base_config, cfg.grid, cfg.sweep_kwargs, etc.

Or with the config map::

    cfg = load_sweep_config_by_name("portfolio_grid_v1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

logger = logging.getLogger(__name__)

from .backtest_config import PortfolioBacktestConfig
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan
from mlstudy.trading.backtest.portfolio.data.data_loader import PortfolioDataLoader
from ..parameters.parameter import PortfolioParameter
from mlstudy.trading.backtest.parameters.parameters_registry import ParameterPreferenceRegistry

_PORTFOLIO_REGISTRY = ParameterPreferenceRegistry(PortfolioParameter)


@dataclass(frozen=True)
class PortfolioSweepConfig:
    """Parsed sweep configuration from a YAML file."""

    grid_name: str
    base_config: PortfolioBacktestConfig
    grid: dict[str, Sequence]
    sweep_kwargs: dict[str, Any]
    ranking_plan: RankingPlan | None = None
    data_loader: PortfolioDataLoader | None = None
    plot_config: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_base_config(
    raw: dict[str, Any],
    grid: dict[str, Sequence],
) -> PortfolioBacktestConfig:
    """Construct ``PortfolioBacktestConfig`` from base_config + grid.

    Grid parameters use the first value from each grid list as the base
    value, so they should NOT appear in ``base_config``.  All non-grid
    fields must be specified in ``base_config``.
    """
    merged = {**raw}
    for key, values in grid.items():
        merged[key] = values[0]
    return PortfolioBacktestConfig(**merged)


def _build_grid(raw: dict[str, list]) -> dict[str, Sequence]:
    """Validate and return the parameter grid."""
    for key, values in raw.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"grid[{key!r}] must be a non-empty list, got {values!r}")
    return raw


def _build_ranking_plan(raw: dict[str, Any] | None) -> RankingPlan | None:
    """Construct ``RankingPlan`` from the ``rank`` YAML section.

    Supports two styles:

    1. Explicit weights::

        rank:
          primary_metrics:
            - ["total_pnl", 1.0]

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
        param_registry=_PORTFOLIO_REGISTRY,
    )


def _build_sweep_kwargs(
    raw: dict[str, Any] | None,
    ranking_plan: RankingPlan | None,
) -> dict[str, Any]:
    """Build keyword arguments for sweep from the ``sweep`` YAML section."""
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
        "chunk_freq": "chunk_freq",
        "start_date": "start_date",
        "end_date": "end_date",
    }
    for yaml_key, kwarg_key in key_map.items():
        if yaml_key in raw:
            kwargs[kwarg_key] = raw[yaml_key]

    if ranking_plan is not None:
        kwargs["ranking_plan"] = ranking_plan

    return kwargs


def _build_data_loader(raw: dict[str, Any] | None) -> PortfolioDataLoader | None:
    """Construct ``PortfolioDataLoader`` from the ``data`` YAML section.

    Returns *None* when no ``data`` section is present.
    """
    if raw is None:
        return None
    return PortfolioDataLoader(**raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sweep_config(path: str | Path) -> PortfolioSweepConfig:
    """Load a sweep configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    PortfolioSweepConfig
    """
    path = Path(path)
    logger.info("Loading sweep config from %s", path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    grid_name = raw.get("grid_name", path.stem)
    raw_base = raw.get("base_config", {})
    grid = _build_grid(raw.get("grid", {}))

    overlap = set(raw_base) & set(grid)
    if overlap:
        raise ValueError(
            f"Parameters {sorted(overlap)} appear in both base_config and grid. "
            f"Grid parameters override base_config, so specifying them in both "
            f"is ambiguous. Remove them from base_config."
        )

    base_config = _build_base_config(raw_base, grid)
    ranking_plan = _build_ranking_plan(raw.get("rank"))
    sweep_kwargs = _build_sweep_kwargs(raw.get("sweep"), ranking_plan)
    data_loader = _build_data_loader(raw.get("data"))

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    logger.info(
        "Config '%s': %d grid params, %d combinations, data_loader=%s",
        grid_name, len(grid), n_combos, data_loader is not None,
    )

    plot_config = raw.get("plot")

    return PortfolioSweepConfig(
        grid_name=grid_name,
        base_config=base_config,
        grid=grid,
        sweep_kwargs=sweep_kwargs,
        ranking_plan=ranking_plan,
        data_loader=data_loader,
        plot_config=plot_config,
    )
