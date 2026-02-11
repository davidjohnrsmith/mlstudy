"""
backtest/scenarios/sweeps.py

Scenario sweep helpers:
- sweep one parameter across values
- grid sweep across multiple parameters
- config patching via dot-path keys

This is intentionally simple and avoids dependencies (no pandas).
"""

from __future__ import annotations

from dataclasses import dataclass, replace, is_dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..core.types import BacktestConfig


@dataclass(frozen=True)
class Scenario:
    """
    A scenario is a config plus some metadata name/labels.
    """
    name: str
    cfg: BacktestConfig
    tags: Dict[str, Any]


def _get_attr(obj: Any, attr: str) -> Any:
    if not hasattr(obj, attr):
        raise AttributeError(f"{type(obj).__name__} has no attribute '{attr}'")
    return getattr(obj, attr)


def _set_attr(obj: Any, attr: str, value: Any) -> Any:
    """
    Set dataclass attribute immutably using dataclasses.replace().
    """
    if not is_dataclass(obj):
        raise TypeError(f"Cannot set attribute on non-dataclass object: {type(obj)}")
    return replace(obj, **{attr: value})


def apply_patch(cfg: BacktestConfig, patch: Dict[str, Any]) -> BacktestConfig:
    """
    Apply a patch dict to a nested BacktestConfig using dot paths.

    Example:
      patch = {
        "execution.size_haircut": 0.5,
        "timing.exec_lag_bars": 1,
        "fees.fee_bps": 1.0,
      }
    """
    out = cfg
    for path, value in patch.items():
        parts = path.split(".")
        if len(parts) == 1:
            out = _set_attr(out, parts[0], value)
            continue

        # Walk down to the parent
        parent = out
        parents = []
        for p in parts[:-1]:
            parents.append((parent, p))
            parent = _get_attr(parent, p)

        leaf_attr = parts[-1]
        parent2 = _set_attr(parent, leaf_attr, value)

        # Rebuild upwards
        for obj, p in reversed(parents):
            parent2 = _set_attr(obj, p, parent2)

        out = parent2

    return out


def sweep_one(
    base_cfg: BacktestConfig,
    *,
    key: str,
    values: Sequence[Any],
    name_prefix: str = "sweep",
) -> List[Scenario]:
    """
    Sweep a single dot-path key across values.

    Returns a list of Scenario objects.
    """
    scenarios: List[Scenario] = []
    for v in values:
        cfg = apply_patch(base_cfg, {key: v})
        scen_name = f"{name_prefix}:{key}={v}"
        scenarios.append(Scenario(name=scen_name, cfg=cfg, tags={key: v}))
    return scenarios


def grid_sweep(
    base_cfg: BacktestConfig,
    *,
    grid: Dict[str, Sequence[Any]],
    name_prefix: str = "grid",
) -> List[Scenario]:
    """
    Cartesian product sweep across multiple keys.

    grid example:
      {
        "execution.size_haircut": [0.3, 0.6, 0.9],
        "execution.max_levels_to_cross": [1, 3, 5],
        "timing.exec_lag_bars": [0, 1],
      }
    """
    keys = list(grid.keys())
    scenarios: List[Scenario] = []

    for combo in product(*[grid[k] for k in keys]):
        patch = {k: v for k, v in zip(keys, combo)}
        cfg = apply_patch(base_cfg, patch)
        tag = {k: v for k, v in patch.items()}
        suffix = ",".join([f"{k}={v}" for k, v in patch.items()])
        scen_name = f"{name_prefix}:{suffix}"
        scenarios.append(Scenario(name=scen_name, cfg=cfg, tags=tag))

    return scenarios
