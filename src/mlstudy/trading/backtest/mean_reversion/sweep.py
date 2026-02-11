"""Parameter sweep for the MR backtester.

Build a grid of ``MRBacktestConfig`` variations, run them against the same
market data, and rank by any ``BacktestMetrics`` field.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from typing import Any, Sequence

import pandas as pd

from ..metrics import BacktestMetrics
from .analysis import compute_performance_metrics
from .engine import MRBacktestConfig, run_backtest
from .results import MRBacktestResults


@dataclass(frozen=True)
class SweepScenario:
    """A single configuration variant in a parameter sweep."""

    name: str
    cfg: MRBacktestConfig
    tags: dict[str, Any]


@dataclass
class SweepResult:
    """Backtest output for one scenario."""

    scenario: SweepScenario
    results: MRBacktestResults
    metrics: BacktestMetrics


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


def make_scenarios(
    base_cfg: MRBacktestConfig,
    grid: dict[str, Sequence],
    *,
    name_prefix: str = "sweep",
) -> list[SweepScenario]:
    """Build the cartesian product of configs from *grid*.

    Parameters
    ----------
    base_cfg:
        Starting configuration; fields not in *grid* are kept as-is.
    grid:
        ``{field_name: [value, ...]}`` — each key must be a field of
        ``MRBacktestConfig``.  Single-key dict gives a 1-D sweep.
    name_prefix:
        Prefix used when auto-generating scenario names.

    Returns
    -------
    list[SweepScenario]
    """
    keys = list(grid.keys())
    scenarios: list[SweepScenario] = []

    for combo in product(*[grid[k] for k in keys]):
        patch = dict(zip(keys, combo))
        cfg = replace(base_cfg, **patch)
        suffix = ",".join(f"{k}={v}" for k, v in patch.items())
        name = f"{name_prefix}:{suffix}"
        scenarios.append(SweepScenario(name=name, cfg=cfg, tags=patch))

    return scenarios


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


def _run_one(scenario: SweepScenario, market_data: dict) -> SweepResult:
    """Run a single backtest and compute metrics."""
    res = run_backtest(cfg=scenario.cfg, **market_data)
    metrics = compute_performance_metrics(res)
    return SweepResult(scenario=scenario, results=res, metrics=metrics)


def run_sweep(
    scenarios: list[SweepScenario],
    *,
    bid_px,
    bid_sz,
    ask_px,
    ask_sz,
    mid_px,
    dv01,
    zscore,
    expected_yield_pnl_bps,
    package_yield_bps,
    hedge_ratios,
    parallel: bool = False,
) -> list[SweepResult]:
    """Run *scenarios* against the same market data.

    Parameters
    ----------
    scenarios:
        Output of :func:`make_scenarios`.
    bid_px, bid_sz, ask_px, ask_sz, mid_px, dv01, zscore,
    expected_yield_pnl_bps, package_yield_bps, hedge_ratios:
        Market data arrays — identical across all runs.
    parallel:
        If *True*, use a :class:`~concurrent.futures.ThreadPoolExecutor`.
        Defaults to *False* because Numba JIT serialisation can be tricky;
        ``ThreadPoolExecutor`` is used (GIL released by numpy/numba).

    Returns
    -------
    list[SweepResult]
        One entry per scenario, in the same order as *scenarios*.
    """
    market_data = dict(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        mid_px=mid_px,
        dv01=dv01,
        zscore=zscore,
        expected_yield_pnl_bps=expected_yield_pnl_bps,
        package_yield_bps=package_yield_bps,
        hedge_ratios=hedge_ratios,
    )

    if not parallel:
        return [_run_one(sc, market_data) for sc in scenarios]

    # Parallel path — ThreadPoolExecutor (GIL released by numpy/numba).
    results: list[SweepResult | None] = [None] * len(scenarios)
    with ThreadPoolExecutor() as pool:
        future_to_idx = {
            pool.submit(_run_one, sc, market_data): i
            for i, sc in enumerate(scenarios)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Ranking / summary
# ---------------------------------------------------------------------------


def rank_results(
    results: list[SweepResult],
    metric: str = "sharpe_ratio",
    top_n: int = 5,
    ascending: bool = False,
) -> list[SweepResult]:
    """Sort *results* by a ``BacktestMetrics`` field and return the top *top_n*.

    Parameters
    ----------
    metric:
        Any attribute of :class:`BacktestMetrics`.
    top_n:
        Number of results to return.  If *top_n* exceeds ``len(results)``
        all results are returned.
    ascending:
        If *True*, lower values rank first (useful for drawdown).
    """
    valid_fields = {f.name for f in fields(BacktestMetrics)}
    if metric not in valid_fields:
        raise ValueError(
            f"Unknown metric {metric!r}; choose from {sorted(valid_fields)}"
        )
    sorted_results = sorted(
        results,
        key=lambda r: getattr(r.metrics, metric),
        reverse=not ascending,
    )
    return sorted_results[:top_n]


def summary_table(results: list[SweepResult]) -> pd.DataFrame:
    """One-row-per-scenario DataFrame with tag columns + all metrics.

    Returns
    -------
    pd.DataFrame
    """
    rows: list[dict[str, Any]] = []
    for sr in results:
        row: dict[str, Any] = {"name": sr.scenario.name}
        row.update(sr.scenario.tags)
        row.update(asdict(sr.metrics))
        rows.append(row)
    return pd.DataFrame(rows)
