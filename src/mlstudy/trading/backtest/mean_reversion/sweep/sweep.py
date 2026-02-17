from __future__ import annotations

import os
import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from .sweep_dispatch import _dispatch
from .sweep_persist import _save_top_full
from .sweep_rank import RankingPlan, rank_scenarios
from .sweep_types import (
    MetricsOnlyResult,
    SweepResult,
    SweepScenario,
    SweepSummary,
)
from ...metrics.metrics import BacktestMetrics


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
    backend: str = "serial",
    n_workers: int | None = None,
    chunk_size: int | None = None,
    mode: str = "full",
    keep_top_k_full: int = 0,
    save_top_full_dir: str | Path | None = None,
    ranking_plan: RankingPlan | None = None,
) -> list[SweepResult] | list[MetricsOnlyResult] | SweepSummary:
    if parallel and backend == "serial":
        warnings.warn(
            "parallel=True is deprecated; use backend='thread'",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = "thread"

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

    workers = n_workers or os.cpu_count() or 1

    n = len(scenarios)
    csize = chunk_size if chunk_size is not None else max(10, n // (10 * workers))
    csize = max(1, csize)

    indexed = list(enumerate(scenarios))

    if mode == "full":
        return _dispatch(indexed, market_data, backend, workers, csize, "full")

    if mode != "metrics_only":
        raise ValueError(f"Unknown mode {mode!r}; choose from 'full', 'metrics_only'")

    metrics_results = _dispatch(indexed, market_data, backend, workers, csize, "metrics_only")

    if keep_top_k_full <= 0:
        return metrics_results

    ranked = rank_scenarios(metrics_results, ranking_plan)
    top_k = ranked[:keep_top_k_full]

    top_indexed = [(r.scenario_idx, r.scenario) for r in top_k]

    top_full_results = _dispatch(top_indexed, market_data, backend, workers, csize, "full")

    rank_order = {r.scenario_idx: i for i, r in enumerate(top_k)}
    top_full_results.sort(key=lambda r: rank_order[r.scenario_idx])

    if save_top_full_dir is not None:
        _save_top_full(top_full_results, save_top_full_dir)

    return SweepSummary(all_metrics=metrics_results, top_full=top_full_results)


_METRICS_ONLY_FIELDS = frozenset(
    {"total_pnl", "final_equity", "n_trades", "max_drawdown", "sharpe_ratio"}
)


def rank_results(
    results: list[SweepResult] | list[MetricsOnlyResult],
    metric: str = "sharpe_ratio",
    top_n: int = 5,
    ascending: bool = False,
) -> list[SweepResult] | list[MetricsOnlyResult]:
    if results and isinstance(results[0], MetricsOnlyResult):
        if metric not in _METRICS_ONLY_FIELDS:
            raise ValueError(
                f"Unknown metric {metric!r} for MetricsOnlyResult; "
                f"choose from {sorted(_METRICS_ONLY_FIELDS)}"
            )
        sorted_results = sorted(
            results,
            key=lambda r: getattr(r, metric),
            reverse=not ascending,
        )
    else:
        valid_fields = {f.name for f in fields(BacktestMetrics)}
        if metric not in valid_fields:
            raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(valid_fields)}")
        sorted_results = sorted(
            results,
            key=lambda r: getattr(r.metrics, metric),
            reverse=not ascending,
        )
    return sorted_results[:top_n]


def summary_table(
    results: list[SweepResult] | list[MetricsOnlyResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {"name": r.scenario.name}
        row.update(r.scenario.tags)

        if isinstance(r, MetricsOnlyResult):
            row.update(
                {
                    "total_pnl": r.total_pnl,
                    "final_equity": r.final_equity,
                    "n_trades": r.n_trades,
                    "max_drawdown": r.max_drawdown,
                    "sharpe_ratio": r.sharpe_ratio,
                }
            )
        else:
            row.update(asdict(r.metrics))
        rows.append(row)
    return pd.DataFrame(rows)
