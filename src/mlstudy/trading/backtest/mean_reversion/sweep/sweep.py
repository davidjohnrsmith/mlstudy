"""Mean-reversion sweep executor.

Uses ``common.sweep.sweep_dispatch`` for parallel execution and
``common.sweep.sweep_persist`` (via ``SweepPersister``) for saving.
"""

from __future__ import annotations

import warnings
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from mlstudy.trading.backtest.metrics.metrics_calculator import MetricsCalculator
from mlstudy.trading.backtest.mean_reversion.single_backtest.engine import run_backtest
from mlstudy.trading.backtest.common.sweep.sweep_dispatch import dispatch
from mlstudy.trading.backtest.common.sweep.sweep_persist import summary_table
from .sweep_persist import SweepPersister
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan, SweepRanker
from mlstudy.trading.backtest.common.sweep.sweep_types import (
    SweepResultLight,
    SweepError,
    SweepResult,
    SweepScenario,
    SweepSummary,
)


# ---------------------------------------------------------------------------
# run-one callback for dispatch
# ---------------------------------------------------------------------------

def _run_one_mr(
    scenario_idx: int,
    scenario: SweepScenario,
    market_data: dict,
    mode: str,
) -> SweepResultLight | SweepResult:
    try:
        res = run_backtest(cfg=scenario.cfg, **market_data)
        bar_df = res.close_bar_df if res.close_bar_df is not None else res.bar_df
        metrics = MetricsCalculator(bar_df, res.trade_df).compute_all()

        if mode == "metrics_only":
            return SweepResultLight(
                scenario_idx=scenario_idx,
                scenario=scenario,
                metrics=metrics,
            )

        return SweepResult(
            scenario_idx=scenario_idx,
            scenario=scenario,
            results=res,
            metrics=metrics,
        )
    except Exception as exc:
        raise SweepError(scenario_idx, scenario.cfg) from exc


# ---------------------------------------------------------------------------
# SweepExecutor
# ---------------------------------------------------------------------------


class SweepExecutor:
    @staticmethod
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
        datetimes=None,
        parallel: bool = False,
        backend: str = "serial",
        n_workers: int | None = None,
        chunk_size: int | None = None,
        mode: str = "full",
        keep_top_k_full: int = 0,
        save_top_full_dir: str | Path | None = None,
        ranking_plan: RankingPlan | None = None,
    ) -> list[SweepResult] | list[SweepResultLight] | SweepSummary:
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
        if datetimes is not None:
            market_data["datetimes"] = datetimes

        workers = n_workers or os.cpu_count() or 1

        n = len(scenarios)
        csize = chunk_size if chunk_size is not None else max(10, n // (10 * workers))
        csize = max(1, csize)

        indexed = list(enumerate(scenarios))

        if mode == "full":
            return dispatch(indexed, market_data, _run_one_mr, backend, workers, csize, "full")

        if mode != "metrics_only":
            raise ValueError(f"Unknown mode {mode!r}; choose from 'full', 'metrics_only'")

        metrics_results = dispatch(indexed, market_data, _run_one_mr, backend, workers, csize, "metrics_only")

        if keep_top_k_full <= 0:
            return metrics_results

        if ranking_plan is None:
            ranking_plan = RankingPlan()

        ranked = SweepRanker.rank_scenarios(metrics_results, ranking_plan)
        top_k = ranked[:keep_top_k_full]

        top_indexed = [(r.scenario_idx, r.scenario) for r in top_k]

        top_full_results = dispatch(top_indexed, market_data, _run_one_mr, backend, workers, csize, "full")

        rank_order = {r.scenario_idx: i for i, r in enumerate(top_k)}
        top_full_results.sort(key=lambda r: rank_order[r.scenario_idx])

        if save_top_full_dir is not None:
            SweepPersister.save_top_full(top_full_results, save_top_full_dir)

        ranked_all = SweepRanker.rank_scenarios(metrics_results, ranking_plan)
        return SweepSummary(all_metrics=ranked_all, top_full=top_full_results)

    @staticmethod
    def summary_table(
        results: list[SweepResult] | list[SweepResultLight],
    ) -> pd.DataFrame:
        return summary_table(results)
