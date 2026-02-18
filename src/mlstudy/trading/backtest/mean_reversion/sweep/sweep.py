from __future__ import annotations

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from mlstudy.trading.backtest.metrics.metrics_calculator import MetricsCalculator
from mlstudy.trading.backtest.mean_reversion.single_backtest.engine import run_backtest
from .sweep_persist import SweepPersister
from .sweep_rank import RankingPlan, SweepRanker
from .sweep_types import (
    SweepResultLight,
    SweepError,
    SweepResult,
    SweepScenario,
    SweepSummary,
)
from ...metrics.metrics import BacktestMetrics


# ---------------------------------------------------------------------------
# Module-level globals/functions required for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

_WORKER_MARKET_DATA: dict | None = None


def _worker_init(market_data: dict) -> None:
    global _WORKER_MARKET_DATA
    _WORKER_MARKET_DATA = market_data


def _run_chunk_process(
    chunk: list[tuple[int, SweepScenario]],
    mode: str,
) -> list[tuple[int, SweepResultLight | SweepResult]]:
    return SweepExecutor._run_chunk(chunk, _WORKER_MARKET_DATA, mode)


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
            return SweepExecutor._dispatch(indexed, market_data, backend, workers, csize, "full")

        if mode != "metrics_only":
            raise ValueError(f"Unknown mode {mode!r}; choose from 'full', 'metrics_only'")

        metrics_results = SweepExecutor._dispatch(indexed, market_data, backend, workers, csize, "metrics_only")

        if keep_top_k_full <= 0:
            return metrics_results

        ranked = SweepRanker.rank_scenarios(metrics_results, ranking_plan)
        top_k = ranked[:keep_top_k_full]

        top_indexed = [(r.scenario_idx, r.scenario) for r in top_k]

        top_full_results = SweepExecutor._dispatch(top_indexed, market_data, backend, workers, csize, "full")

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
        rows: list[dict[str, Any]] = []
        for r in results:
            row: dict[str, Any] = {"name": r.scenario.name}
            row.update(r.scenario.tags)
            row.update(asdict(r.metrics))
            rows.append(row)
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------
    # Dispatch internals (absorbed from sweep_dispatch.py)
    # -------------------------------------------------------------------

    @staticmethod
    def _run_one(
        scenario_idx: int,
        scenario: SweepScenario,
        market_data: dict,
        mode: str,
    ) -> SweepResultLight | SweepResult:
        try:
            res = run_backtest(cfg=scenario.cfg, **market_data)
            metrics = MetricsCalculator(res.bar_df, res.trade_df).compute_all()

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

    @staticmethod
    def _make_chunks(
        items: list[tuple[int, SweepScenario]],
        chunk_size: int,
    ) -> list[list[tuple[int, SweepScenario]]]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    @staticmethod
    def _run_chunk(
        chunk: list[tuple[int, SweepScenario]],
        market_data: dict,
        mode: str,
    ) -> list[tuple[int, SweepResultLight | SweepResult]]:
        return [(idx, SweepExecutor._run_one(idx, sc, market_data, mode)) for idx, sc in chunk]

    @staticmethod
    def _dispatch(
        indexed_scenarios: list[tuple[int, SweepScenario]],
        market_data: dict,
        backend: str,
        n_workers: int,
        chunk_size: int,
        mode: str,
    ) -> list[SweepResultLight] | list[SweepResult]:
        if not indexed_scenarios:
            return []

        chunks = SweepExecutor._make_chunks(indexed_scenarios, chunk_size)

        if backend == "serial":
            pairs: list[tuple[int, Any]] = []
            for chunk in chunks:
                pairs.extend(SweepExecutor._run_chunk(chunk, market_data, mode))

        elif backend == "thread":
            pairs = []
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {pool.submit(SweepExecutor._run_chunk, c, market_data, mode): None for c in chunks}
                for fut in as_completed(futs):
                    pairs.extend(fut.result())

        elif backend == "process":
            pairs = []
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(market_data,),
            ) as pool:
                futs = {pool.submit(_run_chunk_process, c, mode): None for c in chunks}
                for fut in as_completed(futs):
                    pairs.extend(fut.result())

        else:
            raise ValueError(
                f"Unknown backend {backend!r}; choose from 'serial', 'thread', 'process'"
            )

        pairs.sort(key=lambda p: p[0])
        return [r for _, r in pairs]
