from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

from mlstudy.trading.backtest.mean_reversion.analysis import compute_code_distribution, compute_performance_metrics
from mlstudy.trading.backtest.mean_reversion.single_backtest.engine import run_backtest
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import MetricsOnlyResult, SweepError, SweepResult, \
    SweepScenario

_WORKER_MARKET_DATA: dict | None = None


def _worker_init(market_data: dict) -> None:
    global _WORKER_MARKET_DATA
    _WORKER_MARKET_DATA = market_data


def _run_one(
    scenario_idx: int,
    scenario: SweepScenario,
    market_data: dict,
    mode: str,
) -> MetricsOnlyResult | SweepResult:
    try:
        res = run_backtest(cfg=scenario.cfg, **market_data)
        metrics = compute_performance_metrics(res)

        if mode == "metrics_only":
            code_counts = compute_code_distribution(res)
            return MetricsOnlyResult(
                scenario_idx=scenario_idx,
                scenario=scenario,
                total_pnl=metrics.total_pnl,
                final_equity=float(res.equity[-1]),
                n_trades=metrics.n_trades,
                max_drawdown=metrics.max_drawdown,
                sharpe_ratio=metrics.sharpe_ratio,
                code_counts=code_counts,
            )

        return SweepResult(
            scenario_idx=scenario_idx,
            scenario=scenario,
            results=res,
            metrics=metrics,
        )
    except Exception as exc:
        raise SweepError(scenario_idx, scenario.cfg) from exc


def _make_chunks(
    items: list[tuple[int, SweepScenario]],
    chunk_size: int,
) -> list[list[tuple[int, SweepScenario]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_chunk(
    chunk: list[tuple[int, SweepScenario]],
    market_data: dict,
    mode: str,
) -> list[tuple[int, MetricsOnlyResult | SweepResult]]:
    return [(idx, _run_one(idx, sc, market_data, mode)) for idx, sc in chunk]


def _run_chunk_process(
    chunk: list[tuple[int, SweepScenario]],
    mode: str,
) -> list[tuple[int, MetricsOnlyResult | SweepResult]]:
    return _run_chunk(chunk, _WORKER_MARKET_DATA, mode)


def _dispatch(
    indexed_scenarios: list[tuple[int, SweepScenario]],
    market_data: dict,
    backend: str,
    n_workers: int,
    chunk_size: int,
    mode: str,
) -> list[MetricsOnlyResult] | list[SweepResult]:
    if not indexed_scenarios:
        return []

    chunks = _make_chunks(indexed_scenarios, chunk_size)

    if backend == "serial":
        pairs: list[tuple[int, Any]] = []
        for chunk in chunks:
            pairs.extend(_run_chunk(chunk, market_data, mode))

    elif backend == "thread":
        pairs = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_run_chunk, c, market_data, mode): None for c in chunks}
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
