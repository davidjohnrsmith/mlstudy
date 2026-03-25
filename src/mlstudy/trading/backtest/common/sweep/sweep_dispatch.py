"""Strategy-agnostic parallel dispatch for sweep execution.

The caller provides a ``run_one_fn`` callback that runs a single scenario
and returns a ``SweepResultLight`` or ``SweepResult``.
"""

from __future__ import annotations

import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Tuple

from .sweep_types import SweepResultLight, SweepResult, SweepScenario


# ---------------------------------------------------------------------------
# Module-level globals for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

_WORKER_MARKET_DATA: dict | None = None
_WORKER_RUN_ONE_FN: Callable | None = None


def _worker_init(market_data: dict, run_one_fn: Callable) -> None:
    global _WORKER_MARKET_DATA, _WORKER_RUN_ONE_FN
    _WORKER_MARKET_DATA = market_data
    _WORKER_RUN_ONE_FN = run_one_fn


def _run_chunk_process(
    chunk: list[tuple[int, SweepScenario]],
    mode: str,
) -> list[tuple[int, Any]]:
    try:
        return _run_chunk(chunk, _WORKER_MARKET_DATA, _WORKER_RUN_ONE_FN, mode)
    except Exception as exc:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Worker process failed:\n{tb}"
        ) from None


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _run_chunk(
    chunk: list[tuple[int, SweepScenario]],
    market_data: dict,
    run_one_fn: Callable,
    mode: str,
) -> list[tuple[int, Any]]:
    return [(idx, run_one_fn(idx, sc, market_data, mode)) for idx, sc in chunk]


def make_chunks(
    items: list[tuple[int, SweepScenario]],
    chunk_size: int,
) -> list[list[tuple[int, SweepScenario]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(
    indexed_scenarios: list[tuple[int, SweepScenario]],
    market_data: dict,
    run_one_fn: Callable,
    backend: str,
    n_workers: int,
    chunk_size: int,
    mode: str,
) -> list[SweepResultLight] | list[SweepResult]:
    """Run scenarios via the specified backend.

    Parameters
    ----------
    indexed_scenarios : list of (idx, SweepScenario)
    market_data : dict of arrays
    run_one_fn : callable(scenario_idx, scenario, market_data, mode) -> result
    backend : "serial" | "thread" | "process"
    n_workers : number of parallel workers
    chunk_size : scenarios per chunk
    mode : "full" | "metrics_only"

    Returns
    -------
    list of SweepResultLight or SweepResult, sorted by scenario index
    """
    if not indexed_scenarios:
        return []

    cpu_count = os.cpu_count() or 1
    if n_workers > cpu_count:
        n_workers = cpu_count

    chunks = make_chunks(indexed_scenarios, chunk_size)

    if backend == "serial":
        pairs: list[tuple[int, Any]] = []
        for chunk in chunks:
            pairs.extend(_run_chunk(chunk, market_data, run_one_fn, mode))

    elif backend == "thread":
        pairs = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_run_chunk, c, market_data, run_one_fn, mode): None
                for c in chunks
            }
            for fut in as_completed(futs):
                pairs.extend(fut.result())

    elif backend == "process":
        pairs = []
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(market_data, run_one_fn),
        ) as pool:
            futs = {
                pool.submit(_run_chunk_process, c, mode): None
                for c in chunks
            }
            for fut in as_completed(futs):
                pairs.extend(fut.result())

    else:
        raise ValueError(
            f"Unknown backend {backend!r}; choose from 'serial', 'thread', 'process'"
        )

    pairs.sort(key=lambda p: p[0])
    return [r for _, r in pairs]
