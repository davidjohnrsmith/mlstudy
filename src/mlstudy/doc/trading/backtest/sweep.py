"""Parameter sweep for the MR backtester.

Build a grid of ``MRBacktestConfig`` variations, run them against the same
market data, and rank by any ``BacktestMetrics`` field.

Design choices
--------------
- **Backends**: ``"serial"`` (default), ``"thread"`` (GIL-released by
  numpy/numba), ``"process"`` (true parallelism via
  ``ProcessPoolExecutor``).  The process backend uses a ``spawn`` context
  and transfers market data once per worker via the pool initializer so
  that large numpy arrays are not re-pickled per task.

- **Chunking**: Scenarios are split into chunks before dispatch.  This
  controls per-task overhead and keeps the futures list small.  Default
  chunk size is ``max(10, n_scenarios // (10 * n_workers))``.

- **Memory modes**: ``mode="full"`` stores full ``MRBacktestResults`` per
  scenario (same as v1).  ``mode="metrics_only"`` computes scalar metrics
  and discards the heavy result arrays, dramatically reducing memory.

- **Top-K rerun**: When ``keep_top_k_full > 0`` in metrics-only mode, a
  second pass reruns the best scenarios in full mode, returning a
  ``SweepSummary`` with both lightweight metrics for all scenarios and
  full results for the winners.

- **Persistence**: ``save_top_full_dir`` writes each top-K result as
  ``scenario_NNN/spec.json`` + ``.npy`` array files (numpy + json only).

Notes for maintainers
---------------------
- This module is intentionally "thin": it orchestrates running many configs,
  but does not implement strategy logic. The actual trading loop lives in
  ``loop.py`` and the orchestrator lives in ``engine.py``.
- Deterministic ordering matters in sweeps. Even when using futures and
  ``as_completed``, this module re-sorts results back into scenario index order.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..metrics import BacktestMetrics
from .analysis import compute_code_distribution, compute_performance_metrics
from .engine import MRBacktestConfig, run_backtest
from .results import MRBacktestResults


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepScenario:
    """A single configuration variant in a parameter sweep.

    Attributes
    ----------
    name:
        Human-readable scenario name. Usually contains key=value patches.
        Mainly for logging/plot legends.
    cfg:
        The concrete MRBacktestConfig for this scenario.
    tags:
        Small metadata dict, typically the same patch values used to produce cfg.
        Used by summary_table() to create columns for grid parameters.
    """

    name: str
    cfg: MRBacktestConfig
    tags: dict[str, Any]


@dataclass(frozen=True)
class MetricsOnlyResult:
    """Lightweight result storing only scalar metrics (no heavy arrays).

    Used by ``mode="metrics_only"`` to avoid holding full
    ``MRBacktestResults`` in memory across thousands of scenarios.

    Notes
    -----
    - This object is intended to be cheap to keep in RAM for large sweeps.
    - It stores a code distribution so you can later inspect execution outcomes
      (e.g. too wide vs no liquidity) without rerunning full traces.
    """

    scenario_idx: int
    scenario: SweepScenario
    total_pnl: float
    final_equity: float
    n_trades: int
    max_drawdown: float
    sharpe_ratio: float
    code_counts: dict[str, int]


@dataclass
class SweepResult:
    """Backtest output for one scenario.

    This is the "full" mode output: it contains MRBacktestResults (per-bar arrays)
    plus a scalar BacktestMetrics summary.

    scenario_idx is duplicated here to keep ordering stable even after
    dispatch/collection (important for process/thread backends).
    """

    scenario: SweepScenario
    results: MRBacktestResults
    metrics: BacktestMetrics
    scenario_idx: int = -1


@dataclass
class SweepSummary:
    """Combined output from a metrics-only sweep with top-K full rerun.

    Attributes
    ----------
    all_metrics:
        One MetricsOnlyResult per scenario, in original (stable) order.
        This is "scan wide" output.
    top_full:
        Top-K SweepResult objects, in ranked order (best first).
        This is "inspect deep" output suitable for plotting.
    """

    all_metrics: list[MetricsOnlyResult]
    top_full: list[SweepResult]


class SweepError(Exception):
    """Raised when a single scenario fails during sweep execution.

    We wrap the original exception so callers see scenario index + config,
    which is critical when debugging large sweeps (otherwise you just get a
    worker exception without context).
    """

    def __init__(self, scenario_idx: int, scenario_cfg: MRBacktestConfig):
        self.scenario_idx = scenario_idx
        self.scenario_cfg = scenario_cfg
        super().__init__(f"Scenario {scenario_idx} failed")


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
        {field_name: [value, ...]} — each key must be a field of
        MRBacktestConfig. Single-key dict gives a 1-D sweep.
    name_prefix:
        Prefix used when auto-generating scenario names.

    Returns
    -------
    list[SweepScenario]

    Implementation notes
    --------------------
    - We use dataclasses.replace(base_cfg, **patch) to avoid mutating base_cfg.
    - Scenario ordering is deterministic: it follows itertools.product order
      over grid keys as provided (keys = list(grid.keys())).
    - tags are stored so summary_table can expose grid parameters as columns.
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
# Process pool global — market data transferred once per worker
# ---------------------------------------------------------------------------

# In process mode (spawn), passing market_data to every pool.submit() would
# pickle/copy the (potentially huge) numpy arrays per task. Instead:
# - initargs=(market_data,) transfers it once to each worker
# - _worker_init stores it in this module-level global inside that worker
# - per-task functions only receive small chunks + scalars (mode)
_WORKER_MARKET_DATA: dict | None = None


def _worker_init(market_data: dict) -> None:
    """Initializer for ProcessPoolExecutor workers.

    Stores market_data in a module-level global so that each worker
    receives the (large) numpy arrays exactly once, not per-task.

    Important detail:
    - With spawn, workers start fresh; globals are not inherited.
    - initializer runs once per worker process.
    """
    global _WORKER_MARKET_DATA
    _WORKER_MARKET_DATA = market_data


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------


def _run_one(
    scenario_idx: int,
    scenario: SweepScenario,
    market_data: dict,
    mode: str,
) -> MetricsOnlyResult | SweepResult:
    """Run one backtest and return the appropriate result type.

    This function is intentionally small and "pure":
    - input: scenario config + shared market arrays
    - output: either MetricsOnlyResult or SweepResult
    - exceptions: wrapped into SweepError with scenario context

    In process mode:
    - market_data comes from the worker global (via _run_chunk_process),
      not per-task pickling.
    """
    try:
        # The backtester core (engine/loop) executes using cfg + arrays.
        res = run_backtest(cfg=scenario.cfg, **market_data)

        # Compute scalar performance metrics from the full results.
        # In metrics_only mode we still compute metrics, but we do not keep res.
        metrics = compute_performance_metrics(res)

        if mode == "metrics_only":
            # Distribution of per-bar action/outcome codes (ENTRY_OK, etc).
            # This is useful for later diagnosing why a config performs poorly
            # without needing to keep full per-bar arrays for every scenario.
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

        # Full mode: store heavy arrays + scalar metrics.
        return SweepResult(
            scenario_idx=scenario_idx,
            scenario=scenario,
            results=res,
            metrics=metrics,
        )
    except Exception as exc:
        # Wrap errors with scenario context so the caller can identify which
        # parameter combination caused the failure.
        raise SweepError(scenario_idx, scenario.cfg) from exc


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _make_chunks(
    items: list[tuple[int, SweepScenario]],
    chunk_size: int,
) -> list[list[tuple[int, SweepScenario]]]:
    """Split items into sub-lists of at most chunk_size.

    Chunking rationale:
    - Reduces per-task scheduling overhead (especially in thread/process pools).
    - Amortizes any one-time costs inside a worker (e.g., Numba compilation).
    - Keeps the number of futures reasonable (fewer tasks to manage).
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_chunk(
    chunk: list[tuple[int, SweepScenario]],
    market_data: dict,
    mode: str,
) -> list[tuple[int, MetricsOnlyResult | SweepResult]]:
    """Run a chunk of scenarios, returning (idx, result) pairs.

    We return (idx, result) rather than just result because chunk tasks can
    complete out-of-order (as_completed). The caller can sort by idx to
    restore stable scenario ordering.
    """
    return [(idx, _run_one(idx, sc, market_data, mode)) for idx, sc in chunk]


def _run_chunk_process(
    chunk: list[tuple[int, SweepScenario]],
    mode: str,
) -> list[tuple[int, MetricsOnlyResult | SweepResult]]:
    """Process-pool wrapper — reads market data from the worker global.

    This function is the one submitted to ProcessPoolExecutor.
    It avoids sending market arrays in each task payload.
    """
    return _run_chunk(chunk, _WORKER_MARKET_DATA, mode)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _dispatch(
    indexed_scenarios: list[tuple[int, SweepScenario]],
    market_data: dict,
    backend: str,
    n_workers: int,
    chunk_size: int,
    mode: str,
) -> list[MetricsOnlyResult] | list[SweepResult]:
    """Route work to the chosen backend, collect results in stable order.

    Important behavior:
    - Uses as_completed() for parallel backends to gather finished tasks quickly.
    - Always sorts collected (idx, result) pairs by idx so output ordering matches
      the input scenario list order (deterministic and reproducible).
    """
    if not indexed_scenarios:
        return []

    # Prepare chunked worklist: each chunk is a list of (scenario_idx, scenario).
    chunks = _make_chunks(indexed_scenarios, chunk_size)

    if backend == "serial":
        # Serial execution is simplest and easiest to debug.
        pairs: list[tuple[int, Any]] = []
        for chunk in chunks:
            pairs.extend(_run_chunk(chunk, market_data, mode))

    elif backend == "thread":
        # Thread backend:
        # - Works well if numpy/numba releases the GIL inside heavy compute.
        # - Avoids process spawn overhead.
        # - Still shares memory (no pickling), so market_data is cheap to pass.
        pairs = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # We submit one future per chunk, not per scenario, to reduce overhead.
            futs = {
                pool.submit(_run_chunk, c, market_data, mode): None
                for c in chunks
            }
            for fut in as_completed(futs):
                # If a chunk raises, fut.result() will raise and bubble up.
                pairs.extend(fut.result())

    elif backend == "process":
        # Process backend:
        # - True CPU parallelism.
        # - Uses spawn context for cross-platform correctness.
        # - Market data is transferred once per worker via initializer.
        pairs = []
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(market_data,),
        ) as pool:
            # We submit one future per chunk. Each worker runs many scenarios per task.
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

    # Stable ordering by scenario_idx:
    # Futures complete in arbitrary order; we always re-order to match the
    # original scenario ordering (scenario index).
    pairs.sort(key=lambda p: p[0])
    return [r for _, r in pairs]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# List of array-like fields on MRBacktestResults that we want to persist.
# This enables lightweight "artifact" saving so users can plot later without
# rerunning those top scenarios.
#
# Note: This list is intentionally explicit rather than dir()/__dict__ iteration
# so output structure is stable and predictable.
_ARRAY_FIELDS = [
    "positions",
    "cash",
    "equity",
    "pnl",
    "codes",
    "state",
    "holding",
    "tr_bar",
    "tr_type",
    "tr_side",
    "tr_sizes",
    "tr_risks",
    "tr_vwaps",
    "tr_mids",
    "tr_cost",
    "tr_code",
    "tr_pkg_yield",
]


def _save_top_full(results: list[SweepResult], output_dir: str | Path) -> None:
    """Persist top-K full results to disk.

    Creates scenario_NNN/ subdirectories under output_dir, each containing:
    - spec.json: scenario name, tags, full config, scalar metrics, scenario_idx
    - *.npy: arrays in MRBacktestResults listed in _ARRAY_FIELDS

    This is intentionally minimal (json + numpy only) to avoid adding dependencies.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, sr in enumerate(results):
        # rank is "position in the ranked list", not scenario_idx.
        scenario_dir = output_dir / f"scenario_{rank:03d}"
        scenario_dir.mkdir(exist_ok=True)

        # spec.json is meant to be a self-contained "run manifest":
        # you can inspect it later to know exactly what produced these arrays.
        spec = {
            "name": sr.scenario.name,
            "tags": sr.scenario.tags,
            "config": asdict(sr.scenario.cfg),
            "metrics": asdict(sr.metrics),
            "scenario_idx": sr.scenario_idx,
        }
        with open(scenario_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2, default=str)

        # Save array fields as separate .npy files for easy loading.
        for field_name in _ARRAY_FIELDS:
            arr = getattr(sr.results, field_name)
            np.save(scenario_dir / f"{field_name}.npy", arr)


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------


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
) -> list[SweepResult] | list[MetricsOnlyResult] | SweepSummary:
    """Run scenarios against the same market data.

    Parameters
    ----------
    scenarios:
        Output of make_scenarios().
    bid_px, bid_sz, ask_px, ask_sz, mid_px, dv01, zscore,
    expected_yield_pnl_bps, package_yield_bps, hedge_ratios:
        Market data arrays - identical across all runs.
        NOTE: These are passed into market_data dict once and shared across scenarios.
    parallel:
        Deprecated - use backend="thread" instead. If True and backend is "serial",
        switches to "thread" and emits a DeprecationWarning.
    backend:
        "serial" (default), "thread", or "process".
    n_workers:
        Max workers for thread/process pools. Defaults to os.cpu_count().
    chunk_size:
        Scenarios per task. Defaults to max(10, n_scenarios // (10 * n_workers)).
        Larger chunk_size reduces overhead but can reduce load balancing.
    mode:
        "full" (default) stores full MRBacktestResults per scenario.
        "metrics_only" stores only scalar metrics + code distribution.
    keep_top_k_full:
        When mode="metrics_only", rerun the top-K scenarios in full mode and return
        a SweepSummary. 0 disables.
    save_top_full_dir:
        Directory to persist the top-K full results (JSON + npy). Only used when
        keep_top_k_full > 0.

    Returns
    -------
    list[SweepResult]
        When mode="full".
    list[MetricsOnlyResult]
        When mode="metrics_only" and keep_top_k_full == 0.
    SweepSummary
        When mode="metrics_only" and keep_top_k_full > 0.

    Implementation flow
    -------------------
    1) Pack market arrays into market_data dict.
    2) Dispatch all scenarios in either full or metrics-only mode.
    3) If metrics-only and keep_top_k_full > 0:
       - rank by total_pnl (tie-break by scenario_idx)
       - rerun only the top K in full mode
       - optionally persist those top-K artifacts.
    """
    # -- backward-compat: parallel=True -> thread backend ----------------
    if parallel and backend == "serial":
        warnings.warn(
            "parallel=True is deprecated; use backend='thread'",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = "thread"

    # Bundle all shared arrays. This dict is the single "market snapshot" passed
    # to run_backtest for every scenario.
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

    # -- resolve defaults -----------------------------------------------
    # n_workers default: best-effort CPU count, with 1 as safe fallback.
    workers = n_workers or os.cpu_count() or 1

    # Default chunk size: enough scenarios per task to reduce overhead.
    # For tiny sweeps, csize becomes 1.
    n = len(scenarios)
    csize = chunk_size if chunk_size is not None else max(10, n // (10 * workers))
    csize = max(1, csize)  # ensure at least 1

    # Pair each scenario with its stable index. That index is used for:
    # - stable ordering
    # - tie-break in top-K selection
    indexed = list(enumerate(scenarios))

    # -- full mode (backward-compatible default) -------------------------
    if mode == "full":
        # Full mode returns a list of SweepResult (heavy results per scenario).
        return _dispatch(indexed, market_data, backend, workers, csize, "full")

    # -- metrics_only mode -----------------------------------------------
    if mode != "metrics_only":
        raise ValueError(
            f"Unknown mode {mode!r}; choose from 'full', 'metrics_only'"
        )

    # Metrics-only dispatch returns lightweight results per scenario.
    metrics_results = _dispatch(
        indexed, market_data, backend, workers, csize, "metrics_only"
    )

    # If no top-K rerun is requested, return lightweight results directly.
    if keep_top_k_full <= 0:
        return metrics_results

    # -- top-K full rerun ------------------------------------------------
    # Current ranking rule:
    # - best total_pnl first
    # - tie-break by scenario_idx ascending
    #
    # This is deterministic and stable across backends.
    ranked = sorted(
        metrics_results,
        key=lambda r: (-r.total_pnl, r.scenario_idx),
    )
    top_k = ranked[:keep_top_k_full]

    # Convert back to (idx, scenario) pairs so we can dispatch rerun.
    top_indexed = [(r.scenario_idx, r.scenario) for r in top_k]

    # Rerun only top-K in full mode to get MRBacktestResults for plotting.
    top_full_results = _dispatch(
        top_indexed, market_data, backend, workers, csize, "full"
    )

    # Ensure top_full is sorted by ranked order as well. We use full metrics here
    # (r.metrics.total_pnl) but it should match metrics_results.total_pnl.
    top_full_results.sort(
        key=lambda r: (-r.metrics.total_pnl, r.scenario_idx),
    )

    # Optional artifact saving for top-K (so you can plot later without rerunning).
    if save_top_full_dir is not None:
        _save_top_full(top_full_results, save_top_full_dir)

    return SweepSummary(all_metrics=metrics_results, top_full=top_full_results)


# ---------------------------------------------------------------------------
# Ranking / summary
# ---------------------------------------------------------------------------

# Fields available on MetricsOnlyResult for ranking.
# This is a small, explicit allowlist to keep rank_results error messages helpful.
_METRICS_ONLY_FIELDS = frozenset(
    {"total_pnl", "final_equity", "n_trades", "max_drawdown", "sharpe_ratio"}
)


def rank_results(
    results: list[SweepResult] | list[MetricsOnlyResult],
    metric: str = "sharpe_ratio",
    top_n: int = 5,
    ascending: bool = False,
) -> list[SweepResult] | list[MetricsOnlyResult]:
    """Sort results by a metric field and return the top top_n.

    Parameters
    ----------
    metric:
        Attribute of BacktestMetrics (for SweepResult) OR
        a scalar field of MetricsOnlyResult.
    top_n:
        Number of results to return. If top_n exceeds len(results),
        all results are returned.
    ascending:
        If True, lower values rank first (useful for drawdown).

    Notes
    -----
    - This is a lightweight helper intended for notebooks/quick inspection.
    - For deterministic tuning selection policies (primary + tie-breaks),
      you'd typically implement a richer ranker; this stays intentionally simple.
    """
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
            raise ValueError(
                f"Unknown metric {metric!r}; choose from {sorted(valid_fields)}"
            )
        sorted_results = sorted(
            results,
            key=lambda r: getattr(r.metrics, metric),
            reverse=not ascending,
        )
    return sorted_results[:top_n]


def summary_table(
    results: list[SweepResult] | list[MetricsOnlyResult],
) -> pd.DataFrame:
    """One-row-per-scenario DataFrame with tag columns + all metrics.

    The DataFrame includes:
    - scenario name
    - grid parameter tags (each key becomes a column)
    - metrics columns:
      - for MetricsOnlyResult: a fixed subset of scalars
      - for SweepResult: all BacktestMetrics fields

    Returns
    -------
    pd.DataFrame
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        # Start with scenario identity and grid tags.
        row: dict[str, Any] = {"name": r.scenario.name}
        row.update(r.scenario.tags)

        # Add metrics depending on result type.
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
