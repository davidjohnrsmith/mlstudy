"""End-to-end sweep interface: load config, build scenarios, run sweep, persist results.

Usage from a script or notebook::

    from mlstudy.trading.backtest.mean_reversion.sweep_runner import run_sweep_from_config

    # Option A: by YAML path
    summary = run_sweep_from_config("configs/mr_grid_v1.yaml")

    # Option B: by config-map name
    summary = run_sweep_from_config("mr_grid_v1")

    # Option C: with market data already loaded
    summary = run_sweep_from_config("mr_grid_v1", market_data=my_market_data)

The function returns a ``SweepRunResult`` that bundles:
- the parsed config
- all raw results (list or SweepSummary)
- the summary table as a DataFrame
- the output directory where artifacts are saved
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .sweep import run_sweep, summary_table
from .sweep_build import make_scenarios
from .sweep_config import SweepConfig, load_sweep_config, load_sweep_config_by_name
from .sweep_types import MetricsOnlyResult, SweepResult, SweepSummary


@dataclass
class SweepRunResult:
    """Everything produced by a single sweep run."""

    config: SweepConfig
    raw: list[SweepResult] | list[MetricsOnlyResult] | SweepSummary
    table: pd.DataFrame
    output_dir: Path | None

    @property
    def all_metrics(self) -> list[MetricsOnlyResult] | None:
        if isinstance(self.raw, SweepSummary):
            return self.raw.all_metrics
        if self.raw and isinstance(self.raw[0], MetricsOnlyResult):
            return self.raw
        return None

    @property
    def top_full(self) -> list[SweepResult] | None:
        if isinstance(self.raw, SweepSummary):
            return self.raw.top_full
        if self.raw and isinstance(self.raw[0], SweepResult):
            return self.raw
        return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_run_metadata(
    output_dir: Path,
    cfg: SweepConfig,
    n_scenarios: int,
    elapsed_seconds: float | None,
) -> None:
    """Write run metadata to ``output_dir/run_meta.json``."""
    meta = {
        "grid_name": cfg.grid_name,
        "n_scenarios": n_scenarios,
        "base_config": asdict(cfg.base_config),
        "grid": {k: list(v) for k, v in cfg.grid.items()},
        "sweep_kwargs": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in cfg.sweep_kwargs.items()
            if k != "ranking_plan"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if elapsed_seconds is not None:
        meta["elapsed_seconds"] = round(elapsed_seconds, 2)
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _save_summary_table(output_dir: Path, table: pd.DataFrame) -> None:
    """Write the summary table as CSV."""
    table.to_csv(output_dir / "summary.csv", index=False)


def _save_config_snapshot(output_dir: Path, cfg: SweepConfig) -> None:
    """Snapshot the parsed config as JSON for reproducibility."""
    import yaml

    snapshot: dict[str, Any] = {
        "grid_name": cfg.grid_name,
        "base_config": asdict(cfg.base_config),
        "grid": {k: list(v) for k, v in cfg.grid.items()},
    }
    if cfg.ranking_plan is not None:
        rp = cfg.ranking_plan
        snapshot["rank"] = {
            "primary_metrics": [list(t) for t in rp.primary_metrics],
            "tie_metrics": [list(t) for t in rp.tie_metrics],
            "primary_params": [list(t) for t in rp.primary_params],
            "tie_params": [list(t) for t in rp.tie_params],
        }
    sweep_kw = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in cfg.sweep_kwargs.items()
        if k != "ranking_plan"
    }
    snapshot["sweep"] = sweep_kw

    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)


def _save_metrics_results(
    output_dir: Path,
    results: list[MetricsOnlyResult],
) -> None:
    """Save all metrics-only results as a single CSV."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "scenario_idx": r.scenario_idx,
            "name": r.scenario.name,
        }
        row.update(r.scenario.tags)
        row.update(r.metrics_dict())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "all_metrics.csv", index=False)


def _save_full_results(
    output_dir: Path,
    results: list[SweepResult],
    label: str = "full",
) -> None:
    """Save full backtest results (arrays + spec) per scenario."""
    from .sweep_persist import _save_top_full

    full_dir = output_dir / label
    _save_top_full(results, full_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sweep_from_config(
    config: str | Path | SweepConfig,
    *,
    market_data: dict[str, Any] | None = None,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
    config_map_path: str | Path | None = None,
    **market_data_kwargs: Any,
) -> SweepRunResult:
    """End-to-end: load config, build scenarios, run sweep, save results.

    Parameters
    ----------
    config : str, Path, or SweepConfig
        One of:
        - A ``SweepConfig`` object (already loaded).
        - A path to a YAML config file (contains ``/`` or ends with ``.yaml``).
        - A config-map name (looked up via ``config_map.yaml``).
    market_data : dict, optional
        Pre-loaded market data arrays.  Keys must match ``run_sweep``
        signature: ``bid_px``, ``bid_sz``, ``ask_px``, ``ask_sz``,
        ``mid_px``, ``dv01``, ``zscore``, ``expected_yield_pnl_bps``,
        ``package_yield_bps``, ``hedge_ratios``.
        If *None*, you must pass them as ``**market_data_kwargs``.
    data_path : str or Path, optional
        Directory containing parquet files.  Passed to
        ``BacktestDataLoader.load()`` when the config has a ``data``
        section.  Use this to keep the YAML config platform-independent
        and supply the data directory at runtime.
    output_dir : str or Path, optional
        Directory to save results.  Defaults to ``runs/<grid_name>/``.
    save : bool
        Whether to persist results to disk (default True).
    config_map_path : str or Path, optional
        Path to the config map YAML (only used when *config* is a name).
    **market_data_kwargs
        Market data arrays passed as keyword arguments.  Merged with
        *market_data* dict (kwargs take precedence).

    Returns
    -------
    SweepRunResult
    """
    import time

    # --- 1. Load config -------------------------------------------------------
    cfg = _resolve_config(config, config_map_path)

    # --- 2. Merge market data -------------------------------------------------
    md = dict(market_data or {})
    md.update(market_data_kwargs)
    if not md and cfg.data_loader is not None:
        logger.info("Loading market data from config data section")
        loaded = cfg.data_loader.load(data_path=data_path)
        md = loaded.to_dict()
    if not md:
        raise ValueError(
            "No market data provided.  Pass market_data=dict(...), "
            "individual arrays as keyword arguments, or add a 'data' "
            "section to the YAML config."
        )

    _REQUIRED_KEYS = {
        "bid_px", "bid_sz", "ask_px", "ask_sz", "mid_px",
        "dv01", "zscore", "expected_yield_pnl_bps",
        "package_yield_bps", "hedge_ratios",
    }
    missing = _REQUIRED_KEYS - set(md)
    if missing:
        raise ValueError(f"Missing market data keys: {sorted(missing)}")

    # --- 3. Build scenarios ---------------------------------------------------
    scenarios = make_scenarios(
        cfg.base_config,
        cfg.grid,
        name_prefix=cfg.grid_name,
    )

    # --- 4. Run sweep ---------------------------------------------------------
    t0 = time.perf_counter()
    raw = run_sweep(scenarios, **md, **cfg.sweep_kwargs)
    elapsed = time.perf_counter() - t0

    # --- 5. Build summary table -----------------------------------------------
    if isinstance(raw, SweepSummary):
        table = summary_table(raw.all_metrics)
    else:
        table = summary_table(raw)

    # --- 6. Persist results ---------------------------------------------------
    resolved_output_dir = None
    if save:
        resolved_output_dir = _resolve_output_dir(output_dir, cfg)
        _persist(resolved_output_dir, cfg, raw, table, len(scenarios), elapsed)

    return SweepRunResult(
        config=cfg,
        raw=raw,
        table=table,
        output_dir=resolved_output_dir,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _resolve_config(
    config: str | Path | SweepConfig,
    config_map_path: str | Path | None,
) -> SweepConfig:
    if isinstance(config, SweepConfig):
        return config

    path = Path(config)
    # Heuristic: if it looks like a file path, load directly
    if path.suffix in (".yaml", ".yml") or "/" in str(config) or "\\" in str(config):
        return load_sweep_config(path)

    # Otherwise treat as a config-map name
    return load_sweep_config_by_name(str(config), config_map_path=config_map_path)


def _resolve_output_dir(
    output_dir: str | Path | None,
    cfg: SweepConfig,
) -> Path:
    if output_dir is not None:
        d = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("runs") / cfg.grid_name / timestamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist(
    output_dir: Path,
    cfg: SweepConfig,
    raw: list[SweepResult] | list[MetricsOnlyResult] | SweepSummary,
    table: pd.DataFrame,
    n_scenarios: int,
    elapsed: float,
) -> None:
    _save_config_snapshot(output_dir, cfg)
    _save_run_metadata(output_dir, cfg, n_scenarios, elapsed)
    _save_summary_table(output_dir, table)

    if isinstance(raw, SweepSummary):
        _save_metrics_results(output_dir, raw.all_metrics)
        if raw.top_full:
            _save_full_results(output_dir, raw.top_full, label="top_full")
    elif raw and isinstance(raw[0], MetricsOnlyResult):
        _save_metrics_results(output_dir, raw)
    elif raw and isinstance(raw[0], SweepResult):
        _save_full_results(output_dir, raw, label="full")
