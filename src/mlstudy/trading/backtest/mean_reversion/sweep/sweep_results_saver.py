from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.mean_reversion.configs.sweep_config import SweepConfig
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import SweepResult, MetricsOnlyResult, SweepSummary

logger = logging.getLogger(__name__)


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


def _save_scenario_plots(
    output_dir: Path,
    results: list[SweepResult],
    zscore: np.ndarray | None,
    label: str = "plots",
) -> None:
    """Generate and save scenario dashboard plots."""
    try:
        from .plots import plot_scenario
        from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import FullScenario
    except ImportError:
        logger.debug("matplotlib not available, skipping plot generation")
        return

    plots_dir = output_dir / label
    plots_dir.mkdir(parents=True, exist_ok=True)

    for rank, sr in enumerate(results):
        spec = {
            "name": sr.scenario.name,
            "tags": sr.scenario.tags,
            "config": asdict(sr.scenario.cfg),
            "metrics": asdict(sr.metrics),
            "scenario_idx": sr.scenario_idx,
        }
        fs = FullScenario(spec=spec, results=sr.results, directory=plots_dir)

        save_path = plots_dir / f"scenario_{rank:03d}.png"
        try:
            fig = plot_scenario(fs, zscore=zscore, save_path=save_path)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot scenario %d", rank, exc_info=True)


def _persist(
    output_dir: Path,
    cfg: SweepConfig,
    raw: list[SweepResult] | list[MetricsOnlyResult] | SweepSummary,
    table: pd.DataFrame,
    n_scenarios: int,
    elapsed: float,
    zscore: np.ndarray | None = None,
) -> None:
    _save_config_snapshot(output_dir, cfg)
    _save_run_metadata(output_dir, cfg, n_scenarios, elapsed)
    _save_summary_table(output_dir, table)

    if isinstance(raw, SweepSummary):
        _save_metrics_results(output_dir, raw.all_metrics)
        if raw.top_full:
            _save_full_results(output_dir, raw.top_full, label="top_full")
            _save_scenario_plots(output_dir, raw.top_full, zscore, label="plots")
    elif raw and isinstance(raw[0], MetricsOnlyResult):
        _save_metrics_results(output_dir, raw)
    elif raw and isinstance(raw[0], SweepResult):
        _save_full_results(output_dir, raw, label="full")
        _save_scenario_plots(output_dir, raw, zscore, label="plots")
