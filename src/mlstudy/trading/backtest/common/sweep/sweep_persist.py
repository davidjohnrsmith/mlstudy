"""Strategy-agnostic sweep persistence helpers.

Handles saving metadata, summary tables, config snapshots, and metrics CSVs.
Strategy-specific persistence (full results with arrays, plots) is left to
the strategy-specific sweep modules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics
from .sweep_types import SweepResultLight, SweepResult, SweepSummary

logger = logging.getLogger(__name__)


def save_run_metadata(
    output_dir: Path,
    grid_name: str,
    base_config: Any,
    grid: dict,
    sweep_kwargs: dict,
    n_scenarios: int,
    elapsed_seconds: float | None,
) -> None:
    """Write run metadata to ``output_dir/run_meta.json``."""
    meta = {
        "grid_name": grid_name,
        "n_scenarios": n_scenarios,
        "base_config": asdict(base_config),
        "grid": {k: list(v) for k, v in grid.items()},
        "sweep_kwargs": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in sweep_kwargs.items()
            if k != "ranking_plan"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if elapsed_seconds is not None:
        meta["elapsed_seconds"] = round(elapsed_seconds, 2)
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def save_summary_table(output_dir: Path, table: pd.DataFrame) -> None:
    """Write the summary table as CSV."""
    table.to_csv(output_dir / "summary.csv", index=False)


def save_config_snapshot(
    output_dir: Path,
    grid_name: str,
    base_config: Any,
    grid: dict,
    sweep_kwargs: dict,
    ranking_plan: Any | None = None,
) -> None:
    """Snapshot the parsed config as YAML for reproducibility."""
    import yaml

    snapshot: dict[str, Any] = {
        "grid_name": grid_name,
        "base_config": asdict(base_config),
        "grid": {k: list(v) for k, v in grid.items()},
    }
    if ranking_plan is not None:
        rp = ranking_plan
        snapshot["rank"] = {
            "primary_metrics": [list(t) for t in rp.primary_metrics],
            "tie_metrics": [list(t) for t in rp.tie_metrics],
            "primary_params": [list(t) for t in rp.primary_params],
            "tie_params": [list(t) for t in rp.tie_params],
        }
    kw = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in sweep_kwargs.items()
        if k != "ranking_plan"
    }
    snapshot["sweep"] = kw

    with open(output_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)


def save_metrics_results(
    output_dir: Path,
    results: list[SweepResultLight],
) -> None:
    """Save all metrics-only results as a single CSV with rank column."""
    rows = []
    for rank_idx, r in enumerate(results, 1):
        row: dict[str, Any] = {
            "rank": rank_idx,
            "scenario_idx": r.scenario_idx,
            "name": r.scenario.name,
        }
        row.update(r.scenario.tags)
        row.update(asdict(r.metrics))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "all_metrics.csv", index=False)


def save_metrics_averages(
    output_dir: Path,
    results: list[SweepResultLight],
    top_n: int,
) -> None:
    """Save avg_top_n.csv and avg_all.csv."""
    if not results:
        return

    metric_fields = [f.name for f in BacktestMetrics.__dataclass_fields__.values()]

    def _avg_row(subset: list[SweepResultLight]) -> dict[str, float]:
        if not subset:
            return {}
        row: dict[str, float] = {}
        for name in metric_fields:
            vals = [getattr(r.metrics, name) for r in subset]
            row[name] = sum(vals) / len(vals)
        return row

    avg_all = _avg_row(results)
    pd.DataFrame([avg_all]).to_csv(output_dir / "avg_all.csv", index=False)

    top_subset = results[:top_n]
    avg_top = _avg_row(top_subset)
    pd.DataFrame([avg_top]).to_csv(output_dir / "avg_top_n.csv", index=False)


def summary_table(
    results: list[SweepResult] | list[SweepResultLight],
) -> pd.DataFrame:
    """Build a summary DataFrame from sweep results."""
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {"name": r.scenario.name}
        row.update(r.scenario.tags)
        row.update(asdict(r.metrics))
        rows.append(row)
    return pd.DataFrame(rows)
