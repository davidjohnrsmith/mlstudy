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
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import SweepResult, SweepResultLight, SweepSummary

logger = logging.getLogger(__name__)

ARRAY_FIELDS = [
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


class SweepPersister:
    @staticmethod
    def save_top_full(results: list[SweepResult], output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for rank, sr in enumerate(results):
            scenario_dir = output_dir / f"scenario_{rank:03d}"
            scenario_dir.mkdir(exist_ok=True)

            spec = {
                "name": sr.scenario.name,
                "tags": sr.scenario.tags,
                "config": asdict(sr.scenario.cfg),
                "metrics": asdict(sr.metrics),
                "scenario_idx": sr.scenario_idx,
            }
            with open(scenario_dir / "spec.json", "w") as f:
                json.dump(spec, f, indent=2, default=str)

            for field_name in ARRAY_FIELDS:
                arr = getattr(sr.results, field_name)
                np.save(scenario_dir / f"{field_name}.npy", arr)

    @staticmethod
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

    @staticmethod
    def _save_summary_table(output_dir: Path, table: pd.DataFrame) -> None:
        """Write the summary table as CSV."""
        table.to_csv(output_dir / "summary.csv", index=False)

    @staticmethod
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

    @staticmethod
    def _save_metrics_results(
        output_dir: Path,
        results: list[SweepResultLight],
    ) -> None:
        """Save all metrics-only results as a single CSV."""
        rows = []
        for r in results:
            row: dict[str, Any] = {
                "scenario_idx": r.scenario_idx,
                "name": r.scenario.name,
            }
            row.update(r.scenario.tags)
            row.update(asdict(r.metrics))
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "all_metrics.csv", index=False)

    @staticmethod
    def _save_full_results(
        output_dir: Path,
        results: list[SweepResult],
        label: str = "full",
    ) -> None:
        """Save full backtest results (arrays + spec) per scenario."""
        full_dir = output_dir / label
        SweepPersister.save_top_full(results, full_dir)

    @staticmethod
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

    @staticmethod
    def persist(
        output_dir: Path,
        cfg: SweepConfig,
        raw: list[SweepResult] | list[SweepResultLight] | SweepSummary,
        table: pd.DataFrame,
        n_scenarios: int,
        elapsed: float,
        zscore: np.ndarray | None = None,
    ) -> None:
        SweepPersister._save_config_snapshot(output_dir, cfg)
        SweepPersister._save_run_metadata(output_dir, cfg, n_scenarios, elapsed)
        SweepPersister._save_summary_table(output_dir, table)

        if isinstance(raw, SweepSummary):
            SweepPersister._save_metrics_results(output_dir, raw.all_metrics)
            if raw.top_full:
                SweepPersister._save_full_results(output_dir, raw.top_full, label="top_full")
                SweepPersister._save_scenario_plots(output_dir, raw.top_full, zscore, label="plots")
        elif raw and isinstance(raw[0], SweepResultLight):
            SweepPersister._save_metrics_results(output_dir, raw)
        elif raw and isinstance(raw[0], SweepResult):
            SweepPersister._save_full_results(output_dir, raw, label="full")
            SweepPersister._save_scenario_plots(output_dir, raw, zscore, label="plots")
