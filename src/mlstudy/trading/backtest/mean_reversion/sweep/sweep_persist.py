"""MR-specific sweep persistence.

Delegates generic helpers (metadata, summary, config snapshot, metrics CSVs)
to ``common.sweep.sweep_persist`` and adds MR-specific concerns: saving full
results with numpy arrays and generating scenario plots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.common.sweep.sweep_persist import (
    save_run_metadata,
    save_summary_table,
    save_config_snapshot,
    save_metrics_results,
    save_metrics_averages,
    summary_table,
)
from mlstudy.trading.backtest.mean_reversion.configs.sweep_config import SweepConfig
from mlstudy.trading.backtest.mean_reversion.single_backtest.results import ARRAY_FIELDS
from mlstudy.trading.backtest.common.sweep.sweep_types import (
    SweepResult,
    SweepResultLight,
    SweepSummary,
)

logger = logging.getLogger(__name__)


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

            # Persist DataFrames as CSV for easy inspection
            if sr.results.bar_df is not None:
                sr.results.bar_df.to_csv(scenario_dir / "bar_df.csv", index=False)
            if sr.results.trade_df is not None:
                sr.results.trade_df.to_csv(scenario_dir / "trade_df.csv", index=False)

    @staticmethod
    def _save_scenario_plots(
        output_dir: Path,
        results: list[SweepResult],
        zscore: np.ndarray | None,
        label: str = "plots",
    ) -> None:
        """Generate and save scenario dashboard plots."""
        try:
            from mlstudy.trading.backtest.mean_reversion.sweep.plots import plot_scenario
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
                fig = plot_scenario(fs, save_path=save_path)
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
        top_n: int = 10,
    ) -> None:
        # Generic persistence via common helpers
        save_config_snapshot(
            output_dir, cfg.grid_name, cfg.base_config, cfg.grid,
            cfg.sweep_kwargs, cfg.ranking_plan,
        )
        save_run_metadata(
            output_dir, cfg.grid_name, cfg.base_config, cfg.grid,
            cfg.sweep_kwargs, n_scenarios, elapsed,
        )
        save_summary_table(output_dir, table)

        if isinstance(raw, SweepSummary):
            save_metrics_results(output_dir, raw.all_metrics)
            save_metrics_averages(output_dir, raw.all_metrics, top_n)
            if raw.top_full:
                full_dir = output_dir / "top_full"
                SweepPersister.save_top_full(raw.top_full, full_dir)
                SweepPersister._save_scenario_plots(output_dir, raw.top_full, zscore, label="plots")
        elif raw and isinstance(raw[0], SweepResultLight):
            save_metrics_results(output_dir, raw)
            save_metrics_averages(output_dir, raw, top_n)
        elif raw and isinstance(raw[0], SweepResult):
            full_dir = output_dir / "full"
            SweepPersister.save_top_full(raw, full_dir)
            SweepPersister._save_scenario_plots(output_dir, raw, zscore, label="plots")
