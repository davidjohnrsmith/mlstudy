"""Portfolio-specific sweep persistence.

Delegates generic helpers (metadata, summary, config snapshot, metrics CSVs)
to ``common.sweep.sweep_persist`` and adds portfolio-specific concerns:
saving full results with numpy arrays.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
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
from mlstudy.trading.backtest.portfolio.configs.sweep_config import PortfolioSweepConfig
from mlstudy.trading.backtest.common.sweep.sweep_types import SweepResult, SweepResultLight, SweepSummary

logger = logging.getLogger(__name__)

# Array field names to persist from PortfolioBacktestResults
PORTFOLIO_ARRAY_FIELDS = [
    "positions",
    "cash",
    "equity",
    "pnl",
    "gross_pnl",
    "codes",
    "n_trades_bar",
    "cooldown",
    "hedge_positions",
    "tr_bar",
    "tr_instrument",
    "tr_side",
    "tr_qty_req",
    "tr_qty_fill",
    "tr_dv01_req",
    "tr_dv01_fill",
    "tr_alpha",
    "tr_fair_type",
    "tr_vwap",
    "tr_mid",
    "tr_cost",
    "tr_code",
    "tr_hedge_sizes",
    "tr_hedge_vwaps",
    "tr_hedge_fills",
    "tr_hedge_cost",
]


class PortfolioSweepPersister:
    @staticmethod
    def save_top_full(results: list[SweepResult], output_dir: str | Path) -> None:
        """Save full results (spec + numpy arrays) per scenario."""
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

            for field_name in PORTFOLIO_ARRAY_FIELDS:
                arr = getattr(sr.results, field_name)
                np.save(scenario_dir / f"{field_name}.npy", arr)

    @staticmethod
    def persist(
        output_dir: Path,
        cfg: PortfolioSweepConfig,
        raw: list[SweepResult] | list[SweepResultLight] | SweepSummary,
        table: pd.DataFrame,
        n_scenarios: int,
        elapsed: float,
        top_n: int = 10,
    ) -> None:
        """Persist sweep results to disk."""
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
                PortfolioSweepPersister.save_top_full(raw.top_full, full_dir)
        elif raw and isinstance(raw[0], SweepResultLight):
            save_metrics_results(output_dir, raw)
            save_metrics_averages(output_dir, raw, top_n)
        elif raw and isinstance(raw[0], SweepResult):
            full_dir = output_dir / "full"
            PortfolioSweepPersister.save_top_full(raw, full_dir)
