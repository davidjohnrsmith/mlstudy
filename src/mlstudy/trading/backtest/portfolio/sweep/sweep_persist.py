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
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan, SweepRanker
from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics
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
    "net_instrument_dv01",
    "gross_instrument_dv01",
    "net_hedge_dv01",
    "gross_hedge_dv01",
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

            # Persist instrument/hedge ID → index mapping
            id_map = {
                "instrument_ids": {
                    str(i): str(iid)
                    for i, iid in enumerate(sr.results.instrument_ids)
                } if sr.results.instrument_ids else {},
                "hedge_ids": {
                    str(i): str(hid)
                    for i, hid in enumerate(sr.results.hedge_ids)
                } if sr.results.hedge_ids else {},
            }
            with open(scenario_dir / "id_map.json", "w") as f:
                json.dump(id_map, f, indent=2)

            # Persist DataFrames as CSV for easy inspection
            if sr.results.bar_df is not None:
                sr.results.bar_df.to_csv(scenario_dir / "bar_df.csv", index=False)
            if sr.results.close_bar_df is not None:
                sr.results.close_bar_df.to_csv(scenario_dir / "close_bar_df.csv", index=False)
            if sr.results.trade_df is not None:
                sr.results.trade_df.to_csv(scenario_dir / "trade_df.csv", index=False)
            if sr.results.instrument_pnl_df is not None:
                sr.results.instrument_pnl_df.to_csv(scenario_dir / "instrument_pnl_df.csv", index=False)

    @staticmethod
    def build_param_leaderboard(
        table: pd.DataFrame,
        grid_keys: list[str],
        ranking_plan: RankingPlan | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Aggregate metrics by param combo, then rank.

        Parameters
        ----------
        table : pd.DataFrame
            Summary table from ``summary_table()`` containing grid param
            columns and metric columns.
        grid_keys : list[str]
            Grid parameter names to group by.
        ranking_plan : RankingPlan or None
            Ranking plan for ordering.  Defaults to ``RankingPlan()``.
        top_n : int
            Return the top *N* param combos.

        Returns
        -------
        pd.DataFrame
            Ranked leaderboard with ``rank`` column (1-based).
        """
        if table.empty:
            return pd.DataFrame()

        valid_keys = [k for k in grid_keys if k in table.columns]
        if not valid_keys:
            return pd.DataFrame()

        if ranking_plan is None:
            ranking_plan = RankingPlan()

        metric_fields = [f.name for f in BacktestMetrics.__dataclass_fields__.values()]
        available_metrics = [m for m in metric_fields if m in table.columns]
        if not available_metrics:
            return pd.DataFrame()

        agg_cols = valid_keys + available_metrics
        grouped = table[agg_cols].groupby(valid_keys, dropna=False)[available_metrics]
        agg = grouped.mean().reset_index()

        ranked = SweepRanker.rank_dataframe(agg, ranking_plan)
        return ranked.head(top_n).reset_index(drop=True)

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

        # Param leaderboard
        grid_keys = list(cfg.grid.keys())
        leaderboard = PortfolioSweepPersister.build_param_leaderboard(
            table, grid_keys, cfg.ranking_plan, top_n,
        )
        if not leaderboard.empty:
            leaderboard.to_csv(output_dir / "param_leaderboard.csv", index=False)

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

        # Generate plots for full scenarios if matplotlib available
        full_results = None
        if isinstance(raw, SweepSummary) and raw.top_full:
            full_results = raw.top_full
        elif raw and isinstance(raw[0], SweepResult):
            full_results = raw

        if full_results:
            _save_scenario_plots(
                full_results, output_dir / "plots",
                plot_config=cfg.plot_config,
            )


def _save_scenario_plots(
    results: list[SweepResult],
    plots_dir: Path,
    plot_config: dict[str, Any] | None = None,
) -> None:
    """Generate dashboard and detail PNGs for full scenarios."""
    try:
        from mlstudy.trading.backtest.portfolio.sweep.plots import (
            plot_scenario,
            plot_scenario_detail,
        )
        from mlstudy.trading.backtest.portfolio.sweep.sweep_results_reader import (
            PortfolioFullScenario,
        )
    except ImportError:
        logger.debug("matplotlib not available; skipping plot generation")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating plots for %d scenarios in %s", len(results), plots_dir)

    for rank, sr in enumerate(results):
        try:
            sc = PortfolioFullScenario(
                spec={
                    "name": sr.scenario.name,
                    "tags": sr.scenario.tags,
                    "config": asdict(sr.scenario.cfg),
                    "metrics": asdict(sr.metrics),
                    "scenario_idx": sr.scenario_idx,
                },
                results=sr.results,
                directory=plots_dir,
            )
            import matplotlib.pyplot as plt

            fig = plot_scenario(
                sc, save_path=plots_dir / f"scenario_{rank:03d}.png",
                plot_config=plot_config,
            )
            plt.close(fig)

            fig_d = plot_scenario_detail(
                sc, save_path=plots_dir / f"scenario_{rank:03d}_detail.png",
            )
            plt.close(fig_d)
        except Exception:
            logger.warning("Failed to plot scenario %d", rank, exc_info=True)
