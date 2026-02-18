"""Orchestrate parameter sweeps across multiple reference instruments.

Usage::

    from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_runner import (
        MultiRefSweepRunner,
    )

    result = MultiRefSweepRunner.run(
        config="mr_grid_v1",
        ref_instrument_ids=["UST_2Y", "UST_5Y", "UST_10Y"],
        data_path="/data/backtest",
    )

    result.per_ref_best          # best scenario per ref
    result.param_leaderboard     # param combos ranked across refs
    result.per_ref_results       # dict of SweepRunResult per ref
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..configs.sweep_config import SweepConfig
from ..configs.utils import _resolve_config
from .sweep_rank import RankingPlan, SweepRanker
from .sweep_runner import SweepRunResult, SweepRunner
from ...metrics.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class MultiRefSweepResult:
    """Everything produced by a multi-ref sweep run."""

    output_dir: Path | None
    cross_ref_summary: pd.DataFrame
    per_ref_best: pd.DataFrame
    param_leaderboard: pd.DataFrame
    per_ref_results: dict[str, SweepRunResult]


class MultiRefSweepRunner:
    @staticmethod
    def run(
        config: str | Path | SweepConfig,
        ref_instrument_ids: list[str],
        *,
        data_path: str | Path | None = None,
        instrument_ids: list[str] | None = None,
        output_dir: str | Path | None = None,
        save: bool = True,
        top_n: int = 10,
        config_map_path: str | Path | None = None,
        **sweep_kwargs: Any,
    ) -> MultiRefSweepResult:
        """Run a parameter sweep for each ref instrument and produce cross-ref analytics.

        Parameters
        ----------
        config : str, Path, or SweepConfig
            Sweep configuration (YAML path, config-map name, or object).
        ref_instrument_ids : list[str]
            Reference instrument IDs to sweep over sequentially.
        data_path : str or Path, optional
            Directory containing parquet files.
        instrument_ids : list[str], optional
            Ordered list of instrument IDs. Auto-detected per ref if None.
        output_dir : str or Path, optional
            Base output directory. Defaults to ``runs/<grid_name>/<timestamp>``.
        save : bool
            Whether to persist results to disk (default True).
        top_n : int
            Number of top scenarios to use for averages and leaderboards.
        config_map_path : str or Path, optional
            Path to the config map YAML.
        **sweep_kwargs
            Additional keyword arguments forwarded to
            ``SweepRunner.run_sweep_from_config``.

        Returns
        -------
        MultiRefSweepResult
        """
        t0 = time.perf_counter()

        # Resolve config once so each per-ref run uses the same object
        cfg = _resolve_config(config, config_map_path)
        ranking_plan = cfg.ranking_plan

        # Resolve base output directory
        if output_dir is not None:
            base_output_dir = Path(output_dir) / cfg.grid_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = Path("runs") / cfg.grid_name / timestamp
        if save:
            base_output_dir.mkdir(parents=True, exist_ok=True)

        per_ref_results: dict[str, SweepRunResult] = {}
        all_tables: list[pd.DataFrame] = []
        grid_keys = list(cfg.grid.keys())
        metric_fields = [f.name for f in fields(BacktestMetrics)]

        for i, ref_id in enumerate(ref_instrument_ids):
            logger.info(
                "[%d/%d] Running sweep for %s",
                i + 1,
                len(ref_instrument_ids),
                ref_id,
            )
            ref_output_dir = base_output_dir / ref_id
            result = SweepRunner.run_sweep_from_config(
                cfg,
                ref_instrument_id=ref_id,
                instrument_ids=instrument_ids,
                data_path=data_path,
                output_dir=ref_output_dir,
                save=save,
                **sweep_kwargs,
            )
            per_ref_results[ref_id] = result

            # Save per-ref param leaderboard
            if save:
                MultiRefSweepRunner._save_per_ref_param_leaderboard(
                    ref_output_dir, result.table, grid_keys,
                    metric_fields, ranking_plan, top_n,
                )

            ref_table = result.table.copy()
            ref_table.insert(0, "ref_instrument_id", ref_id)
            all_tables.append(ref_table)

        # Build cross-ref analytics
        cross_ref_summary = MultiRefSweepRunner._build_cross_ref_summary(all_tables)
        per_ref_best = MultiRefSweepRunner._build_per_ref_best(
            cross_ref_summary, ranking_plan,
        )
        param_leaderboard = MultiRefSweepRunner._build_param_leaderboard(
            cross_ref_summary, grid_keys, metric_fields, ranking_plan, top_n,
        )

        # Cross-ref averages
        avg_top_n_refs = MultiRefSweepRunner._build_cross_ref_averages(
            per_ref_results, metric_fields, top_n, mode="top_n",
        )
        avg_all_refs = MultiRefSweepRunner._build_cross_ref_averages(
            per_ref_results, metric_fields, top_n, mode="all",
        )

        elapsed = time.perf_counter() - t0

        # Persist cross-ref analytics
        resolved_output_dir: Path | None = None
        if save:
            resolved_output_dir = base_output_dir
            MultiRefSweepRunner._save_analytics(
                resolved_output_dir,
                cross_ref_summary,
                per_ref_best,
                param_leaderboard,
                avg_top_n_refs,
                avg_all_refs,
                ref_instrument_ids,
                elapsed,
                cfg,
            )
        else:
            resolved_output_dir = None

        return MultiRefSweepResult(
            output_dir=resolved_output_dir,
            cross_ref_summary=cross_ref_summary,
            per_ref_best=per_ref_best,
            param_leaderboard=param_leaderboard,
            per_ref_results=per_ref_results,
        )

    @staticmethod
    def _build_cross_ref_summary(
        all_tables: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Concatenate per-ref summary tables into one DataFrame."""
        if not all_tables:
            return pd.DataFrame()
        return pd.concat(all_tables, ignore_index=True)

    @staticmethod
    def _build_per_ref_best(
        cross_ref_summary: pd.DataFrame,
        ranking_plan: RankingPlan | None,
    ) -> pd.DataFrame:
        """For each ref, select the rank-1 scenario and re-rank cross-ref."""
        if cross_ref_summary.empty:
            return pd.DataFrame()

        # Rank within each ref, take the best
        best_rows = []
        for ref_id, group in cross_ref_summary.groupby("ref_instrument_id"):
            ranked = SweepRanker.rank_dataframe(group.drop(columns=["ref_instrument_id"], errors="ignore"), ranking_plan)
            best_row = ranked.iloc[:1].copy()
            best_row.insert(0, "ref_instrument_id", ref_id)
            if "rank" in best_row.columns:
                best_row = best_row.drop(columns=["rank"])
            best_rows.append(best_row)

        if not best_rows:
            return pd.DataFrame()

        combined = pd.concat(best_rows, ignore_index=True)
        # Re-rank the best rows across refs
        non_ref = combined.drop(columns=["ref_instrument_id"])
        ranked = SweepRanker.rank_dataframe(non_ref, ranking_plan)
        ranked.insert(1, "ref_instrument_id", combined.loc[ranked["rank"].values - 1, "ref_instrument_id"].values)
        return ranked

    @staticmethod
    def _build_param_leaderboard(
        cross_ref_summary: pd.DataFrame,
        grid_keys: list[str],
        metric_fields: list[str],
        ranking_plan: RankingPlan | None,
        top_n: int,
    ) -> pd.DataFrame:
        """Aggregate metrics by param combo across refs, then rank."""
        if cross_ref_summary.empty:
            return pd.DataFrame()

        valid_keys = [k for k in grid_keys if k in cross_ref_summary.columns]
        if not valid_keys:
            return pd.DataFrame()

        # For each ref, take top N scenarios (ranked by plan), then concatenate
        top_rows: list[pd.DataFrame] = []
        for ref_id, group in cross_ref_summary.groupby("ref_instrument_id"):
            ref_data = group.drop(columns=["ref_instrument_id"], errors="ignore")
            ranked = SweepRanker.rank_dataframe(ref_data, ranking_plan)
            top_rows.append(ranked.head(top_n).drop(columns=["rank"]))

        if not top_rows:
            return pd.DataFrame()

        combined = pd.concat(top_rows, ignore_index=True)

        # Average ALL metric columns by param combo
        available_metrics = [m for m in metric_fields if m in combined.columns]
        agg_cols = valid_keys + available_metrics
        combined_subset = combined[agg_cols]

        grouped = combined_subset.groupby(valid_keys, dropna=False)[available_metrics]
        agg = grouped.mean().reset_index()

        # Rank using plan
        ranked = SweepRanker.rank_dataframe(agg, ranking_plan)
        return ranked.head(top_n).reset_index(drop=True)

    @staticmethod
    def _build_cross_ref_averages(
        per_ref_results: dict[str, SweepRunResult],
        metric_fields: list[str],
        top_n: int,
        mode: str,
    ) -> pd.DataFrame:
        """Build one-row average across refs.

        mode="top_n": for each ref, average its top N scenarios' metrics, then average across refs.
        mode="all": for each ref, average all scenarios' metrics, then average across refs.
        """
        if not per_ref_results:
            return pd.DataFrame()

        ref_avgs: list[dict[str, float]] = []
        for ref_id, result in per_ref_results.items():
            all_metrics = result.all_metrics
            if not all_metrics:
                continue

            subset = all_metrics[:top_n] if mode == "top_n" else all_metrics
            if not subset:
                continue

            avg: dict[str, float] = {}
            for name in metric_fields:
                vals = [getattr(r.metrics, name) for r in subset]
                avg[name] = sum(vals) / len(vals)
            ref_avgs.append(avg)

        if not ref_avgs:
            return pd.DataFrame()

        # Average across refs
        final: dict[str, float] = {}
        for name in metric_fields:
            vals = [a[name] for a in ref_avgs if name in a]
            if vals:
                final[name] = sum(vals) / len(vals)

        return pd.DataFrame([final])

    @staticmethod
    def _save_per_ref_param_leaderboard(
        ref_output_dir: Path,
        table: pd.DataFrame,
        grid_keys: list[str],
        metric_fields: list[str],
        ranking_plan: RankingPlan | None,
        top_n: int,
    ) -> None:
        """Build and save a per-ref param_leaderboard.csv."""
        if table.empty:
            return

        valid_keys = [k for k in grid_keys if k in table.columns]
        if not valid_keys:
            return

        # Rank the table, take top N
        ranked = SweepRanker.rank_dataframe(table, ranking_plan)
        top = ranked.head(top_n).drop(columns=["rank"])

        # Average metrics by param combo
        available_metrics = [m for m in metric_fields if m in top.columns]
        if not available_metrics:
            return

        agg_cols = valid_keys + available_metrics
        grouped = top[agg_cols].groupby(valid_keys, dropna=False)[available_metrics]
        agg = grouped.mean().reset_index()

        # Rank averages
        lb = SweepRanker.rank_dataframe(agg, ranking_plan)
        ref_output_dir.mkdir(parents=True, exist_ok=True)
        lb.to_csv(ref_output_dir / "param_leaderboard.csv", index=False)

    @staticmethod
    def _save_analytics(
        output_dir: Path,
        cross_ref_summary: pd.DataFrame,
        per_ref_best: pd.DataFrame,
        param_leaderboard: pd.DataFrame,
        avg_top_n_refs: pd.DataFrame,
        avg_all_refs: pd.DataFrame,
        ref_instrument_ids: list[str],
        elapsed: float,
        cfg: SweepConfig,
    ) -> None:
        """Write cross-ref CSV files and run metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cross_ref_summary.to_csv(output_dir / "cross_ref_summary.csv", index=False)
        per_ref_best.to_csv(output_dir / "per_ref_best.csv", index=False)
        param_leaderboard.to_csv(output_dir / "param_leaderboard.csv", index=False)
        avg_top_n_refs.to_csv(output_dir / "avg_top_n_refs.csv", index=False)
        avg_all_refs.to_csv(output_dir / "avg_all_refs.csv", index=False)

        meta = {
            "ref_instrument_ids": ref_instrument_ids,
            "n_refs": len(ref_instrument_ids),
            "grid_name": cfg.grid_name,
            "grid": {k: list(v) for k, v in cfg.grid.items()},
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
