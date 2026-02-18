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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..configs.sweep_config import SweepConfig
from ..configs.utils import _resolve_config
from .sweep_runner import SweepRunResult, SweepRunner

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
        ranking_metric: str = "sharpe_ratio",
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
            Number of rows to keep in per_ref_best and param_leaderboard.
        ranking_metric : str
            Metric column used to rank scenarios (default ``"sharpe_ratio"``).
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

        # Resolve base output directory
        if output_dir is not None:
            base_output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = Path("runs") / cfg.grid_name / timestamp
        if save:
            base_output_dir.mkdir(parents=True, exist_ok=True)

        per_ref_results: dict[str, SweepRunResult] = {}
        all_tables: list[pd.DataFrame] = []

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

            ref_table = result.table.copy()
            ref_table.insert(0, "ref_instrument_id", ref_id)
            all_tables.append(ref_table)

        # Build cross-ref analytics
        cross_ref_summary = MultiRefSweepRunner._build_cross_ref_summary(all_tables)
        per_ref_best = MultiRefSweepRunner._build_per_ref_best(
            cross_ref_summary, ranking_metric, top_n,
        )
        grid_keys = list(cfg.grid.keys())
        param_leaderboard = MultiRefSweepRunner._build_param_leaderboard(
            cross_ref_summary, grid_keys, ranking_metric, top_n,
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
        ranking_metric: str,
        top_n: int,
    ) -> pd.DataFrame:
        """For each ref, select the row with the best ranking_metric."""
        if cross_ref_summary.empty or ranking_metric not in cross_ref_summary.columns:
            return pd.DataFrame()

        idx = cross_ref_summary.groupby("ref_instrument_id")[ranking_metric].idxmax()
        best = cross_ref_summary.loc[idx].sort_values(
            ranking_metric, ascending=False,
        )
        return best.head(top_n).reset_index(drop=True)

    @staticmethod
    def _build_param_leaderboard(
        cross_ref_summary: pd.DataFrame,
        grid_keys: list[str],
        ranking_metric: str,
        top_n: int,
    ) -> pd.DataFrame:
        """Aggregate ranking_metric by param combo across all refs."""
        if cross_ref_summary.empty or ranking_metric not in cross_ref_summary.columns:
            return pd.DataFrame()

        # Only use grid keys that are actually present in the summary
        valid_keys = [k for k in grid_keys if k in cross_ref_summary.columns]
        if not valid_keys:
            return pd.DataFrame()

        grouped = cross_ref_summary.groupby(valid_keys, dropna=False)[ranking_metric]
        agg = grouped.agg(
            mean="mean",
            median="median",
            n_refs="count",
            n_positive=lambda x: (x > 0).sum(),
        ).reset_index()
        agg = agg.sort_values("mean", ascending=False).head(top_n)
        return agg.reset_index(drop=True)

    @staticmethod
    def _save_analytics(
        output_dir: Path,
        cross_ref_summary: pd.DataFrame,
        per_ref_best: pd.DataFrame,
        param_leaderboard: pd.DataFrame,
        ref_instrument_ids: list[str],
        elapsed: float,
        cfg: SweepConfig,
    ) -> None:
        """Write cross-ref CSV files and run metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cross_ref_summary.to_csv(output_dir / "cross_ref_summary.csv", index=False)
        per_ref_best.to_csv(output_dir / "per_ref_best.csv", index=False)
        param_leaderboard.to_csv(output_dir / "param_leaderboard.csv", index=False)

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
