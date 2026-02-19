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
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan, SweepRanker
from .sweep_runner import SweepRunResult, SweepRunner
from ...metrics.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


def partition_instruments(
    ref_instrument_ids: list[str],
    num_partitions: int,
    partition_index: int,
) -> list[str]:
    """Return the contiguous subset of *ref_instrument_ids* for *partition_index*.

    Uses ``divmod(N, M)`` so the first ``remainder`` partitions get one extra
    element.  Order is preserved.  Returns an empty list when
    *partition_index* >= len(ref_instrument_ids).
    """
    n = len(ref_instrument_ids)
    if partition_index >= n:
        return []
    base_size, remainder = divmod(n, num_partitions)
    # First `remainder` partitions get base_size+1 items
    if partition_index < remainder:
        start = partition_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (partition_index - remainder) * base_size
        end = start + base_size
    return ref_instrument_ids[start:end]


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
        _base_output_dir_override: Path | None = None,
        _extra_meta: dict[str, Any] | None = None,
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
        if _base_output_dir_override is not None:
            base_output_dir = _base_output_dir_override
        elif output_dir is not None:
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
                extra_meta=_extra_meta,
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
        extra_meta: dict[str, Any] | None = None,
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
        if extra_meta:
            meta.update(extra_meta)
        with open(output_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Distributed partition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def run_partition(
        config: str | Path | SweepConfig,
        ref_instrument_ids: list[str],
        *,
        num_partitions: int,
        partition_index: int,
        data_path: str | Path | None = None,
        instrument_ids: list[str] | None = None,
        output_dir: str | Path | None = None,
        save: bool = True,
        top_n: int = 10,
        config_map_path: str | Path | None = None,
        **sweep_kwargs: Any,
    ) -> MultiRefSweepResult:
        """Run a single partition of a distributed multi-ref sweep.

        Parameters
        ----------
        config : str, Path, or SweepConfig
            Sweep configuration.
        ref_instrument_ids : list[str]
            The **full** sorted list of reference instruments (same on every
            machine).  ``partition_instruments`` selects this machine's subset.
        num_partitions : int
            Total number of partitions (machines).
        partition_index : int
            Zero-based index of this partition.
        output_dir, data_path, instrument_ids, save, top_n, config_map_path
            Forwarded to ``run()``.
        **sweep_kwargs
            Forwarded to ``SweepRunner.run_sweep_from_config``.

        Returns
        -------
        MultiRefSweepResult
            Result for this partition's subset of refs.  Empty result if the
            subset is empty (more machines than instruments).
        """
        subset = partition_instruments(ref_instrument_ids, num_partitions, partition_index)

        if not subset:
            return MultiRefSweepResult(
                output_dir=None,
                cross_ref_summary=pd.DataFrame(),
                per_ref_best=pd.DataFrame(),
                param_leaderboard=pd.DataFrame(),
                per_ref_results={},
            )

        cfg = _resolve_config(config, config_map_path)
        base = Path(output_dir) if output_dir is not None else Path("runs")
        partition_dir = base / cfg.grid_name / f"partition_{partition_index:03d}"

        extra_meta = {
            "partition_index": partition_index,
            "num_partitions": num_partitions,
            "all_ref_instrument_ids": ref_instrument_ids,
        }

        return MultiRefSweepRunner.run(
            cfg,
            subset,
            data_path=data_path,
            instrument_ids=instrument_ids,
            save=save,
            top_n=top_n,
            _base_output_dir_override=partition_dir,
            _extra_meta=extra_meta,
            **sweep_kwargs,
        )

    @staticmethod
    def _build_cross_ref_averages_from_summary(
        cross_ref_summary: pd.DataFrame,
        metric_fields: list[str],
        ranking_plan: RankingPlan | None,
        top_n: int,
        mode: str,
    ) -> pd.DataFrame:
        """Compute one-row cross-ref averages from a summary DataFrame.

        This is the CSV-based counterpart of ``_build_cross_ref_averages``,
        used by ``collect_partitions`` when in-memory ``SweepRunResult``
        objects are not available.

        Parameters
        ----------
        cross_ref_summary : pd.DataFrame
            Must contain a ``ref_instrument_id`` column.
        metric_fields : list[str]
            Metric column names to average.
        ranking_plan : RankingPlan or None
            Used to rank within each ref when *mode* is ``"top_n"``.
        top_n : int
            How many top scenarios per ref (only used when mode="top_n").
        mode : str
            ``"top_n"`` or ``"all"``.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with averaged metrics.
        """
        if cross_ref_summary.empty:
            return pd.DataFrame()

        available_metrics = [m for m in metric_fields if m in cross_ref_summary.columns]
        if not available_metrics:
            return pd.DataFrame()

        ref_avgs: list[dict[str, float]] = []
        for _ref_id, group in cross_ref_summary.groupby("ref_instrument_id"):
            ref_data = group.drop(columns=["ref_instrument_id"], errors="ignore")
            if mode == "top_n":
                ranked = SweepRanker.rank_dataframe(ref_data, ranking_plan)
                subset = ranked.head(top_n)
            else:
                subset = ref_data

            avg = {m: subset[m].mean() for m in available_metrics if m in subset.columns}
            if avg:
                ref_avgs.append(avg)

        if not ref_avgs:
            return pd.DataFrame()

        final: dict[str, float] = {}
        for m in available_metrics:
            vals = [a[m] for a in ref_avgs if m in a]
            if vals:
                final[m] = sum(vals) / len(vals)

        return pd.DataFrame([final])

    @staticmethod
    def collect_partitions(
        base_output_dir: str | Path,
        config: str | Path | SweepConfig,
        *,
        num_partitions: int,
        ref_instrument_ids: list[str] | None = None,
        top_n: int = 10,
        config_map_path: str | Path | None = None,
    ) -> MultiRefSweepResult:
        """Read all partition results and produce combined cross-ref analytics.

        Parameters
        ----------
        base_output_dir : str or Path
            Directory that contains ``partition_000/``, ``partition_001/``, etc.
        config : str, Path, or SweepConfig
            Sweep configuration (needed for grid keys / ranking plan).
        num_partitions : int
            Expected number of partitions.
        ref_instrument_ids : list[str], optional
            Full ordered list of ref instruments.  If *None*, reconstructed
            from partition ``run_meta.json`` files.
        top_n : int
            Number of top scenarios for averages / leaderboard.
        config_map_path : str or Path, optional
            Path to config map YAML.

        Returns
        -------
        MultiRefSweepResult

        Raises
        ------
        FileNotFoundError
            If any expected partition directory is missing.
        """
        import time as _time

        t0 = _time.perf_counter()
        base_output_dir = Path(base_output_dir)
        cfg = _resolve_config(config, config_map_path)
        grid_keys = list(cfg.grid.keys())
        metric_fields = [f.name for f in fields(BacktestMetrics)]
        ranking_plan = cfg.ranking_plan

        all_summaries: list[pd.DataFrame] = []
        all_ref_ids: list[str] = []

        for i in range(num_partitions):
            part_dir = base_output_dir / f"partition_{i:03d}"
            if not part_dir.is_dir():
                raise FileNotFoundError(f"Partition directory not found: {part_dir}")

            # Read partition meta for ref ids
            meta_path = part_dir / "run_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    part_meta = json.load(f)
                part_refs = part_meta.get("ref_instrument_ids", [])
            else:
                part_refs = []
            all_ref_ids.extend(part_refs)

            # Read partition cross-ref summary
            csv_path = part_dir / "cross_ref_summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_summaries.append(df)

        # Use caller-supplied ref ids if provided
        if ref_instrument_ids is not None:
            all_ref_ids = list(ref_instrument_ids)

        # Combine
        if all_summaries:
            cross_ref_summary = pd.concat(all_summaries, ignore_index=True)
        else:
            cross_ref_summary = pd.DataFrame()

        per_ref_best = MultiRefSweepRunner._build_per_ref_best(
            cross_ref_summary, ranking_plan,
        )
        param_leaderboard = MultiRefSweepRunner._build_param_leaderboard(
            cross_ref_summary, grid_keys, metric_fields, ranking_plan, top_n,
        )
        avg_top_n_refs = MultiRefSweepRunner._build_cross_ref_averages_from_summary(
            cross_ref_summary, metric_fields, ranking_plan, top_n, mode="top_n",
        )
        avg_all_refs = MultiRefSweepRunner._build_cross_ref_averages_from_summary(
            cross_ref_summary, metric_fields, ranking_plan, top_n, mode="all",
        )

        elapsed = _time.perf_counter() - t0

        combined_dir = base_output_dir / "combined"
        MultiRefSweepRunner._save_analytics(
            combined_dir,
            cross_ref_summary,
            per_ref_best,
            param_leaderboard,
            avg_top_n_refs,
            avg_all_refs,
            all_ref_ids,
            elapsed,
            cfg,
            extra_meta={"source": "collect_partitions"},
        )

        return MultiRefSweepResult(
            output_dir=combined_dir,
            cross_ref_summary=cross_ref_summary,
            per_ref_best=per_ref_best,
            param_leaderboard=param_leaderboard,
            per_ref_results={},
        )
