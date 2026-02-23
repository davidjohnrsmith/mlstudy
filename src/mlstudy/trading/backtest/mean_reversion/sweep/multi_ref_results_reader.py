"""Load multi-ref sweep results from disk.

Usage::

    from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_results_reader import (
        MultiRefResultsReader,
    )

    run = MultiRefResultsReader.load("runs/mr_grid_v1/20240101_120000")
    run.cross_ref_summary       # pd.DataFrame
    run.per_ref_best            # pd.DataFrame
    run.param_leaderboard       # pd.DataFrame
    run.ref_instrument_ids      # list[str]

    # Lazy-load a single ref's full results
    ref_data = run.load_ref("UST_2Y")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .sweep_results_reader import SweepResultsReader, SweepRunData


@dataclass
class MultiRefRunData:
    """Cross-ref summary plus on-demand access to individual ref results."""

    directory: Path
    meta: dict[str, Any]
    cross_ref_summary: pd.DataFrame
    per_ref_best: pd.DataFrame
    param_leaderboard: pd.DataFrame
    ref_instrument_ids: list[str]

    def load_ref(self, ref_id: str) -> SweepRunData:
        """Lazy-load a single ref instrument's full sweep results."""
        return MultiRefResultsReader.load_ref(self.directory, ref_id)


class MultiRefResultsReader:
    @staticmethod
    def load(run_dir: str | Path) -> MultiRefRunData:
        """Load cross-ref analytics from a multi-ref sweep run directory.

        Parameters
        ----------
        run_dir : str or Path
            Directory written by ``MultiRefSweepRunner.run(save=True)``.

        Returns
        -------
        MultiRefRunData
        """
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Metadata
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        ref_instrument_ids = meta.get("ref_instrument_ids", [])

        # Cross-ref CSVs
        def _read_csv(name: str) -> pd.DataFrame:
            p = run_dir / name
            return pd.read_csv(p) if p.exists() else pd.DataFrame()

        cross_ref_summary = _read_csv("cross_ref_summary.csv")
        per_ref_best = _read_csv("per_ref_best.csv")
        param_leaderboard = _read_csv("param_leaderboard.csv")

        return MultiRefRunData(
            directory=run_dir,
            meta=meta,
            cross_ref_summary=cross_ref_summary,
            per_ref_best=per_ref_best,
            param_leaderboard=param_leaderboard,
            ref_instrument_ids=ref_instrument_ids,
        )

    @staticmethod
    def load_partition(base_dir: str | Path, partition_index: int) -> MultiRefRunData:
        """Load a single partition's results.

        Parameters
        ----------
        base_dir : str or Path
            Directory containing ``partition_NNN/`` subdirectories.
        partition_index : int
            Zero-based partition index.

        Returns
        -------
        MultiRefRunData
        """
        return MultiRefResultsReader.load(
            Path(base_dir) / f"partition_{partition_index:03d}",
        )

    @staticmethod
    def load_combined(base_dir: str | Path) -> MultiRefRunData:
        """Load combined results produced by ``collect_partitions``.

        Parameters
        ----------
        base_dir : str or Path
            Directory containing the ``combined/`` subdirectory.

        Returns
        -------
        MultiRefRunData
        """
        return MultiRefResultsReader.load(Path(base_dir) / "combined")

    @staticmethod
    def load_ref(run_dir: str | Path, ref_instrument_id: str) -> SweepRunData:
        """Load a single ref instrument's full sweep results.

        Parameters
        ----------
        run_dir : str or Path
            Multi-ref run directory (parent of per-ref subdirectories).
        ref_instrument_id : str
            The ref instrument ID whose results to load.

        Returns
        -------
        SweepRunData
        """
        ref_dir = Path(run_dir) / ref_instrument_id
        return SweepResultsReader.load_sweep_run(ref_dir)
