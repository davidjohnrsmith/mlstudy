"""Load persisted sweep results from disk for post-hoc analysis.

Usage::

    from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import (
        SweepResultsReader,
    )

    run = SweepResultsReader.load_sweep_run("runs/mr_grid_v1/20240101_120000")

    # Summary table (all scenarios)
    run.summary                     # pd.DataFrame

    # All-metrics table (scenario_idx, name, tags, metrics)
    run.all_metrics                 # pd.DataFrame or None

    # Run metadata
    run.meta                        # dict (grid_name, n_scenarios, elapsed, ...)

    # Config snapshot
    run.config_snapshot             # dict (parsed YAML)

    # Full results for top-k scenarios
    run.full_scenarios              # list[FullScenario]
    sc = run.full_scenarios[0]
    sc.spec                         # dict (name, tags, config, metrics)
    sc.results                      # MRBacktestResults
    sc.equity                       # shorthand for sc.results.equity
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults, ARRAY_FIELDS




@dataclass(frozen=True)
class FullScenario:
    """A single scenario loaded from disk (arrays + spec)."""

    spec: dict[str, Any]
    results: MRBacktestResults
    directory: Path

    @property
    def name(self) -> str:
        return self.spec.get("name", "")

    @property
    def tags(self) -> dict[str, Any]:
        return self.spec.get("tags", {})

    @property
    def config(self) -> dict[str, Any]:
        return self.spec.get("config", {})

    @property
    def metrics(self) -> dict[str, Any]:
        return self.spec.get("metrics", {})

    @property
    def scenario_idx(self) -> int:
        return self.spec.get("scenario_idx", -1)

    # Convenience accessors for commonly used arrays
    @property
    def equity(self) -> np.ndarray:
        return self.results.equity

    @property
    def pnl(self) -> np.ndarray:
        return self.results.pnl

    @property
    def positions(self) -> np.ndarray:
        return self.results.positions

    @property
    def codes(self) -> np.ndarray:
        return self.results.codes


@dataclass
class SweepRunData:
    """Everything loaded from a persisted sweep run directory."""

    directory: Path
    meta: dict[str, Any]
    config_snapshot: dict[str, Any]
    summary: pd.DataFrame
    all_metrics: pd.DataFrame | None
    full_scenarios: list[FullScenario]

    @property
    def grid_name(self) -> str:
        return self.meta.get("grid_name", "")

    @property
    def n_scenarios(self) -> int:
        return self.meta.get("n_scenarios", len(self.summary))

    @property
    def elapsed_seconds(self) -> float | None:
        return self.meta.get("elapsed_seconds")


class SweepResultsReader:
    @staticmethod
    def load_full_scenario(scenario_dir: Path) -> FullScenario:
        """Load one scenario sub-directory (spec.json + .npy arrays)."""
        with open(scenario_dir / "spec.json") as f:
            spec = json.load(f)

        arrays = {}
        for name in ARRAY_FIELDS:
            npy_path = scenario_dir / f"{name}.npy"
            if npy_path.exists():
                arrays[name] = np.load(npy_path)

        # n_trades: infer from trade arrays (they are already trimmed on save)
        n_trades = len(arrays.get("tr_bar", []))

        # gross_pnl may be absent in old saved runs; fall back to pnl
        gross_pnl = arrays.get("gross_pnl", arrays["pnl"])

        # Optional market data arrays
        mid_px = None
        mid_px_path = scenario_dir / "mid_px.npy"
        if mid_px_path.exists():
            mid_px = np.load(mid_px_path)

        package_yield_bps = None
        pkg_yield_path = scenario_dir / "package_yield_bps.npy"
        if pkg_yield_path.exists():
            package_yield_bps = np.load(pkg_yield_path)

        zscore = None
        zscore_path = scenario_dir / "zscore.npy"
        if zscore_path.exists():
            zscore = np.load(zscore_path)

        results = MRBacktestResults(
            positions=arrays["positions"],
            cash=arrays["cash"],
            equity=arrays["equity"],
            pnl=arrays["pnl"],
            gross_pnl=gross_pnl,
            codes=arrays["codes"],
            state=arrays["state"],
            holding=arrays["holding"],
            tr_bar=arrays.get("tr_bar", np.array([], dtype=np.int64)),
            tr_type=arrays.get("tr_type", np.array([], dtype=np.int32)),
            tr_side=arrays.get("tr_side", np.array([], dtype=np.int32)),
            tr_sizes=arrays.get("tr_sizes", np.empty((0, 0), dtype=np.float64)),
            tr_risks=arrays.get("tr_risks", np.empty((0, 0), dtype=np.float64)),
            tr_vwaps=arrays.get("tr_vwaps", np.empty((0, 0), dtype=np.float64)),
            tr_mids=arrays.get("tr_mids", np.empty((0, 0), dtype=np.float64)),
            tr_cost=arrays.get("tr_cost", np.array([], dtype=np.float64)),
            tr_code=arrays.get("tr_code", np.array([], dtype=np.int32)),
            tr_pkg_yield=arrays.get("tr_pkg_yield", np.array([], dtype=np.float64)),
            n_trades=n_trades,
            mid_px=mid_px,
            package_yield_bps=package_yield_bps,
            zscore=zscore,
        )

        return FullScenario(spec=spec, results=results, directory=scenario_dir)

    @staticmethod
    def load_full_dir(full_dir: Path) -> list[FullScenario]:
        """Load all scenario sub-directories under a full-results directory."""
        if not full_dir.is_dir():
            return []

        scenario_dirs = sorted(
            d for d in full_dir.iterdir() if d.is_dir() and (d / "spec.json").exists()
        )

        return [SweepResultsReader.load_full_scenario(d) for d in scenario_dirs]

    @staticmethod
    def load_sweep_run(run_dir: str | Path) -> SweepRunData:
        """Load a persisted sweep run from *run_dir*.

        Parameters
        ----------
        run_dir : str or Path
            Directory written by ``run_sweep_from_config(save=True)``.
            Expected contents: ``run_meta.json``, ``config_snapshot.yaml``,
            ``summary.csv``, and optionally ``all_metrics.csv`` and
            ``top_full/`` or ``full/`` sub-directories.

        Returns
        -------
        SweepRunData
        """
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # --- Metadata ---
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        # --- Config snapshot ---
        config_path = run_dir / "config_snapshot.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                config_snapshot = yaml.safe_load(f) or {}
        else:
            config_snapshot = {}

        # --- Summary table ---
        summary_path = run_dir / "summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
        else:
            summary = pd.DataFrame()

        # --- All-metrics table ---
        metrics_path = run_dir / "all_metrics.csv"
        all_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None

        # --- Full results (check top_full/ first, then full/) ---
        full_scenarios: list[FullScenario] = []
        for label in ("top_full", "full"):
            full_dir = run_dir / label
            if full_dir.is_dir():
                full_scenarios = SweepResultsReader.load_full_dir(full_dir)
                break

        return SweepRunData(
            directory=run_dir,
            meta=meta,
            config_snapshot=config_snapshot,
            summary=summary,
            all_metrics=all_metrics,
            full_scenarios=full_scenarios,
        )
