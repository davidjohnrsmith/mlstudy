"""Load persisted portfolio sweep results from disk for post-hoc analysis.

Usage::

    from mlstudy.trading.backtest.portfolio.sweep.sweep_results_reader import (
        PortfolioSweepResultsReader,
    )

    run = PortfolioSweepResultsReader.load_sweep_run("runs/portfolio_v1/20240101_120000")

    # Summary table (all scenarios)
    run.summary                     # pd.DataFrame

    # All-metrics table
    run.all_metrics                 # pd.DataFrame or None

    # Run metadata
    run.meta                        # dict

    # Config snapshot
    run.config_snapshot             # dict

    # Full results for top-k scenarios
    run.full_scenarios              # list[PortfolioFullScenario]
    sc = run.full_scenarios[0]
    sc.spec                         # dict (name, tags, config, metrics)
    sc.results                      # PortfolioBacktestResults
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlstudy.trading.backtest.portfolio.single_backtest.results import PortfolioBacktestResults
from .sweep_persist import PORTFOLIO_ARRAY_FIELDS


@dataclass(frozen=True)
class PortfolioFullScenario:
    """A single portfolio scenario loaded from disk (arrays + spec)."""

    spec: dict[str, Any]
    results: PortfolioBacktestResults
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

    @property
    def equity(self) -> np.ndarray:
        return self.results.equity

    @property
    def pnl(self) -> np.ndarray:
        return self.results.pnl

    @property
    def positions(self) -> np.ndarray:
        return self.results.positions


@dataclass
class PortfolioSweepRunData:
    """Everything loaded from a persisted portfolio sweep run directory."""

    directory: Path
    meta: dict[str, Any]
    config_snapshot: dict[str, Any]
    summary: pd.DataFrame
    all_metrics: pd.DataFrame | None
    full_scenarios: list[PortfolioFullScenario]

    @property
    def grid_name(self) -> str:
        return self.meta.get("grid_name", "")

    @property
    def n_scenarios(self) -> int:
        return self.meta.get("n_scenarios", len(self.summary))

    @property
    def elapsed_seconds(self) -> float | None:
        return self.meta.get("elapsed_seconds")


class PortfolioSweepResultsReader:
    @staticmethod
    def load_full_scenario(scenario_dir: Path) -> PortfolioFullScenario:
        """Load one scenario sub-directory (spec.json + .npy arrays)."""
        with open(scenario_dir / "spec.json") as f:
            spec = json.load(f)

        arrays: dict[str, np.ndarray] = {}
        for name in PORTFOLIO_ARRAY_FIELDS:
            npy_path = scenario_dir / f"{name}.npy"
            if npy_path.exists():
                arrays[name] = np.load(npy_path)

        n_trades = len(arrays.get("tr_bar", []))

        results = PortfolioBacktestResults(
            positions=arrays["positions"],
            cash=arrays["cash"],
            equity=arrays["equity"],
            pnl=arrays["pnl"],
            gross_pnl=arrays.get("gross_pnl", arrays["pnl"]),
            codes=arrays["codes"],
            n_trades_bar=arrays["n_trades_bar"],
            cooldown=arrays["cooldown"],
            hedge_positions=arrays.get("hedge_positions", np.empty((len(arrays["equity"]), 0))),
            tr_bar=arrays.get("tr_bar", np.array([], dtype=np.int64)),
            tr_instrument=arrays.get("tr_instrument", np.array([], dtype=np.int64)),
            tr_side=arrays.get("tr_side", np.array([], dtype=np.int32)),
            tr_qty_req=arrays.get("tr_qty_req", np.array([], dtype=np.float64)),
            tr_qty_fill=arrays.get("tr_qty_fill", np.array([], dtype=np.float64)),
            tr_dv01_req=arrays.get("tr_dv01_req", np.array([], dtype=np.float64)),
            tr_dv01_fill=arrays.get("tr_dv01_fill", np.array([], dtype=np.float64)),
            tr_alpha=arrays.get("tr_alpha", np.array([], dtype=np.float64)),
            tr_fair_type=arrays.get("tr_fair_type", np.array([], dtype=np.int32)),
            tr_vwap=arrays.get("tr_vwap", np.array([], dtype=np.float64)),
            tr_mid=arrays.get("tr_mid", np.array([], dtype=np.float64)),
            tr_cost=arrays.get("tr_cost", np.array([], dtype=np.float64)),
            tr_code=arrays.get("tr_code", np.array([], dtype=np.int32)),
            tr_hedge_sizes=arrays.get("tr_hedge_sizes", np.empty((n_trades, 0))),
            tr_hedge_vwaps=arrays.get("tr_hedge_vwaps", np.empty((n_trades, 0))),
            tr_hedge_fills=arrays.get("tr_hedge_fills", np.empty((n_trades, 0))),
            tr_hedge_cost=arrays.get("tr_hedge_cost", np.array([], dtype=np.float64)),
            n_trades=n_trades,
        )

        # Load instrument/hedge ID mapping if available
        id_map_path = scenario_dir / "id_map.json"
        if id_map_path.exists():
            with open(id_map_path) as f:
                id_map = json.load(f)
            inst_map = id_map.get("instrument_ids", {})
            if inst_map:
                results.instrument_ids = [
                    inst_map[str(i)] for i in range(len(inst_map))
                ]
            hedge_map = id_map.get("hedge_ids", {})
            if hedge_map:
                results.hedge_ids = [
                    hedge_map[str(i)] for i in range(len(hedge_map))
                ]

        return PortfolioFullScenario(spec=spec, results=results, directory=scenario_dir)

    @staticmethod
    def load_full_dir(full_dir: Path) -> list[PortfolioFullScenario]:
        """Load all scenario sub-directories under a full-results directory."""
        if not full_dir.is_dir():
            return []

        scenario_dirs = sorted(
            d for d in full_dir.iterdir() if d.is_dir() and (d / "spec.json").exists()
        )

        return [PortfolioSweepResultsReader.load_full_scenario(d) for d in scenario_dirs]

    @staticmethod
    def load_sweep_run(run_dir: str | Path) -> PortfolioSweepRunData:
        """Load a persisted portfolio sweep run from *run_dir*.

        Parameters
        ----------
        run_dir : str or Path
            Directory written by the portfolio sweep persister.

        Returns
        -------
        PortfolioSweepRunData
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

        # --- Full results ---
        full_scenarios: list[PortfolioFullScenario] = []
        for label in ("top_full", "full"):
            full_dir = run_dir / label
            if full_dir.is_dir():
                full_scenarios = PortfolioSweepResultsReader.load_full_dir(full_dir)
                break

        return PortfolioSweepRunData(
            directory=run_dir,
            meta=meta,
            config_snapshot=config_snapshot,
            summary=summary,
            all_metrics=all_metrics,
            full_scenarios=full_scenarios,
        )
