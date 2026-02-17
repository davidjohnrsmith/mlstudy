"""Tests for sweep_results_reader — loading persisted sweep runs from disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import (
    FullScenario,
    SweepRunData,
    load_sweep_run,
    _load_full_scenario,
)


# ---------------------------------------------------------------------------
# Helpers — write synthetic run directories
# ---------------------------------------------------------------------------

_ARRAY_FIELDS_1D = [
    "cash", "equity", "pnl", "codes", "state", "holding",
    "tr_bar", "tr_type", "tr_side", "tr_cost", "tr_code", "tr_pkg_yield",
]
_ARRAY_FIELDS_2D = [
    "positions", "tr_sizes", "tr_risks", "tr_vwaps", "tr_mids",
]


def _write_run_meta(run_dir: Path, **overrides) -> None:
    meta = {
        "grid_name": "test_grid",
        "n_scenarios": 4,
        "base_config": {"ref_leg_idx": 1},
        "grid": {"entry_z_threshold": [1.5, 2.0]},
        "sweep_kwargs": {"backend": "serial"},
        "timestamp": "2024-01-01T00:00:00+00:00",
        "elapsed_seconds": 3.14,
    }
    meta.update(overrides)
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f)


def _write_config_snapshot(run_dir: Path) -> None:
    snapshot = {
        "grid_name": "test_grid",
        "base_config": {"ref_leg_idx": 1},
        "grid": {"entry_z_threshold": [1.5, 2.0]},
        "sweep": {"backend": "serial"},
    }
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snapshot, f)


def _write_summary_csv(run_dir: Path, n_rows: int = 4) -> pd.DataFrame:
    df = pd.DataFrame({
        "name": [f"scenario_{i}" for i in range(n_rows)],
        "total_pnl": np.random.default_rng(42).normal(0, 10, n_rows),
        "sharpe_ratio": np.random.default_rng(43).normal(0.5, 0.3, n_rows),
    })
    df.to_csv(run_dir / "summary.csv", index=False)
    return df


def _write_all_metrics_csv(run_dir: Path, n_rows: int = 4) -> pd.DataFrame:
    df = pd.DataFrame({
        "scenario_idx": list(range(n_rows)),
        "name": [f"scenario_{i}" for i in range(n_rows)],
        "entry_z_threshold": [1.5, 2.0, 1.5, 2.0],
        "total_pnl": [10.0, 20.0, -5.0, 15.0],
        "sharpe_ratio": [0.5, 1.0, -0.2, 0.8],
    })
    df.to_csv(run_dir / "all_metrics.csv", index=False)
    return df


def _write_full_scenario(
    scenario_dir: Path,
    T: int = 10,
    N: int = 3,
    n_trades: int = 2,
    scenario_idx: int = 0,
) -> None:
    """Write a synthetic scenario directory with spec.json + .npy files."""
    scenario_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "name": f"scenario_{scenario_idx}",
        "tags": {"entry_z_threshold": 2.0},
        "config": {"ref_leg_idx": 1, "entry_z_threshold": 2.0},
        "metrics": {"total_pnl": 15.5, "sharpe_ratio": 0.8},
        "scenario_idx": scenario_idx,
    }
    with open(scenario_dir / "spec.json", "w") as f:
        json.dump(spec, f)

    rng = np.random.default_rng(scenario_idx)

    # Per-bar arrays
    np.save(scenario_dir / "positions.npy", rng.normal(0, 1, (T, N)))
    np.save(scenario_dir / "cash.npy", rng.normal(0, 1, T))
    np.save(scenario_dir / "equity.npy", np.cumsum(rng.normal(0, 1, T)))
    np.save(scenario_dir / "pnl.npy", rng.normal(0, 1, T))
    np.save(scenario_dir / "codes.npy", np.zeros(T, dtype=np.int32))
    np.save(scenario_dir / "state.npy", np.zeros(T, dtype=np.int32))
    np.save(scenario_dir / "holding.npy", np.zeros(T, dtype=np.int32))

    # Per-trade arrays (already trimmed to n_trades)
    np.save(scenario_dir / "tr_bar.npy", np.arange(n_trades, dtype=np.int64))
    np.save(scenario_dir / "tr_type.npy", np.zeros(n_trades, dtype=np.int32))
    np.save(scenario_dir / "tr_side.npy", np.ones(n_trades, dtype=np.int32))
    np.save(scenario_dir / "tr_sizes.npy", rng.normal(0, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_risks.npy", rng.normal(0, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_vwaps.npy", rng.normal(100, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_mids.npy", rng.normal(100, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_cost.npy", rng.uniform(0, 1, n_trades))
    np.save(scenario_dir / "tr_code.npy", np.full(n_trades, 3, dtype=np.int32))
    np.save(scenario_dir / "tr_pkg_yield.npy", rng.normal(100, 1, n_trades))


def _write_minimal_run(run_dir: Path, with_metrics: bool = True, with_full: bool = False):
    """Write a complete minimal run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_meta(run_dir)
    _write_config_snapshot(run_dir)
    _write_summary_csv(run_dir)
    if with_metrics:
        _write_all_metrics_csv(run_dir)
    if with_full:
        for i in range(3):
            _write_full_scenario(run_dir / "top_full" / f"scenario_{i:03d}", scenario_idx=i)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadSweepRunBasic:
    def test_loads_meta(self, tmp_path):
        _write_minimal_run(tmp_path)
        run = load_sweep_run(tmp_path)
        assert run.meta["grid_name"] == "test_grid"
        assert run.meta["n_scenarios"] == 4
        assert run.meta["elapsed_seconds"] == 3.14

    def test_properties(self, tmp_path):
        _write_minimal_run(tmp_path)
        run = load_sweep_run(tmp_path)
        assert run.grid_name == "test_grid"
        assert run.n_scenarios == 4
        assert run.elapsed_seconds == 3.14
        assert run.directory == tmp_path

    def test_loads_config_snapshot(self, tmp_path):
        _write_minimal_run(tmp_path)
        run = load_sweep_run(tmp_path)
        assert run.config_snapshot["grid_name"] == "test_grid"
        assert "base_config" in run.config_snapshot

    def test_loads_summary(self, tmp_path):
        _write_minimal_run(tmp_path)
        run = load_sweep_run(tmp_path)
        assert isinstance(run.summary, pd.DataFrame)
        assert len(run.summary) == 4
        assert "total_pnl" in run.summary.columns

    def test_loads_all_metrics(self, tmp_path):
        _write_minimal_run(tmp_path, with_metrics=True)
        run = load_sweep_run(tmp_path)
        assert run.all_metrics is not None
        assert len(run.all_metrics) == 4
        assert "scenario_idx" in run.all_metrics.columns

    def test_no_metrics_file(self, tmp_path):
        _write_minimal_run(tmp_path, with_metrics=False)
        run = load_sweep_run(tmp_path)
        assert run.all_metrics is None

    def test_no_full_scenarios(self, tmp_path):
        _write_minimal_run(tmp_path, with_full=False)
        run = load_sweep_run(tmp_path)
        assert run.full_scenarios == []

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sweep_run("/nonexistent/path")


class TestLoadSweepRunWithFull:
    def test_loads_full_scenarios(self, tmp_path):
        _write_minimal_run(tmp_path, with_full=True)
        run = load_sweep_run(tmp_path)
        assert len(run.full_scenarios) == 3

    def test_scenario_spec(self, tmp_path):
        _write_minimal_run(tmp_path, with_full=True)
        run = load_sweep_run(tmp_path)
        sc = run.full_scenarios[0]
        assert sc.name == "scenario_0"
        assert sc.scenario_idx == 0
        assert sc.tags == {"entry_z_threshold": 2.0}
        assert sc.config["ref_leg_idx"] == 1
        assert sc.metrics["total_pnl"] == 15.5

    def test_scenario_arrays(self, tmp_path):
        T, N, n_trades = 10, 3, 2
        _write_minimal_run(tmp_path, with_full=True)
        run = load_sweep_run(tmp_path)
        sc = run.full_scenarios[0]

        assert sc.results.positions.shape == (T, N)
        assert sc.results.equity.shape == (T,)
        assert sc.results.pnl.shape == (T,)
        assert sc.results.n_trades == n_trades
        assert sc.results.tr_bar.shape == (n_trades,)
        assert sc.results.tr_sizes.shape == (n_trades, N)

    def test_convenience_accessors(self, tmp_path):
        _write_minimal_run(tmp_path, with_full=True)
        sc = load_sweep_run(tmp_path).full_scenarios[0]
        # Shorthand accessors should return same arrays
        np.testing.assert_array_equal(sc.equity, sc.results.equity)
        np.testing.assert_array_equal(sc.pnl, sc.results.pnl)
        np.testing.assert_array_equal(sc.positions, sc.results.positions)
        np.testing.assert_array_equal(sc.codes, sc.results.codes)

    def test_scenarios_sorted_by_dir_name(self, tmp_path):
        _write_minimal_run(tmp_path, with_full=True)
        run = load_sweep_run(tmp_path)
        idxs = [sc.scenario_idx for sc in run.full_scenarios]
        assert idxs == [0, 1, 2]

    def test_full_dir_fallback(self, tmp_path):
        """If top_full/ doesn't exist but full/ does, load from full/."""
        _write_minimal_run(tmp_path, with_full=False)
        for i in range(2):
            _write_full_scenario(tmp_path / "full" / f"scenario_{i:03d}", scenario_idx=i)
        run = load_sweep_run(tmp_path)
        assert len(run.full_scenarios) == 2


class TestLoadFullScenario:
    def test_single_scenario(self, tmp_path):
        sd = tmp_path / "scenario_000"
        _write_full_scenario(sd, T=20, N=2, n_trades=3, scenario_idx=5)
        sc = _load_full_scenario(sd)

        assert sc.name == "scenario_5"
        assert sc.scenario_idx == 5
        assert sc.results.positions.shape == (20, 2)
        assert sc.results.n_trades == 3
        assert sc.results.tr_sizes.shape == (3, 2)
        assert sc.directory == sd


class TestMissingFiles:
    """Graceful handling of missing optional files."""

    def test_missing_meta(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        _write_summary_csv(tmp_path)
        _write_config_snapshot(tmp_path)
        run = load_sweep_run(tmp_path)
        assert run.meta == {}
        assert run.grid_name == ""
        assert run.elapsed_seconds is None

    def test_missing_config_snapshot(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        _write_run_meta(tmp_path)
        _write_summary_csv(tmp_path)
        run = load_sweep_run(tmp_path)
        assert run.config_snapshot == {}

    def test_missing_summary(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        _write_run_meta(tmp_path)
        run = load_sweep_run(tmp_path)
        assert run.summary.empty
