"""Tests for MultiRefSweepRunner and MultiRefResultsReader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from mlstudy.trading.backtest.mean_reversion.configs.backtest_config import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.configs.sweep_config import SweepConfig
from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_runner import (
    MultiRefSweepResult,
    MultiRefSweepRunner,
)
from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_results_reader import (
    MultiRefResultsReader,
    MultiRefRunData,
)
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_runner import SweepRunResult


# =========================================================================
# Helpers
# =========================================================================

def _make_sweep_config(grid: dict | None = None) -> SweepConfig:
    """Create a minimal SweepConfig for testing."""
    return SweepConfig(
        grid_name="test_multi",
        base_config=MRBacktestConfig(
            ref_leg_idx=1,
            target_notional_ref=100.0,
            entry_z_threshold=2.0,
        ),
        grid=grid or {"entry_z_threshold": [1.5, 2.0, 3.0]},
        sweep_kwargs={"backend": "serial", "mode": "metrics_only"},
        ranking_plan=None,
    )


def _make_fake_table(ref_id: str, grid: dict | None = None) -> pd.DataFrame:
    """Build a fake summary table as SweepRunner would produce."""
    grid = grid or {"entry_z_threshold": [1.5, 2.0, 3.0]}
    rng = np.random.default_rng(hash(ref_id) % 2**32)
    rows = []
    for i, ez in enumerate(grid["entry_z_threshold"]):
        rows.append({
            "name": f"test_multi_{i:03d}",
            "entry_z_threshold": ez,
            "total_pnl": rng.normal(10, 5),
            "sharpe_ratio": rng.normal(1.0, 0.5),
            "max_drawdown": rng.uniform(0, 10),
            "n_trades": rng.integers(5, 50),
        })
    return pd.DataFrame(rows)


def _make_fake_run_result(
    ref_id: str,
    cfg: SweepConfig,
    output_dir: Path | None = None,
) -> SweepRunResult:
    """Create a fake SweepRunResult."""
    table = _make_fake_table(ref_id, dict(cfg.grid))
    return SweepRunResult(
        config=cfg,
        raw=[],  # not needed for multi-ref analytics
        table=table,
        output_dir=output_dir,
    )


REF_IDS = ["UST_2Y", "UST_5Y", "UST_10Y"]


@pytest.fixture
def sweep_config():
    return _make_sweep_config()


@pytest.fixture
def patched_sweep_runner(tmp_path, sweep_config):
    """Patch SweepRunner.run_sweep_from_config to return fake results."""
    def fake_run(config, *, ref_instrument_id=None, output_dir=None, **kw):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # Write minimal per-ref files to verify directory structure
            (Path(output_dir) / "summary.csv").write_text("name,total_pnl\nfoo,1.0\n")
            (Path(output_dir) / "run_meta.json").write_text(json.dumps({"grid_name": "test"}))
        return _make_fake_run_result(ref_instrument_id, config, output_dir)

    with patch.object(
        __import__(
            "mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_runner",
            fromlist=["SweepRunner"],
        ).SweepRunner,
        "run_sweep_from_config",
        side_effect=fake_run,
    ) as mock:
        yield mock


# =========================================================================
# MultiRefSweepRunner.run — basic behavior
# =========================================================================


class TestMultiRefSweepRunnerRun:
    def test_returns_multi_ref_result(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        assert isinstance(result, MultiRefSweepResult)

    def test_per_ref_results_keys(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        assert set(result.per_ref_results.keys()) == set(REF_IDS)
        for ref_id in REF_IDS:
            assert isinstance(result.per_ref_results[ref_id], SweepRunResult)

    def test_calls_sweep_runner_for_each_ref(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        assert patched_sweep_runner.call_count == len(REF_IDS)

    def test_no_save(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", save=False,
        )
        assert result.output_dir is None


# =========================================================================
# Cross-ref summary
# =========================================================================


class TestCrossRefSummary:
    def test_shape(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        n_scenarios = len(sweep_config.grid["entry_z_threshold"])
        expected_rows = len(REF_IDS) * n_scenarios
        assert len(result.cross_ref_summary) == expected_rows

    def test_has_ref_instrument_id_column(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        assert "ref_instrument_id" in result.cross_ref_summary.columns
        assert set(result.cross_ref_summary["ref_instrument_id"]) == set(REF_IDS)

    def test_all_refs_and_scenarios_present(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        for ref_id in REF_IDS:
            ref_rows = result.cross_ref_summary[
                result.cross_ref_summary["ref_instrument_id"] == ref_id
            ]
            assert len(ref_rows) == len(sweep_config.grid["entry_z_threshold"])


# =========================================================================
# Per-ref best
# =========================================================================


class TestPerRefBest:
    def test_one_row_per_ref(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        assert len(result.per_ref_best) == len(REF_IDS)
        assert set(result.per_ref_best["ref_instrument_id"]) == set(REF_IDS)

    def test_sorted_descending(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        vals = result.per_ref_best["sharpe_ratio"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_top_n_limits(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=2,
        )
        assert len(result.per_ref_best) <= 2

    def test_correct_best_per_ref(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        # Verify each row is actually the best for its ref
        for _, row in result.per_ref_best.iterrows():
            ref_id = row["ref_instrument_id"]
            ref_rows = result.cross_ref_summary[
                result.cross_ref_summary["ref_instrument_id"] == ref_id
            ]
            assert row["sharpe_ratio"] == ref_rows["sharpe_ratio"].max()


# =========================================================================
# Param leaderboard
# =========================================================================


class TestParamLeaderboard:
    def test_grouped_by_grid_params(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        lb = result.param_leaderboard
        assert "entry_z_threshold" in lb.columns
        assert "mean" in lb.columns
        assert "median" in lb.columns
        assert "n_refs" in lb.columns
        assert "n_positive" in lb.columns

    def test_n_refs_matches(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        # Each param combo should have been seen across all refs
        for _, row in result.param_leaderboard.iterrows():
            assert row["n_refs"] == len(REF_IDS)

    def test_sorted_by_mean_descending(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        vals = result.param_leaderboard["mean"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_top_n_limits(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=2,
        )
        assert len(result.param_leaderboard) <= 2


# =========================================================================
# Output directory structure
# =========================================================================


class TestOutputDirectoryStructure:
    def test_cross_ref_files_saved(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        assert (out / "cross_ref_summary.csv").exists()
        assert (out / "per_ref_best.csv").exists()
        assert (out / "param_leaderboard.csv").exists()
        assert (out / "run_meta.json").exists()

    def test_per_ref_subdirectories(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        for ref_id in REF_IDS:
            assert (out / ref_id).is_dir()
            assert (out / ref_id / "summary.csv").exists()

    def test_run_meta_contents(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        with open(out / "run_meta.json") as f:
            meta = json.load(f)
        assert meta["ref_instrument_ids"] == REF_IDS
        assert meta["n_refs"] == len(REF_IDS)
        assert meta["grid_name"] == "test_multi"
        assert "elapsed_seconds" in meta

    def test_default_output_dir(self, patched_sweep_runner, sweep_config, tmp_path, monkeypatch):
        """When output_dir is None, auto-generates runs/<grid_name>/<timestamp>."""
        monkeypatch.chdir(tmp_path)
        result = MultiRefSweepRunner.run(sweep_config, REF_IDS)
        assert result.output_dir is not None
        assert "test_multi" in str(result.output_dir)


# =========================================================================
# MultiRefResultsReader — round-trip
# =========================================================================


class TestMultiRefResultsReader:
    def test_load_round_trip(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        original = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=out,
        )
        loaded = MultiRefResultsReader.load(out)
        assert isinstance(loaded, MultiRefRunData)
        assert loaded.ref_instrument_ids == REF_IDS
        assert len(loaded.cross_ref_summary) == len(original.cross_ref_summary)
        assert len(loaded.per_ref_best) == len(original.per_ref_best)
        assert len(loaded.param_leaderboard) == len(original.param_leaderboard)

    def test_load_ref(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        loaded = MultiRefResultsReader.load(out)
        # load_ref delegates to SweepResultsReader which expects full structure;
        # our fake writes a minimal summary.csv and run_meta.json
        ref_data = loaded.load_ref("UST_2Y")
        assert ref_data.directory == out / "UST_2Y"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MultiRefResultsReader.load(tmp_path / "nonexistent")

    def test_load_ref_static(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        ref_data = MultiRefResultsReader.load_ref(out, "UST_5Y")
        assert ref_data.directory == out / "UST_5Y"


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_single_ref(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, ["UST_2Y"], output_dir=tmp_path / "out",
        )
        assert len(result.per_ref_results) == 1
        assert len(result.per_ref_best) == 1

    def test_custom_ranking_metric(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS,
            output_dir=tmp_path / "out",
            ranking_metric="total_pnl",
        )
        # per_ref_best should be sorted by total_pnl
        vals = result.per_ref_best["total_pnl"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_empty_ref_list(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, [], output_dir=tmp_path / "out", save=False,
        )
        assert len(result.per_ref_results) == 0
        assert result.cross_ref_summary.empty
