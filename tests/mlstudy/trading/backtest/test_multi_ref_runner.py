"""Tests for MultiRefSweepRunner and MultiRefResultsReader."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
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
    partition_instruments,
)
from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_results_reader import (
    MultiRefResultsReader,
    MultiRefRunData,
)
from mlstudy.trading.backtest.common.sweep.sweep_rank import (
    RankingPlan,
    SweepRanker,
)
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_runner import SweepRunResult
from mlstudy.trading.backtest.common.sweep.sweep_types import SweepResultLight, SweepScenario
from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics


# =========================================================================
# Helpers
# =========================================================================

def _make_sweep_config(grid: dict | None = None, ranking_plan: RankingPlan | None = None) -> SweepConfig:
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
        ranking_plan=ranking_plan,
    )


def _make_fake_metrics(rng: np.random.Generator) -> BacktestMetrics:
    """Create a BacktestMetrics with random values."""
    return BacktestMetrics(
        total_pnl=rng.normal(10, 5),
        mean_daily_return=rng.normal(0.001, 0.0005),
        std_daily_return=abs(rng.normal(0.01, 0.005)),
        sharpe_ratio=rng.normal(1.0, 0.5),
        sortino_ratio=rng.normal(1.5, 0.5),
        max_drawdown=rng.uniform(-20, 0),
        max_drawdown_duration=int(rng.integers(1, 100)),
        calmar_ratio=rng.normal(0.5, 0.3),
        turnover_annual=rng.uniform(1, 10),
        avg_holding_period=rng.uniform(1, 30),
        hit_rate=rng.uniform(0.3, 0.7),
        profit_factor=rng.uniform(0.5, 3.0),
        avg_win=rng.uniform(0.1, 5.0),
        avg_loss=rng.uniform(-5.0, -0.1),
        win_loss_ratio=rng.uniform(0.5, 3.0),
        skewness=rng.normal(0, 1),
        kurtosis=rng.normal(3, 1),
        var_95=rng.uniform(-5, 0),
        cvar_95=rng.uniform(-8, -1),
        n_trades=int(rng.integers(5, 50)),
        pct_time_in_market=rng.uniform(0.3, 0.9),
    )


def _make_fake_table(ref_id: str, grid: dict | None = None) -> pd.DataFrame:
    """Build a fake summary table as SweepRunner would produce."""
    grid = grid or {"entry_z_threshold": [1.5, 2.0, 3.0]}
    rng = np.random.default_rng(hash(ref_id) % 2**32)
    rows = []
    for i, ez in enumerate(grid["entry_z_threshold"]):
        m = _make_fake_metrics(rng)
        row = {"name": f"test_multi_{i:03d}", "entry_z_threshold": ez}
        row.update(asdict(m))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_fake_sweep_result_lights(
    ref_id: str, grid: dict | None = None,
) -> list[SweepResultLight]:
    """Create fake SweepResultLight objects for a ref."""
    grid = grid or {"entry_z_threshold": [1.5, 2.0, 3.0]}
    rng = np.random.default_rng(hash(ref_id) % 2**32)
    results = []
    for i, ez in enumerate(grid["entry_z_threshold"]):
        cfg = MRBacktestConfig(
            ref_leg_idx=1,
            target_notional_ref=100.0,
            entry_z_threshold=ez,
        )
        scenario = SweepScenario(
            name=f"test_multi_{i:03d}",
            cfg=cfg,
            tags={"entry_z_threshold": ez},
        )
        results.append(SweepResultLight(
            scenario_idx=i,
            scenario=scenario,
            metrics=_make_fake_metrics(rng),
        ))
    return results


def _make_fake_run_result(
    ref_id: str,
    cfg: SweepConfig,
    output_dir: Path | None = None,
) -> SweepRunResult:
    """Create a fake SweepRunResult."""
    table = _make_fake_table(ref_id, dict(cfg.grid))
    lights = _make_fake_sweep_result_lights(ref_id, dict(cfg.grid))
    return SweepRunResult(
        config=cfg,
        raw=lights,
        table=table,
        output_dir=output_dir,
    )


REF_IDS = ["UST_2Y", "UST_5Y", "UST_10Y"]


@pytest.fixture
def sweep_config():
    return _make_sweep_config()


@pytest.fixture
def sweep_config_with_plan():
    plan = RankingPlan(
        primary_metrics=(("sharpe_ratio", 1.0),),
        tie_metrics=(("total_pnl", 0.5),),
    )
    return _make_sweep_config(ranking_plan=plan)


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
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        assert len(result.per_ref_best) == len(REF_IDS)
        assert set(result.per_ref_best["ref_instrument_id"]) == set(REF_IDS)

    def test_has_rank_column(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out",
        )
        assert "rank" in result.per_ref_best.columns
        assert list(result.per_ref_best["rank"]) == [1, 2, 3]


# =========================================================================
# Param leaderboard
# =========================================================================


class TestParamLeaderboard:
    def test_has_rank_column(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        lb = result.param_leaderboard
        assert "rank" in lb.columns
        assert "entry_z_threshold" in lb.columns

    def test_has_metric_columns(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        result = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=tmp_path / "out", top_n=100,
        )
        lb = result.param_leaderboard
        assert "sharpe_ratio" in lb.columns
        assert "total_pnl" in lb.columns

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
        base = out / "test_multi"
        assert (base / "cross_ref_summary.csv").exists()
        assert (base / "per_ref_best.csv").exists()
        assert (base / "param_leaderboard.csv").exists()
        assert (base / "run_meta.json").exists()
        assert (base / "avg_top_n_refs.csv").exists()
        assert (base / "avg_all_refs.csv").exists()

    def test_per_ref_subdirectories(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        for ref_id in REF_IDS:
            assert (base / ref_id).is_dir()
            assert (base / ref_id / "summary.csv").exists()

    def test_per_ref_param_leaderboard_saved(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        for ref_id in REF_IDS:
            assert (base / ref_id / "param_leaderboard.csv").exists()
            lb = pd.read_csv(base / ref_id / "param_leaderboard.csv")
            assert "rank" in lb.columns
            assert "entry_z_threshold" in lb.columns

    def test_run_meta_contents(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        with open(base / "run_meta.json") as f:
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

    def test_avg_top_n_refs_csv_contents(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        df = pd.read_csv(base / "avg_top_n_refs.csv")
        assert len(df) == 1
        assert "sharpe_ratio" in df.columns
        assert "total_pnl" in df.columns

    def test_avg_all_refs_csv_contents(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        df = pd.read_csv(base / "avg_all_refs.csv")
        assert len(df) == 1
        assert "sharpe_ratio" in df.columns


# =========================================================================
# MultiRefResultsReader — round-trip
# =========================================================================


class TestMultiRefResultsReader:
    def test_load_round_trip(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        original = MultiRefSweepRunner.run(
            sweep_config, REF_IDS, output_dir=out,
        )
        base = out / "test_multi"
        loaded = MultiRefResultsReader.load(base)
        assert isinstance(loaded, MultiRefRunData)
        assert loaded.ref_instrument_ids == REF_IDS
        assert len(loaded.cross_ref_summary) == len(original.cross_ref_summary)
        assert len(loaded.per_ref_best) == len(original.per_ref_best)
        assert len(loaded.param_leaderboard) == len(original.param_leaderboard)

    def test_load_ref(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        loaded = MultiRefResultsReader.load(base)
        ref_data = loaded.load_ref("UST_2Y")
        assert ref_data.directory == base / "UST_2Y"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MultiRefResultsReader.load(tmp_path / "nonexistent")

    def test_load_ref_static(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        MultiRefSweepRunner.run(sweep_config, REF_IDS, output_dir=out)
        base = out / "test_multi"
        ref_data = MultiRefResultsReader.load_ref(base, "UST_5Y")
        assert ref_data.directory == base / "UST_5Y"


# =========================================================================
# rank_dataframe
# =========================================================================


class TestRankDataframe:
    def test_basic_ranking(self):
        df = pd.DataFrame({
            "total_pnl": [10.0, 30.0, 20.0],
            "sharpe_ratio": [0.5, 1.5, 1.0],
        })
        plan = RankingPlan(primary_metrics=(("total_pnl", 1.0),))
        ranked = SweepRanker.rank_dataframe(df, plan)
        assert "rank" in ranked.columns
        assert list(ranked["rank"]) == [1, 2, 3]
        # total_pnl higher is better, so best first
        assert ranked.iloc[0]["total_pnl"] == 30.0
        assert ranked.iloc[1]["total_pnl"] == 20.0
        assert ranked.iloc[2]["total_pnl"] == 10.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["total_pnl", "sharpe_ratio"])
        ranked = SweepRanker.rank_dataframe(df)
        assert "rank" in ranked.columns
        assert len(ranked) == 0

    def test_default_plan(self):
        df = pd.DataFrame({"total_pnl": [5.0, 15.0, 10.0]})
        ranked = SweepRanker.rank_dataframe(df)
        assert ranked.iloc[0]["total_pnl"] == 15.0

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({"total_pnl": [10.0, 20.0]})
        plan = RankingPlan(
            primary_metrics=(("total_pnl", 1.0),),
            tie_metrics=(("nonexistent_metric_col", 1.0),),
        )
        # Should not raise, just skip the missing column
        ranked = SweepRanker.rank_dataframe(df, plan)
        assert len(ranked) == 2

    def test_multi_stage_ranking(self):
        df = pd.DataFrame({
            "total_pnl": [10.0, 10.0, 20.0],
            "sharpe_ratio": [2.0, 1.0, 0.5],
        })
        plan = RankingPlan(
            primary_metrics=(("total_pnl", 1.0),),
            tie_metrics=(("sharpe_ratio", 1.0),),
        )
        ranked = SweepRanker.rank_dataframe(df, plan)
        # total_pnl=20 is best, then the two 10s are broken by sharpe_ratio
        assert ranked.iloc[0]["total_pnl"] == 20.0
        assert ranked.iloc[1]["sharpe_ratio"] == 2.0
        assert ranked.iloc[2]["sharpe_ratio"] == 1.0


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

    def test_empty_ref_list(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config, [], output_dir=tmp_path / "out", save=False,
        )
        assert len(result.per_ref_results) == 0
        assert result.cross_ref_summary.empty

    def test_with_ranking_plan(self, patched_sweep_runner, sweep_config_with_plan, tmp_path):
        result = MultiRefSweepRunner.run(
            sweep_config_with_plan, REF_IDS, output_dir=tmp_path / "out",
        )
        assert isinstance(result, MultiRefSweepResult)
        assert "rank" in result.per_ref_best.columns
        assert "rank" in result.param_leaderboard.columns


# =========================================================================
# partition_instruments
# =========================================================================


class TestPartitionInstruments:
    def test_even_split(self):
        ids = ["A", "B", "C", "D", "E", "F"]
        assert partition_instruments(ids, 3, 0) == ["A", "B"]
        assert partition_instruments(ids, 3, 1) == ["C", "D"]
        assert partition_instruments(ids, 3, 2) == ["E", "F"]

    def test_uneven_split(self):
        ids = list("ABCDEFGHIJ")  # 10 items
        p0 = partition_instruments(ids, 3, 0)
        p1 = partition_instruments(ids, 3, 1)
        p2 = partition_instruments(ids, 3, 2)
        assert len(p0) == 4  # first partition gets extra
        assert len(p1) == 3
        assert len(p2) == 3

    def test_more_machines_than_instruments(self):
        ids = ["A", "B"]
        assert partition_instruments(ids, 5, 0) == ["A"]
        assert partition_instruments(ids, 5, 1) == ["B"]
        assert partition_instruments(ids, 5, 2) == []
        assert partition_instruments(ids, 5, 3) == []
        assert partition_instruments(ids, 5, 4) == []

    def test_single_partition_returns_all(self):
        ids = ["A", "B", "C"]
        assert partition_instruments(ids, 1, 0) == ids

    def test_full_coverage(self):
        ids = list("ABCDEFGHIJ")
        num_partitions = 3
        combined = []
        for i in range(num_partitions):
            combined.extend(partition_instruments(ids, num_partitions, i))
        assert combined == ids


# =========================================================================
# run_partition
# =========================================================================


class TestRunPartition:
    def test_runs_correct_subset(self, patched_sweep_runner, sweep_config, tmp_path):
        result = MultiRefSweepRunner.run_partition(
            sweep_config,
            REF_IDS,
            num_partitions=2,
            partition_index=0,
            output_dir=tmp_path / "out",
        )
        assert isinstance(result, MultiRefSweepResult)
        # Partition 0 of 3 refs split into 2 partitions -> 2 refs
        assert len(result.per_ref_results) == 2
        assert set(result.per_ref_results.keys()) == {"UST_2Y", "UST_5Y"}

    def test_output_in_partition_subdir(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        result = MultiRefSweepRunner.run_partition(
            sweep_config,
            REF_IDS,
            num_partitions=2,
            partition_index=0,
            output_dir=out,
        )
        expected_dir = out / "test_multi" / "partition_000"
        assert result.output_dir == expected_dir
        assert (expected_dir / "cross_ref_summary.csv").exists()
        assert (expected_dir / "per_ref_best.csv").exists()
        assert (expected_dir / "param_leaderboard.csv").exists()
        assert (expected_dir / "run_meta.json").exists()

    def test_run_meta_includes_partition_info(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        out = tmp_path / "out"
        MultiRefSweepRunner.run_partition(
            sweep_config,
            REF_IDS,
            num_partitions=2,
            partition_index=1,
            output_dir=out,
        )
        meta_path = out / "test_multi" / "partition_001" / "run_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["partition_index"] == 1
        assert meta["num_partitions"] == 2
        assert meta["all_ref_instrument_ids"] == REF_IDS

    def test_empty_partition_returns_empty(self, patched_sweep_runner, sweep_config, tmp_path):
        # 3 refs, 5 partitions -> partition 4 is empty
        result = MultiRefSweepRunner.run_partition(
            sweep_config,
            REF_IDS,
            num_partitions=5,
            partition_index=4,
            output_dir=tmp_path / "out",
        )
        assert result.output_dir is None
        assert result.cross_ref_summary.empty
        assert len(result.per_ref_results) == 0


# =========================================================================
# collect_partitions
# =========================================================================


class TestCollectPartitions:
    def _run_all_partitions(self, patched_sweep_runner, sweep_config, tmp_path, num_partitions=2):
        """Helper: run all partitions and return the base dir."""
        out = tmp_path / "out"
        for i in range(num_partitions):
            MultiRefSweepRunner.run_partition(
                sweep_config,
                REF_IDS,
                num_partitions=num_partitions,
                partition_index=i,
                output_dir=out,
            )
        return out / "test_multi"

    def test_combined_has_all_refs(self, patched_sweep_runner, sweep_config, tmp_path):
        base = self._run_all_partitions(patched_sweep_runner, sweep_config, tmp_path)
        result = MultiRefSweepRunner.collect_partitions(
            base, sweep_config, num_partitions=2,
        )
        ref_ids_in_summary = set(result.cross_ref_summary["ref_instrument_id"])
        assert ref_ids_in_summary == set(REF_IDS)

    def test_combined_per_ref_best(self, patched_sweep_runner, sweep_config, tmp_path):
        base = self._run_all_partitions(patched_sweep_runner, sweep_config, tmp_path)
        result = MultiRefSweepRunner.collect_partitions(
            base, sweep_config, num_partitions=2,
        )
        assert len(result.per_ref_best) == len(REF_IDS)
        assert "rank" in result.per_ref_best.columns

    def test_combined_param_leaderboard(self, patched_sweep_runner, sweep_config, tmp_path):
        base = self._run_all_partitions(patched_sweep_runner, sweep_config, tmp_path)
        result = MultiRefSweepRunner.collect_partitions(
            base, sweep_config, num_partitions=2,
        )
        assert "rank" in result.param_leaderboard.columns

    def test_combined_dir_has_expected_files(
        self, patched_sweep_runner, sweep_config, tmp_path,
    ):
        base = self._run_all_partitions(patched_sweep_runner, sweep_config, tmp_path)
        MultiRefSweepRunner.collect_partitions(
            base, sweep_config, num_partitions=2,
        )
        combined = base / "combined"
        assert (combined / "cross_ref_summary.csv").exists()
        assert (combined / "per_ref_best.csv").exists()
        assert (combined / "param_leaderboard.csv").exists()
        assert (combined / "avg_top_n_refs.csv").exists()
        assert (combined / "avg_all_refs.csv").exists()
        assert (combined / "run_meta.json").exists()

    def test_missing_partition_raises(self, patched_sweep_runner, sweep_config, tmp_path):
        out = tmp_path / "out"
        # Only run partition 0, but claim 2 partitions exist
        MultiRefSweepRunner.run_partition(
            sweep_config,
            REF_IDS,
            num_partitions=2,
            partition_index=0,
            output_dir=out,
        )
        base = out / "test_multi"
        with pytest.raises(FileNotFoundError):
            MultiRefSweepRunner.collect_partitions(
                base, sweep_config, num_partitions=2,
            )

    def test_load_combined_round_trip(self, patched_sweep_runner, sweep_config, tmp_path):
        base = self._run_all_partitions(patched_sweep_runner, sweep_config, tmp_path)
        result = MultiRefSweepRunner.collect_partitions(
            base, sweep_config, num_partitions=2,
        )
        loaded = MultiRefResultsReader.load_combined(base)
        assert isinstance(loaded, MultiRefRunData)
        assert len(loaded.cross_ref_summary) == len(result.cross_ref_summary)
        assert loaded.meta["source"] == "collect_partitions"
