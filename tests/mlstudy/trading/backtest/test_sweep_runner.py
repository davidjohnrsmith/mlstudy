"""Tests for the end-to-end sweep runner (sweep_runner.py + sweep_config.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlstudy.trading.backtest.mean_reversion import MRBacktestConfig, load_config_map, load_sweep_config_by_name
from mlstudy.trading.backtest.mean_reversion.configs.sweep_config import (
    load_sweep_config,
)
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_runner import (
    SweepRunResult,
    SweepRunner,
)
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_rank import RankingPlan
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import (
    MetricsOnlyResult,
    SweepResult,
    SweepSummary,
)


# =========================================================================
# Helpers (reuse the same market-data factory from test_mr_sweep.py)
# =========================================================================

def _make_book(mid_px, half_spread, level2_offset, base_sizes):
    T, N = mid_px.shape
    bid0 = mid_px - half_spread[None, :]
    ask0 = mid_px + half_spread[None, :]
    bid1 = bid0 - level2_offset[None, :]
    ask1 = ask0 + level2_offset[None, :]
    bid_px = np.stack([bid0, bid1], axis=2)
    ask_px = np.stack([ask0, ask1], axis=2)
    bid_sz = np.stack([base_sizes, 0.5 * base_sizes], axis=2)
    ask_sz = np.stack([base_sizes, 0.5 * base_sizes], axis=2)
    return bid_px, bid_sz, ask_px, ask_sz


def _make_scripted_inputs(T: int = 60):
    N = 3
    hedge_ratios = np.tile(
        np.array([-0.5, 1.0, -0.5], dtype=np.float64), (T, 1)
    )
    dv01_vals = np.array([0.02, 0.045, 0.08], dtype=np.float64)
    dv01 = np.tile(dv01_vals, (T, 1))

    base_px = np.array([99.0, 98.0, 97.0], dtype=np.float64)
    mid_px = np.tile(base_px, (T, 1))
    rng = np.random.default_rng(42)
    mid_px += np.cumsum(rng.normal(0, 0.001, (T, N)), axis=0)

    half_spread = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    level2_off = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    base_sizes = np.full((T, N), 1000.0, dtype=np.float64)
    bid_px, bid_sz, ask_px, ask_sz = _make_book(
        mid_px, half_spread, level2_off, base_sizes
    )

    zscore = np.zeros(T, dtype=np.float64)
    package_yield_bps = np.zeros(T, dtype=np.float64)
    expected_yield_pnl_bps = np.full(T, 5.0, dtype=np.float64)

    zscore[5] = 3.0
    zscore[6:20] = 2.5
    package_yield_bps[5] = 100.0
    package_yield_bps[6:20] = 100.0
    zscore[20] = -1.0
    package_yield_bps[20] = 94.0

    zscore[24] = 3.5
    package_yield_bps[24] = 100.0
    zscore[25:40] = 2.0
    package_yield_bps[25:40] = 100.0
    zscore[40] = 2.0
    package_yield_bps[40] = 106.0
    zscore[41:] = 0.0
    package_yield_bps[41:] = 100.0

    return dict(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        mid_px=mid_px,
        dv01=dv01,
        zscore=zscore,
        expected_yield_pnl_bps=expected_yield_pnl_bps,
        package_yield_bps=package_yield_bps,
        hedge_ratios=hedge_ratios,
    )


@pytest.fixture
def market_data():
    return _make_scripted_inputs()


# =========================================================================
# YAML fixtures
# =========================================================================

_FULL_MODE_YAML = {
    "grid_name": "test_full",
    "base_config": {
        "ref_leg_idx": 1,
        "target_notional_ref": 100.0,
        "entry_z_threshold": 2.0,
        "take_profit_zscore_soft_threshold": 0.5,
        "take_profit_yield_change_soft_threshold": 1.0,
        "take_profit_yield_change_hard_threshold": 3.0,
        "stop_loss_yield_change_hard_threshold": 5.0,
        "tp_quarantine_bars": 2,
        "sl_quarantine_bars": 3,
        "max_levels_to_cross": 2,
        "validate_scope": "ALL_LEGS",
        "use_jit": False,
    },
    "grid": {
        "entry_z_threshold": [1.5, 2.0, 3.0],
    },
    "sweep": {
        "backend": "serial",
        "mode": "full",
    },
}

_METRICS_ONLY_YAML = {
    **_FULL_MODE_YAML,
    "grid_name": "test_metrics",
    "sweep": {
        "backend": "serial",
        "mode": "metrics_only",
    },
}

_TOP_K_YAML = {
    **_FULL_MODE_YAML,
    "grid_name": "test_topk",
    "grid": {
        "entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "sweep": {
        "backend": "serial",
        "mode": "metrics_only",
        "keep_top_k_full": 2,
    },
    "rank": {
        "primary_metrics": [["total_pnl", 1.0]],
    },
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


@pytest.fixture
def full_mode_yaml(tmp_path):
    return _write_yaml(tmp_path / "full.yaml", _FULL_MODE_YAML)


@pytest.fixture
def metrics_only_yaml(tmp_path):
    return _write_yaml(tmp_path / "metrics_only.yaml", _METRICS_ONLY_YAML)


@pytest.fixture
def top_k_yaml(tmp_path):
    return _write_yaml(tmp_path / "top_k.yaml", _TOP_K_YAML)


# =========================================================================
# Config loading tests
# =========================================================================


class TestLoadSweepConfig:
    def test_loads_base_config(self, full_mode_yaml):
        cfg = load_sweep_config(full_mode_yaml)
        assert isinstance(cfg.base_config, MRBacktestConfig)
        assert cfg.base_config.ref_leg_idx == 1
        assert cfg.base_config.use_jit is False

    def test_loads_grid(self, full_mode_yaml):
        cfg = load_sweep_config(full_mode_yaml)
        assert "entry_z_threshold" in cfg.grid
        assert cfg.grid["entry_z_threshold"] == [1.5, 2.0, 3.0]

    def test_loads_grid_name(self, full_mode_yaml):
        cfg = load_sweep_config(full_mode_yaml)
        assert cfg.grid_name == "test_full"

    def test_loads_sweep_kwargs(self, full_mode_yaml):
        cfg = load_sweep_config(full_mode_yaml)
        assert cfg.sweep_kwargs["backend"] == "serial"
        assert cfg.sweep_kwargs["mode"] == "full"

    def test_no_rank_section_gives_none(self, full_mode_yaml):
        cfg = load_sweep_config(full_mode_yaml)
        assert cfg.ranking_plan is None

    def test_rank_section_builds_plan(self, top_k_yaml):
        cfg = load_sweep_config(top_k_yaml)
        assert isinstance(cfg.ranking_plan, RankingPlan)
        assert cfg.ranking_plan.primary_metrics == (("total_pnl", 1.0),)

    def test_ranking_plan_in_sweep_kwargs(self, top_k_yaml):
        cfg = load_sweep_config(top_k_yaml)
        assert "ranking_plan" in cfg.sweep_kwargs
        assert cfg.sweep_kwargs["ranking_plan"] is cfg.ranking_plan

    def test_grid_name_defaults_to_stem(self, tmp_path):
        data = {**_FULL_MODE_YAML}
        del data["grid_name"]
        path = _write_yaml(tmp_path / "my_config.yaml", data)
        cfg = load_sweep_config(path)
        assert cfg.grid_name == "my_config"

    def test_empty_grid_raises(self, tmp_path):
        data = {**_FULL_MODE_YAML, "grid": {"entry_z_threshold": []}}
        path = _write_yaml(tmp_path / "bad.yaml", data)
        with pytest.raises(ValueError, match="non-empty list"):
            load_sweep_config(path)

    def test_invalid_top_level(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- a list\n- not a dict\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_sweep_config(path)

    def test_rank_simple_string_format(self, tmp_path):
        data = {
            **_FULL_MODE_YAML,
            "rank": {"primary_metrics": ["total_pnl", "sharpe_ratio"]},
        }
        path = _write_yaml(tmp_path / "simple_rank.yaml", data)
        cfg = load_sweep_config(path)
        assert cfg.ranking_plan.primary_metrics == (
            ("total_pnl", 1.0),
            ("sharpe_ratio", 1.0),
        )


class TestConfigMap:
    def test_load_config_map(self, tmp_path):
        map_data = {"alpha": "alpha.yaml", "beta": "beta.yaml"}
        map_path = _write_yaml(tmp_path / "map.yaml", map_data)
        result = load_config_map(map_path)
        assert result == {"alpha": "alpha.yaml", "beta": "beta.yaml"}

    def test_missing_map_returns_empty(self, tmp_path):
        result = load_config_map(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_load_by_name(self, tmp_path):
        # Write config
        cfg_path = _write_yaml(tmp_path / "my.yaml", _FULL_MODE_YAML)
        # Write map pointing to it
        map_data = {"my_run": "my.yaml"}
        map_path = _write_yaml(tmp_path / "map.yaml", map_data)

        cfg = load_sweep_config_by_name("my_run", config_map_path=map_path)
        assert cfg.grid_name == "test_full"

    def test_unknown_name_raises(self, tmp_path):
        map_data = {"alpha": "alpha.yaml"}
        map_path = _write_yaml(tmp_path / "map.yaml", map_data)
        with pytest.raises(KeyError, match="not found"):
            load_sweep_config_by_name("missing", config_map_path=map_path)


# =========================================================================
# run_sweep_from_config — full mode
# =========================================================================


class TestRunSweepFromConfigFull:
    def test_returns_sweep_run_result(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert isinstance(result, SweepRunResult)

    def test_raw_contains_sweep_results(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.raw) == 3
        for r in result.raw:
            assert isinstance(r, SweepResult)

    def test_table_is_dataframe(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert isinstance(result.table, pd.DataFrame)
        assert len(result.table) == 3
        assert "name" in result.table.columns
        assert "total_pnl" in result.table.columns

    def test_top_full_property(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        # In full mode, top_full returns the raw list
        assert result.top_full is not None
        assert len(result.top_full) == 3

    def test_all_metrics_property_none_in_full(
        self, full_mode_yaml, market_data, tmp_path
    ):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.all_metrics is None

    def test_config_preserved(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config.grid_name == "test_full"
        assert result.config.base_config.ref_leg_idx == 1


# =========================================================================
# run_sweep_from_config — metrics_only mode
# =========================================================================


class TestRunSweepFromConfigMetricsOnly:
    def test_raw_contains_metrics_only_results(
        self, metrics_only_yaml, market_data, tmp_path
    ):
        result = SweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.raw) == 3
        for r in result.raw:
            assert isinstance(r, MetricsOnlyResult)

    def test_all_metrics_property(self, metrics_only_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.all_metrics is not None
        assert len(result.all_metrics) == 3

    def test_top_full_property_none(self, metrics_only_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.top_full is None


# =========================================================================
# run_sweep_from_config — top-K rerun
# =========================================================================


class TestRunSweepFromConfigTopK:
    def test_returns_sweep_summary(self, top_k_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert isinstance(result.raw, SweepSummary)

    def test_all_metrics_count(self, top_k_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.all_metrics) == 5

    def test_top_full_count(self, top_k_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.top_full) == 2

    def test_top_full_are_sweep_results(self, top_k_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        for sr in result.top_full:
            assert isinstance(sr, SweepResult)
            assert sr.results is not None

    def test_table_from_all_metrics(self, top_k_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.table) == 5


# =========================================================================
# Config resolution (SweepConfig object, YAML path, config-map name)
# =========================================================================


class TestConfigResolution:
    def test_accepts_sweep_config_object(self, full_mode_yaml, market_data, tmp_path):
        cfg = load_sweep_config(full_mode_yaml)
        result = SweepRunner.run_sweep_from_config(
            cfg, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config is cfg

    def test_accepts_yaml_path_as_string(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            str(full_mode_yaml), market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config.grid_name == "test_full"

    def test_accepts_yaml_path_as_path(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            Path(full_mode_yaml), market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config.grid_name == "test_full"

    def test_accepts_config_map_name(self, tmp_path, market_data):
        cfg_path = _write_yaml(tmp_path / "my.yaml", _FULL_MODE_YAML)
        map_data = {"my_run": "my.yaml"}
        map_path = _write_yaml(tmp_path / "map.yaml", map_data)

        result = SweepRunner.run_sweep_from_config(
            "my_run",
            market_data=market_data,
            output_dir=tmp_path / "out",
            config_map_path=map_path,
        )
        assert result.config.grid_name == "test_full"


# =========================================================================
# Market data validation
# =========================================================================


class TestMarketDataValidation:
    def test_no_market_data_raises(self, full_mode_yaml, tmp_path):
        with pytest.raises(ValueError, match="No market data"):
            SweepRunner.run_sweep_from_config(
                full_mode_yaml, output_dir=tmp_path / "out"
            )

    def test_missing_key_raises(self, full_mode_yaml, tmp_path, market_data):
        del market_data["zscore"]
        with pytest.raises(ValueError, match="Missing market data"):
            SweepRunner.run_sweep_from_config(
                full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
            )

    def test_market_data_kwargs(self, full_mode_yaml, tmp_path, market_data):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, output_dir=tmp_path / "out", **market_data
        )
        assert len(result.raw) == 3

    def test_kwargs_override_dict(self, full_mode_yaml, tmp_path, market_data):
        # Pass all via dict, then override one via kwargs — should not error
        override_zscore = market_data["zscore"].copy()
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml,
            market_data=market_data,
            output_dir=tmp_path / "out",
            zscore=override_zscore,
        )
        assert len(result.raw) == 3


# =========================================================================
# Persistence
# =========================================================================


class TestRunnerPersistence:
    def test_creates_output_dir(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        assert out.exists()

    def test_saves_config_snapshot(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        snapshot_path = out / "config_snapshot.yaml"
        assert snapshot_path.exists()

        with open(snapshot_path) as f:
            snapshot = yaml.safe_load(f)
        assert snapshot["grid_name"] == "test_full"
        assert "base_config" in snapshot
        assert "grid" in snapshot

    def test_saves_run_metadata(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        meta_path = out / "run_meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["grid_name"] == "test_full"
        assert meta["n_scenarios"] == 3
        assert "timestamp" in meta
        assert "elapsed_seconds" in meta

    def test_saves_summary_csv(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        csv_path = out / "summary.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "name" in df.columns

    def test_saves_all_metrics_csv(self, metrics_only_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=out
        )
        csv_path = out / "all_metrics.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "scenario_idx" in df.columns
        assert "total_pnl" in df.columns

    def test_saves_full_results_npy(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        full_dir = out / "full"
        assert full_dir.exists()
        # 3 scenarios
        scenario_dirs = sorted(full_dir.iterdir())
        assert len(scenario_dirs) == 3
        # Each has spec.json and .npy arrays
        for sd in scenario_dirs:
            assert (sd / "spec.json").exists()
            assert (sd / "equity.npy").exists()
            assert (sd / "pnl.npy").exists()

    def test_saves_top_k_structure(self, top_k_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        SweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=out
        )
        # Should have all_metrics.csv + top_full/
        assert (out / "all_metrics.csv").exists()
        top_full_dir = out / "top_full"
        assert top_full_dir.exists()
        scenario_dirs = sorted(top_full_dir.iterdir())
        assert len(scenario_dirs) == 2

    def test_save_false_no_output(self, full_mode_yaml, market_data, tmp_path):
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, save=False
        )
        assert result.output_dir is None
        # No runs/ directory created (check in cwd -- use tmp_path as proxy)
        assert not (tmp_path / "output").exists()

    def test_output_dir_returned(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        assert result.output_dir == out

    def test_default_output_dir_has_timestamp(
        self, full_mode_yaml, market_data, tmp_path, monkeypatch
    ):
        # Run from tmp_path so runs/ is created there
        monkeypatch.chdir(tmp_path)
        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data
        )
        assert result.output_dir is not None
        assert "test_full" in str(result.output_dir)
        assert result.output_dir.exists()


# =========================================================================
# Consistency: runner output matches direct run_sweep
# =========================================================================


class TestConsistencyWithDirectSweep:
    def test_full_mode_pnl_matches(self, full_mode_yaml, market_data, tmp_path):
        from mlstudy.trading.backtest.mean_reversion.sweep.sweep import SweepExecutor
        from mlstudy.trading.backtest.mean_reversion.sweep.sweep_build import ScenarioBuilder

        cfg = load_sweep_config(full_mode_yaml)
        scenarios = ScenarioBuilder.make_scenarios(cfg.base_config, cfg.grid, name_prefix=cfg.grid_name)
        direct = SweepExecutor.run_sweep(scenarios, **market_data, **cfg.sweep_kwargs)

        result = SweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )

        assert len(result.raw) == len(direct)
        for r, d in zip(result.raw, direct):
            assert r.metrics.total_pnl == pytest.approx(d.metrics.total_pnl)
            assert r.metrics.sharpe_ratio == pytest.approx(d.metrics.sharpe_ratio)

    def test_metrics_only_pnl_matches(
        self, metrics_only_yaml, market_data, tmp_path
    ):
        from mlstudy.trading.backtest.mean_reversion.sweep.sweep import SweepExecutor
        from mlstudy.trading.backtest.mean_reversion.sweep.sweep_build import ScenarioBuilder

        cfg = load_sweep_config(metrics_only_yaml)
        scenarios = ScenarioBuilder.make_scenarios(cfg.base_config, cfg.grid, name_prefix=cfg.grid_name)
        direct = SweepExecutor.run_sweep(scenarios, **market_data, **cfg.sweep_kwargs)

        result = SweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )

        assert len(result.raw) == len(direct)
        for r, d in zip(result.raw, direct):
            assert r.total_pnl == pytest.approx(d.total_pnl)
