"""Tests for the end-to-end portfolio sweep runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mlstudy.trading.backtest.common.sweep.sweep_build import ScenarioBuilder
from mlstudy.trading.backtest.portfolio.configs.backtest_config import PortfolioBacktestConfig
from mlstudy.trading.backtest.portfolio.configs.sweep_config import (
    PortfolioSweepConfig,
    load_sweep_config,
)
from mlstudy.trading.backtest.portfolio.configs.utils import (
    load_sweep_config_by_name,
    load_config_map,
)
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan
from mlstudy.trading.backtest.portfolio.sweep.sweep_runner import (
    SweepRunResult,
    PortfolioSweepRunner,
)
from mlstudy.trading.backtest.common.sweep.sweep_types import (
    SweepResultLight,
    SweepResult,
    SweepSummary,
)


# =========================================================================
# Helpers — build synthetic market data
# =========================================================================

def _make_market(T, B, L=3, mid_base=100.0, spread_bps=10.0):
    mid_px = np.full((T, B), mid_base, dtype=np.float64)
    spread = mid_base * spread_bps / 1e4
    bid_px = np.zeros((T, B, L), dtype=np.float64)
    bid_sz = np.zeros((T, B, L), dtype=np.float64)
    ask_px = np.zeros((T, B, L), dtype=np.float64)
    ask_sz = np.zeros((T, B, L), dtype=np.float64)
    for lev in range(L):
        bid_px[:, :, lev] = mid_base - spread * (lev + 1)
        ask_px[:, :, lev] = mid_base + spread * (lev + 1)
        bid_sz[:, :, lev] = 1000.0
        ask_sz[:, :, lev] = 1000.0
    return bid_px, bid_sz, ask_px, ask_sz, mid_px


def _make_hedge_market(T, H, L=3, mid_base=100.0, spread_bps=10.0):
    mid_px = np.full((T, H), mid_base, dtype=np.float64)
    spread = mid_base * spread_bps / 1e4
    bid_px = np.zeros((T, H, L), dtype=np.float64)
    bid_sz = np.zeros((T, H, L), dtype=np.float64)
    ask_px = np.zeros((T, H, L), dtype=np.float64)
    ask_sz = np.zeros((T, H, L), dtype=np.float64)
    for lev in range(L):
        bid_px[:, :, lev] = mid_base - spread * (lev + 1)
        ask_px[:, :, lev] = mid_base + spread * (lev + 1)
        bid_sz[:, :, lev] = 10000.0
        ask_sz[:, :, lev] = 10000.0
    return bid_px, bid_sz, ask_px, ask_sz, mid_px


def _make_portfolio_inputs(T: int = 30, B: int = 2, H: int = 1):
    L = 3
    bid_px, bid_sz, ask_px, ask_sz, mid_px = _make_market(T, B, L, spread_bps=5.0)
    h_bid_px, h_bid_sz, h_ask_px, h_ask_sz, h_mid_px = _make_hedge_market(T, H, L)

    return dict(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        mid_px=mid_px,
        dv01=np.full((T, B), 0.01, dtype=np.float64),
        fair_price=mid_px + 1.0,
        zscore=np.full((T, B), 3.0, dtype=np.float64),
        adf_p_value=np.full((T, B), 0.01, dtype=np.float64),
        tradable=np.ones(B, dtype=np.float64),
        pos_limits_long=np.full(B, 1e6, dtype=np.float64),
        pos_limits_short=np.full(B, -1e6, dtype=np.float64),
        max_trade_notional_inc=np.full(B, np.inf, dtype=np.float64),
        max_trade_notional_dec=np.full(B, np.inf, dtype=np.float64),
        qty_step=np.zeros(B, dtype=np.float64),
        min_qty_trade=np.zeros(B, dtype=np.float64),
        maturity=np.full((T, B), 5.0, dtype=np.float64),
        issuer_bucket=np.zeros(B, dtype=np.int64),
        maturity_bucket=np.zeros((T, B), dtype=np.int64),
        issuer_dv01_caps=np.empty(0, dtype=np.float64),
        mat_bucket_dv01_caps=np.empty(0, dtype=np.float64),
        hedge_bid_px=h_bid_px,
        hedge_bid_sz=h_bid_sz,
        hedge_ask_px=h_ask_px,
        hedge_ask_sz=h_ask_sz,
        hedge_mid_px=h_mid_px,
        hedge_dv01=np.full((T, H), 0.01, dtype=np.float64),
        hedge_ratios=np.full((T, B, H), -0.5, dtype=np.float64),
        hedge_qty_step=np.zeros(H, dtype=np.float64),
        hedge_min_qty_trade=np.zeros(H, dtype=np.float64),
        instrument_ids=[f"INST_{i}" for i in range(B)],
        datetimes=pd.bdate_range("2024-01-02", periods=T, freq="B").values,
    )


@pytest.fixture
def market_data():
    return _make_portfolio_inputs()


# =========================================================================
# YAML fixtures
# =========================================================================

_FULL_MODE_YAML = {
    "grid_name": "test_portfolio_full",
    "base_config": {
        "use_greedy": False,
        "gross_dv01_cap": 100.0,
        "top_k": 10,
        "p_inc": 0.05,
        "z_dec": 1.0,
        "p_dec": 0.10,
        "alpha_thr_inc": 1.0,
        "alpha_thr_dec": 0.5,
        "max_levels": 3,
        "haircut": 1.0,
        "min_fill_ratio": 0.0,
        "cooldown_bars": 0,
        "cooldown_mode": 0,
        "min_maturity_inc": 0.0,
        "initial_capital": 1_000_000.0,
        "close_time": "none",
    },
    "grid": {
        "z_inc": [1.5, 2.0, 3.0],
    },
    "sweep": {
        "backend": "serial",
        "mode": "full",
    },
}

_METRICS_ONLY_YAML = {
    **_FULL_MODE_YAML,
    "grid_name": "test_portfolio_metrics",
    "sweep": {
        "backend": "serial",
        "mode": "metrics_only",
    },
}

_TOP_K_YAML = {
    **_FULL_MODE_YAML,
    "grid_name": "test_portfolio_topk",
    "grid": {
        "z_inc": [1.0, 1.5, 2.0, 2.5, 3.0],
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
# run_sweep_from_config — full mode
# =========================================================================


class TestRunSweepFromConfigFull:
    def test_returns_sweep_run_result(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert isinstance(result, SweepRunResult)

    def test_raw_contains_sweep_results(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.raw) == 3
        for r in result.raw:
            assert isinstance(r, SweepResult)

    def test_table_is_dataframe(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert isinstance(result.table, pd.DataFrame)
        assert len(result.table) == 3
        assert "name" in result.table.columns
        assert "total_pnl" in result.table.columns

    def test_top_full_property(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.top_full is not None
        assert len(result.top_full) == 3

    def test_all_metrics_property_none_in_full(
        self, full_mode_yaml, market_data, tmp_path
    ):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.all_metrics is None

    def test_config_preserved(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config.grid_name == "test_portfolio_full"


# =========================================================================
# run_sweep_from_config — metrics_only mode
# =========================================================================


class TestRunSweepFromConfigMetricsOnly:
    def test_raw_contains_metrics_only_results(
        self, metrics_only_yaml, market_data, tmp_path
    ):
        result = PortfolioSweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.raw) == 3
        for r in result.raw:
            assert isinstance(r, SweepResultLight)

    def test_all_metrics_property(self, metrics_only_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.all_metrics is not None
        assert len(result.all_metrics) == 3

    def test_top_full_property_none(self, metrics_only_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.top_full is None


# =========================================================================
# run_sweep_from_config — top-K rerun
# =========================================================================


class TestRunSweepFromConfigTopK:
    def test_returns_sweep_summary(self, top_k_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert isinstance(result.raw, SweepSummary)

    def test_all_metrics_count(self, top_k_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.all_metrics) == 5

    def test_top_full_count(self, top_k_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.top_full) == 2

    def test_top_full_are_sweep_results(self, top_k_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        for sr in result.top_full:
            assert isinstance(sr, SweepResult)
            assert sr.results is not None

    def test_table_from_all_metrics(self, top_k_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert len(result.table) == 5


# =========================================================================
# Config resolution
# =========================================================================


class TestConfigResolution:
    def test_accepts_sweep_config_object(self, full_mode_yaml, market_data, tmp_path):
        cfg = load_sweep_config(full_mode_yaml)
        result = PortfolioSweepRunner.run_sweep_from_config(
            cfg, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config is cfg

    def test_accepts_yaml_path_as_string(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            str(full_mode_yaml), market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config.grid_name == "test_portfolio_full"

    def test_accepts_yaml_path_as_path(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            Path(full_mode_yaml), market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.config.grid_name == "test_portfolio_full"

    def test_accepts_config_map_name(self, tmp_path, market_data):
        # Config map resolves names via tuning_configs/ subdir
        tuning_dir = tmp_path / "tuning_configs"
        tuning_dir.mkdir()
        cfg_path = _write_yaml(tuning_dir / "my.yaml", _FULL_MODE_YAML)
        map_data = {"my_run": "my.yaml"}
        map_path = _write_yaml(tmp_path / "map.yaml", map_data)

        result = PortfolioSweepRunner.run_sweep_from_config(
            "my_run",
            market_data=market_data,
            output_dir=tmp_path / "out",
            config_map_path=map_path,
        )
        assert result.config.grid_name == "test_portfolio_full"


# =========================================================================
# Market data validation
# =========================================================================


class TestMarketDataValidation:
    def test_no_market_data_raises(self, full_mode_yaml, tmp_path):
        with pytest.raises(ValueError, match="No market data"):
            PortfolioSweepRunner.run_sweep_from_config(
                full_mode_yaml, output_dir=tmp_path / "out"
            )

    def test_missing_key_raises(self, full_mode_yaml, tmp_path, market_data):
        del market_data["zscore"]
        with pytest.raises(ValueError, match="Missing market data"):
            PortfolioSweepRunner.run_sweep_from_config(
                full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
            )

    def test_market_data_kwargs(self, full_mode_yaml, tmp_path, market_data):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, output_dir=tmp_path / "out", **market_data
        )
        assert len(result.raw) == 3

    def test_kwargs_override_dict(self, full_mode_yaml, tmp_path, market_data):
        override_zscore = market_data["zscore"].copy()
        result = PortfolioSweepRunner.run_sweep_from_config(
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
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        assert out.exists()

    def test_saves_config_snapshot(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        snapshot_path = out / "config_snapshot.yaml"
        assert snapshot_path.exists()

        with open(snapshot_path) as f:
            snapshot = yaml.safe_load(f)
        assert snapshot["grid_name"] == "test_portfolio_full"
        assert "base_config" in snapshot
        assert "grid" in snapshot

    def test_saves_run_metadata(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        meta_path = out / "run_meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["grid_name"] == "test_portfolio_full"
        assert meta["n_scenarios"] == 3
        assert "timestamp" in meta
        assert "elapsed_seconds" in meta

    def test_saves_summary_csv(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        csv_path = out / "summary.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "name" in df.columns

    def test_saves_all_metrics_csv(self, metrics_only_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
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
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        full_dir = out / "full"
        assert full_dir.exists()
        scenario_dirs = sorted(full_dir.iterdir())
        assert len(scenario_dirs) == 3
        for sd in scenario_dirs:
            assert (sd / "spec.json").exists()
            assert (sd / "equity.npy").exists()
            assert (sd / "pnl.npy").exists()

    def test_saves_bar_df_and_trade_df_csv(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        full_dir = out / "full"
        for sd in sorted(full_dir.iterdir()):
            assert (sd / "bar_df.csv").exists()
            assert (sd / "trade_df.csv").exists()
            bar_df = pd.read_csv(sd / "bar_df.csv")
            assert "equity" in bar_df.columns
            assert len(bar_df) > 0

    def test_saves_top_k_structure(self, top_k_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
            top_k_yaml, market_data=market_data, output_dir=out
        )
        assert (out / "all_metrics.csv").exists()
        top_full_dir = out / "top_full"
        assert top_full_dir.exists()
        scenario_dirs = sorted(top_full_dir.iterdir())
        assert len(scenario_dirs) == 2

    def test_save_false_no_output(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, save=False
        )
        assert result.output_dir is None

    def test_output_dir_returned(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        assert result.output_dir == out

    def test_default_output_dir_has_timestamp(
        self, full_mode_yaml, market_data, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data
        )
        assert result.output_dir is not None
        assert "test_portfolio_full" in str(result.output_dir)
        assert result.output_dir.exists()


# =========================================================================
# Param leaderboard
# =========================================================================


class TestParamLeaderboard:
    def test_leaderboard_returned_on_result(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.param_leaderboard is not None
        assert isinstance(result.param_leaderboard, pd.DataFrame)
        assert "rank" in result.param_leaderboard.columns
        assert len(result.param_leaderboard) > 0

    def test_leaderboard_has_grid_params(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        cfg = result.config
        for key in cfg.grid:
            assert key in result.param_leaderboard.columns

    def test_leaderboard_has_metric_columns(self, full_mode_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert "total_pnl" in result.param_leaderboard.columns
        assert "sharpe_ratio" in result.param_leaderboard.columns

    def test_leaderboard_persisted_csv(self, full_mode_yaml, market_data, tmp_path):
        out = tmp_path / "output"
        PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=out
        )
        csv_path = out / "param_leaderboard.csv"
        assert csv_path.exists()
        lb = pd.read_csv(csv_path)
        assert "rank" in lb.columns
        assert "total_pnl" in lb.columns
        assert len(lb) > 0

    def test_leaderboard_no_save(self, full_mode_yaml, market_data):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, save=False
        )
        # Leaderboard should still be computed even without saving
        assert result.param_leaderboard is not None
        assert len(result.param_leaderboard) > 0

    def test_leaderboard_metrics_only_mode(self, metrics_only_yaml, market_data, tmp_path):
        result = PortfolioSweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )
        assert result.param_leaderboard is not None
        assert len(result.param_leaderboard) > 0

    def test_display_param_leaderboard(self, full_mode_yaml, market_data):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, save=False
        )
        output = result.display_param_leaderboard()
        assert "Param Leaderboard" in output
        assert "total_pnl" in output
        assert "sharpe_ratio" in output
        assert "rank" in output

    def test_display_param_leaderboard_empty(self):
        """display_param_leaderboard with no leaderboard returns message."""
        result = SweepRunResult(
            config=None, raw=[], table=pd.DataFrame(),
            output_dir=None, param_leaderboard=None,
        )
        assert "No param leaderboard" in result.display_param_leaderboard()

    def test_display_param_leaderboard_top_n(self, full_mode_yaml, market_data):
        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, save=False
        )
        output = result.display_param_leaderboard(top_n=1)
        assert "top 1" in output


# =========================================================================
# Consistency: runner output matches direct sweep
# =========================================================================


class TestConsistencyWithDirectSweep:
    def test_full_mode_pnl_matches(self, full_mode_yaml, market_data, tmp_path):
        from mlstudy.trading.backtest.portfolio.sweep.sweep import PortfolioSweepExecutor

        cfg = load_sweep_config(full_mode_yaml)
        scenarios = ScenarioBuilder.make_scenarios(
            cfg.base_config, cfg.grid, name_prefix=cfg.grid_name,
        )
        direct = PortfolioSweepExecutor.run_sweep(
            scenarios, **market_data, **cfg.sweep_kwargs,
        )

        result = PortfolioSweepRunner.run_sweep_from_config(
            full_mode_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )

        assert len(result.raw) == len(direct)
        for r, d in zip(result.raw, direct):
            assert r.metrics.total_pnl == pytest.approx(d.metrics.total_pnl)
            assert r.metrics.sharpe_ratio == pytest.approx(d.metrics.sharpe_ratio)

    def test_metrics_only_pnl_matches(
        self, metrics_only_yaml, market_data, tmp_path
    ):
        from mlstudy.trading.backtest.portfolio.sweep.sweep import PortfolioSweepExecutor

        cfg = load_sweep_config(metrics_only_yaml)
        scenarios = ScenarioBuilder.make_scenarios(
            cfg.base_config, cfg.grid, name_prefix=cfg.grid_name,
        )
        direct = PortfolioSweepExecutor.run_sweep(
            scenarios, **market_data, **cfg.sweep_kwargs,
        )

        result = PortfolioSweepRunner.run_sweep_from_config(
            metrics_only_yaml, market_data=market_data, output_dir=tmp_path / "out"
        )

        assert len(result.raw) == len(direct)
        for r, d in zip(result.raw, direct):
            assert r.metrics.total_pnl == pytest.approx(d.metrics.total_pnl)
