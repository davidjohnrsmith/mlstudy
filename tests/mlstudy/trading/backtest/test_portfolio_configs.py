"""Tests for portfolio sweep configs, config map, and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlstudy.trading.backtest.portfolio.configs.backtest_config import (
    PortfolioBacktestConfig,
)
from mlstudy.trading.backtest.portfolio.configs.sweep_config import (
    PortfolioSweepConfig,
    RankingPlan,
    load_sweep_config,
    _build_base_config,
    _build_grid,
    _build_ranking_plan,
)
from mlstudy.trading.backtest.portfolio.configs.utils import (
    load_config_map,
    load_sweep_config_by_name,
    _resolve_config,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[4] / (
    "src/mlstudy/trading/backtest/portfolio/configs"
)
_TUNING_DIR = _CONFIGS_DIR / "tuning_configs"


# =========================================================================
# _build helpers
# =========================================================================


_FULL_BASE_CONFIG = dict(
    use_greedy=False,
    gross_dv01_cap=100.0, top_k=10, z_inc=2.0, p_inc=0.05,
    z_dec=1.0, p_dec=0.10, alpha_thr_inc=1.0, alpha_thr_dec=0.5,
    max_levels=3, haircut=1.0, min_qty_trade=0.0,
    min_fill_ratio=0.0, cooldown_bars=0, cooldown_mode=0,
    min_maturity_inc=0.0, initial_capital=1_000_000.0,
    close_time="none",
)


class TestBuildBaseConfig:
    def test_all_fields_no_grid(self):
        cfg = _build_base_config(_FULL_BASE_CONFIG, {})
        assert isinstance(cfg, PortfolioBacktestConfig)
        assert cfg.gross_dv01_cap == 100.0

    def test_grid_fills_missing_base_fields(self):
        base = {k: v for k, v in _FULL_BASE_CONFIG.items() if k != "top_k"}
        grid = {"top_k": [5, 10, 20]}
        cfg = _build_base_config(base, grid)
        assert cfg.top_k == 5  # first grid value used

    def test_missing_fields_rejected(self):
        with pytest.raises(TypeError):
            _build_base_config({}, {})

    def test_custom_values(self):
        raw = {**_FULL_BASE_CONFIG, "top_k": 20, "haircut": 0.5}
        cfg = _build_base_config(raw, {})
        assert cfg.top_k == 20
        assert cfg.haircut == 0.5


class TestBuildGrid:
    def test_valid(self):
        grid = _build_grid({"z_inc": [1.0, 2.0], "top_k": [5, 10]})
        assert grid["z_inc"] == [1.0, 2.0]

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="non-empty list"):
            _build_grid({"z_inc": []})

    def test_non_list_rejected(self):
        with pytest.raises(ValueError, match="non-empty list"):
            _build_grid({"z_inc": 2.0})


class TestBuildRankingPlan:
    def test_none(self):
        assert _build_ranking_plan(None) is None

    def test_simple_strings(self):
        plan = _build_ranking_plan({"primary_metrics": ["total_pnl", "sharpe_ratio"]})
        assert plan.primary_metrics == (("total_pnl", 1.0), ("sharpe_ratio", 1.0))

    def test_explicit_weights(self):
        plan = _build_ranking_plan({
            "primary_metrics": [["total_pnl", 0.5], ["sharpe_ratio", 0.5]],
        })
        assert plan.primary_metrics == (("total_pnl", 0.5), ("sharpe_ratio", 0.5))

    def test_mixed(self):
        plan = _build_ranking_plan({
            "primary_metrics": ["total_pnl", ["sharpe_ratio", 2.0]],
            "tie_metrics": [["max_drawdown", 1.0]],
        })
        assert plan.primary_metrics[0] == ("total_pnl", 1.0)
        assert plan.primary_metrics[1] == ("sharpe_ratio", 2.0)
        assert plan.tie_metrics == (("max_drawdown", 1.0),)

    def test_bad_entry_rejected(self):
        with pytest.raises(ValueError, match="string or"):
            _build_ranking_plan({"primary_metrics": [123]})


# =========================================================================
# YAML loading
# =========================================================================


class TestLoadSweepConfig:
    def test_load_portfolio_yaml(self):
        cfg = load_sweep_config(_TUNING_DIR / "portfolio.yaml")
        assert isinstance(cfg, PortfolioSweepConfig)
        assert cfg.grid_name == "portfolio_grid_v1"
        assert isinstance(cfg.base_config, PortfolioBacktestConfig)
        assert "z_inc" in cfg.grid
        assert len(cfg.grid["z_inc"]) >= 2

    def test_load_debug_yaml(self):
        cfg = load_sweep_config(_TUNING_DIR / "portfolio_debug.yaml")
        assert cfg.grid_name == "portfolio_grid_debug"
        assert cfg.ranking_plan is not None
        assert len(cfg.ranking_plan.primary_metrics) > 0

    def test_base_config_values_from_yaml(self):
        cfg = load_sweep_config(_TUNING_DIR / "portfolio.yaml")
        # gross_dv01_cap comes from grid (first value)
        assert cfg.base_config.gross_dv01_cap == 50.0
        # initial_capital comes from base_config
        assert cfg.base_config.initial_capital == 1_000_000.0

    def test_sweep_kwargs_from_yaml(self):
        cfg = load_sweep_config(_TUNING_DIR / "portfolio.yaml")
        assert "backend" in cfg.sweep_kwargs
        assert cfg.sweep_kwargs["backend"] == "process"

    def test_ranking_plan_from_yaml(self):
        cfg = load_sweep_config(_TUNING_DIR / "portfolio.yaml")
        assert cfg.ranking_plan is not None
        assert cfg.ranking_plan.primary_metrics == (("total_pnl", 1.0),)


# =========================================================================
# Config map
# =========================================================================


class TestConfigMap:
    def test_load_default_map(self):
        cmap = load_config_map()
        assert "portfolio_grid_v1" in cmap
        assert "portfolio_grid_debug" in cmap

    def test_load_by_name(self):
        cfg = load_sweep_config_by_name("portfolio_grid_v1")
        assert isinstance(cfg, PortfolioSweepConfig)
        assert cfg.grid_name == "portfolio_grid_v1"

    def test_load_by_name_debug(self):
        cfg = load_sweep_config_by_name("portfolio_grid_debug")
        assert cfg.grid_name == "portfolio_grid_debug"

    def test_missing_name_raises(self):
        with pytest.raises(KeyError, match="not found"):
            load_sweep_config_by_name("nonexistent_config")

    def test_missing_map_returns_empty(self, tmp_path):
        cmap = load_config_map(tmp_path / "does_not_exist.yaml")
        assert cmap == {}


# =========================================================================
# _resolve_config
# =========================================================================


class TestResolveConfig:
    def test_passthrough_object(self):
        cfg = load_sweep_config_by_name("portfolio_grid_v1")
        assert _resolve_config(cfg) is cfg

    def test_from_path(self):
        cfg = _resolve_config(str(_TUNING_DIR / "portfolio.yaml"))
        assert isinstance(cfg, PortfolioSweepConfig)

    def test_from_name(self):
        cfg = _resolve_config("portfolio_grid_v1")
        assert isinstance(cfg, PortfolioSweepConfig)
