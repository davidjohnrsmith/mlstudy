"""Tests for fly universe generation and parameter sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest import SizingMode
from mlstudy.trading.strategy.structures.specs.fly.old.fly_universe import (
    ParamGrid,
    build_and_backtest_many_flies,
    filter_valid_flies,
    fly_name,
    generate_flies_from_tenors,
    get_best_fly_params,
    summarize_by_fly,
    summarize_by_params,
)


class TestGenerateFliesFromTenors:
    """Tests for generate_flies_from_tenors."""

    def test_default_tenors(self):
        """Should generate flies from default tenors."""
        flies = generate_flies_from_tenors()

        assert len(flies) > 0
        # All should be tuples of 3
        for fly in flies:
            assert len(fly) == 3
            assert fly[0] < fly[1] < fly[2]

    def test_custom_tenors(self):
        """Should generate flies from custom tenors."""
        flies = generate_flies_from_tenors(tenors=[2, 5, 10])

        # Only one fly possible: 2, 5, 10
        assert len(flies) == 1
        assert flies[0] == (2, 5, 10)

    def test_multiple_flies(self):
        """Should generate all valid combinations."""
        flies = generate_flies_from_tenors(tenors=[2, 5, 10, 30], min_wing_spread=0.5)

        # Expected: (2,5,10), (2,5,30), (2,10,30), (5,10,30)
        assert len(flies) == 4
        assert (2, 5, 10) in flies
        assert (2, 5, 30) in flies
        assert (2, 10, 30) in flies
        assert (5, 10, 30) in flies

    def test_min_wing_spread(self):
        """Should filter out flies with narrow wing spreads."""
        # With tight tenors
        flies_wide = generate_flies_from_tenors(tenors=[1, 2, 3], min_wing_spread=2.0)
        flies_narrow = generate_flies_from_tenors(tenors=[1, 2, 3], min_wing_spread=0.5)

        # Wide spread should have fewer flies
        assert len(flies_wide) < len(flies_narrow)

    def test_symmetric_only(self):
        """Should only return symmetric flies when required."""
        tenors = [2, 5, 7, 10, 15]

        flies_all = generate_flies_from_tenors(tenors, require_symmetric=False)
        flies_sym = generate_flies_from_tenors(tenors, require_symmetric=True)

        # Symmetric should be subset
        assert len(flies_sym) <= len(flies_all)

        # Check symmetry
        for front, belly, back in flies_sym:
            assert abs((belly - front) - (back - belly)) < 0.01

    def test_ordering(self):
        """All flies should have front < belly < back."""
        flies = generate_flies_from_tenors([1, 2, 3, 5, 7, 10])

        for front, belly, back in flies:
            assert front < belly < back


class TestFlyName:
    """Tests for fly_name function."""

    def test_year_tenors(self):
        """Should format year tenors correctly."""
        assert fly_name(2, 5, 10) == "2y5y10y"
        assert fly_name(1, 2, 3) == "1y2y3y"

    def test_fractional_years(self):
        """Should format fractional years correctly."""
        assert fly_name(0.5, 1, 2) == "6m1y2y"

    def test_long_tenors(self):
        """Should handle long tenors."""
        assert fly_name(10, 20, 30) == "10y20y30y"


class TestParamGrid:
    """Tests for ParamGrid."""

    def test_default_params(self):
        """Should have default parameters."""
        grid = ParamGrid()

        assert len(grid.windows) > 0
        assert len(grid.entry_zs) > 0
        assert len(grid.exit_zs) > 0

    def test_iteration(self):
        """Should iterate over all combinations."""
        grid = ParamGrid(
            windows=[10, 20],
            entry_zs=[1.5, 2.0],
            exit_zs=[0.5],
            stop_zs=[4.0],
            sizing_modes=[SizingMode.FIXED_NOTIONAL],
            robust_zscores=[False],
        )

        combos = list(grid)

        # 2 windows * 2 entry * 1 exit * 1 stop * 1 sizing * 1 robust = 4
        assert len(combos) == 4

        # Each combo should be a dict
        for combo in combos:
            assert "window" in combo
            assert "entry_z" in combo
            assert "exit_z" in combo

    def test_length(self):
        """Should report correct length."""
        grid = ParamGrid(
            windows=[10, 20, 30],
            entry_zs=[2.0],
            exit_zs=[0.5],
            stop_zs=[4.0],
            sizing_modes=[SizingMode.FIXED_NOTIONAL],
            robust_zscores=[False],
        )

        assert len(grid) == 3


@pytest.fixture
def synthetic_panel():
    """Create synthetic panel for sweep tests."""
    np.random.seed(42)

    n_dates = 60
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")

    bonds = [
        ("bond_2y", 2.0, 20.0),
        ("bond_5y", 5.0, 50.0),
        ("bond_10y", 10.0, 100.0),
    ]

    records = []
    prev_yields = {b[0]: 3.0 + 0.3 * np.log(b[1] + 1) for b in bonds}

    for date in dates:
        for bond_id, ttm, dv01 in bonds:
            dy = np.random.randn() * 0.01
            new_yield = prev_yields[bond_id] + dy
            price = 100 - (new_yield - 4) * dv01 / 100

            records.append({
                "datetime": date,
                "bond_id": bond_id,
                "ttm_years": ttm,
                "yield": new_yield,
                "price": price,
                "dv01": dv01,
            })
            prev_yields[bond_id] = new_yield

    return pd.DataFrame(records)


class TestBuildAndBacktestManyFlies:
    """Tests for build_and_backtest_many_flies."""

    def test_single_fly_single_param(self, synthetic_panel):
        """Should run single fly with single param set."""
        flies = [(2, 5, 10)]
        param_grid = ParamGrid(
            windows=[10],
            entry_zs=[1.5],
            exit_zs=[0.3],
            stop_zs=[None],
            sizing_modes=[SizingMode.FIXED_NOTIONAL],
            robust_zscores=[False],
        )

        result_df = build_and_backtest_many_flies(
            synthetic_panel,
            flies=flies,
            param_grid=param_grid,
            verbose=False,
        )

        assert len(result_df) == 1
        assert "fly_name" in result_df.columns
        assert "sharpe_ratio" in result_df.columns
        assert result_df.iloc[0]["fly_name"] == "2y5y10y"

    def test_multiple_params(self, synthetic_panel):
        """Should run parameter sweep."""
        flies = [(2, 5, 10)]
        param_grid = ParamGrid(
            windows=[10, 20],
            entry_zs=[1.5, 2.0],
            exit_zs=[0.3],
            stop_zs=[None],
            sizing_modes=[SizingMode.FIXED_NOTIONAL],
            robust_zscores=[False],
        )

        result_df = build_and_backtest_many_flies(
            synthetic_panel,
            flies=flies,
            param_grid=param_grid,
            verbose=False,
        )

        # 2 windows * 2 entry = 4 combos
        assert len(result_df) == 4

    def test_returns_metrics(self, synthetic_panel):
        """Should return all expected metrics."""
        flies = [(2, 5, 10)]
        param_grid = ParamGrid(
            windows=[10],
            entry_zs=[1.5],
            exit_zs=[0.3],
            stop_zs=[None],
            sizing_modes=[SizingMode.FIXED_NOTIONAL],
            robust_zscores=[False],
        )

        result_df = build_and_backtest_many_flies(
            synthetic_panel,
            flies=flies,
            param_grid=param_grid,
            verbose=False,
        )

        expected_cols = [
            "fly_name", "front", "belly", "back",
            "window", "entry_z", "exit_z",
            "sharpe_ratio", "total_pnl", "max_drawdown",
        ]
        for col in expected_cols:
            assert col in result_df.columns


class TestSummaryFunctions:
    """Tests for summary functions."""

    @pytest.fixture
    def sample_sweep_df(self):
        """Create sample sweep results."""
        return pd.DataFrame({
            "fly_name": ["2y5y10y", "2y5y10y", "5y10y30y", "5y10y30y"],
            "front": [2, 2, 5, 5],
            "belly": [5, 5, 10, 10],
            "back": [10, 10, 30, 30],
            "window": [10, 20, 10, 20],
            "entry_z": [2.0, 2.0, 2.0, 2.0],
            "exit_z": [0.5, 0.5, 0.5, 0.5],
            "sharpe_ratio": [1.5, 1.8, 1.2, 1.4],
            "total_pnl": [100, 120, 80, 90],
            "max_drawdown": [-20, -15, -25, -22],
            "n_trades": [10, 12, 8, 9],
            "hit_rate": [0.6, 0.65, 0.55, 0.58],
        })

    def test_summarize_by_fly(self, sample_sweep_df):
        """Should summarize by fly."""
        summary = summarize_by_fly(sample_sweep_df, metric="sharpe_ratio", agg="mean")

        assert len(summary) == 2  # Two unique flies
        assert "fly_name" in summary.columns
        assert "sharpe_ratio" in summary.columns

    def test_summarize_by_params(self, sample_sweep_df):
        """Should summarize by parameters."""
        summary = summarize_by_params(sample_sweep_df, metric="sharpe_ratio", agg="mean")

        assert len(summary) == 2  # Two unique windows
        assert "window" in summary.columns
        assert "sharpe_ratio" in summary.columns

    def test_get_best_fly_params(self, sample_sweep_df):
        """Should return best fly/param combo."""
        best = get_best_fly_params(sample_sweep_df, metric="sharpe_ratio")

        assert best["sharpe_ratio"] == 1.8
        assert best["fly_name"] == "2y5y10y"
        assert best["window"] == 20

    def test_filter_valid_flies(self, sample_sweep_df):
        """Should filter by criteria."""
        filtered = filter_valid_flies(
            sample_sweep_df,
            min_sharpe=1.3,
        )

        assert len(filtered) == 3  # 1.5, 1.8, 1.4 pass
        assert all(filtered["sharpe_ratio"] >= 1.3)

    def test_filter_by_drawdown(self, sample_sweep_df):
        """Should filter by max drawdown."""
        filtered = filter_valid_flies(
            sample_sweep_df,
            max_drawdown=-20,  # Must be >= -20 (i.e., less severe)
        )

        assert len(filtered) == 2  # -20 and -15 pass
        assert all(filtered["max_drawdown"] >= -20)


class TestEndToEnd:
    """End-to-end tests for sweep pipeline."""

    def test_full_sweep_runs(self, synthetic_panel):
        """Full sweep should run without errors."""
        flies = generate_flies_from_tenors([2, 5, 10])
        param_grid = ParamGrid(
            windows=[10],
            entry_zs=[1.5],
            exit_zs=[0.3],
            stop_zs=[None],
            sizing_modes=[SizingMode.FIXED_NOTIONAL],
            robust_zscores=[False],
        )

        result_df = build_and_backtest_many_flies(
            synthetic_panel,
            flies=flies,
            param_grid=param_grid,
            verbose=False,
        )

        assert len(result_df) > 0

        # Can summarize
        fly_summary = summarize_by_fly(result_df)
        assert len(fly_summary) > 0

        # Can get best
        best = get_best_fly_params(result_df)
        assert "fly_name" in best.index
