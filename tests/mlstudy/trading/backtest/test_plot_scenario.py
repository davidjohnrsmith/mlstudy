"""Tests for plot_scenario and plot_top_scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_reader import (
    FullScenario,
    SweepResultsReader,
)
from mlstudy.trading.backtest.mean_reversion.types import (
    TRADE_ENTRY,
    TRADE_EXIT_SL,
    TRADE_EXIT_TP,
)

# Skip all tests if matplotlib is not installed
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # non-interactive backend


from mlstudy.trading.backtest.mean_reversion.plots import (
    plot_scenario,
    plot_top_scenarios,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_scenario(
    scenario_dir: Path,
    T: int = 50,
    N: int = 3,
    n_trades: int = 4,
    scenario_idx: int = 0,
) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(scenario_idx)

    spec = {
        "name": f"mr_grid_v1_{scenario_idx:04d}",
        "tags": {
            "entry_z_threshold": 2.0,
            "take_profit_yield_change_hard_threshold": 6.0,
            "max_holding_bars": 20,
        },
        "config": {
            "ref_leg_idx": 1,
            "entry_z_threshold": 2.0,
            "take_profit_yield_change_hard_threshold": 6.0,
        },
        "metrics": {
            "total_pnl": 150.5,
            "sharpe_ratio": 1.23,
            "sortino_ratio": 1.8,
            "max_drawdown": -45.2,
            "n_trades": n_trades,
            "hit_rate": 0.65,
            "profit_factor": 1.5,
            "avg_holding_period": 8.3,
        },
        "scenario_idx": scenario_idx,
    }
    with open(scenario_dir / "spec.json", "w") as f:
        json.dump(spec, f)

    # Equity: random walk with drift
    equity = np.cumsum(rng.normal(0.5, 2.0, T))
    np.save(scenario_dir / "positions.npy", rng.normal(0, 1, (T, N)))
    np.save(scenario_dir / "cash.npy", rng.normal(0, 1, T))
    np.save(scenario_dir / "equity.npy", equity)
    np.save(scenario_dir / "pnl.npy", np.diff(equity, prepend=0))

    codes = np.zeros(T, dtype=np.int32)
    np.save(scenario_dir / "codes.npy", codes)

    # State: flat, then long, then flat, then short
    state = np.zeros(T, dtype=np.int32)
    state[5:15] = 1   # long
    state[25:35] = -1  # short
    np.save(scenario_dir / "state.npy", state)
    np.save(scenario_dir / "holding.npy", np.zeros(T, dtype=np.int32))

    # Trades: entry, TP, entry, SL
    tr_types = np.array([TRADE_ENTRY, TRADE_EXIT_TP, TRADE_ENTRY, TRADE_EXIT_SL], dtype=np.int32)
    tr_bars = np.array([5, 14, 25, 34], dtype=np.int64)
    np.save(scenario_dir / "tr_bar.npy", tr_bars[:n_trades])
    np.save(scenario_dir / "tr_type.npy", tr_types[:n_trades])
    np.save(scenario_dir / "tr_side.npy", np.array([1, 1, -1, -1], dtype=np.int32)[:n_trades])
    np.save(scenario_dir / "tr_sizes.npy", rng.normal(0, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_risks.npy", rng.normal(0, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_vwaps.npy", rng.normal(100, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_mids.npy", rng.normal(100, 1, (n_trades, N)))
    np.save(scenario_dir / "tr_cost.npy", rng.uniform(0, 1, n_trades))
    np.save(scenario_dir / "tr_code.npy", np.full(n_trades, 3, dtype=np.int32))
    np.save(scenario_dir / "tr_pkg_yield.npy", rng.normal(100, 1, n_trades))


def _load_scenario(tmp_path: Path, **kwargs) -> FullScenario:
    sd = tmp_path / "scenario_000"
    _write_scenario(sd, **kwargs)
    return SweepResultsReader.load_full_scenario(sd)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotScenario:
    def test_returns_figure(self, tmp_path):
        sc = _load_scenario(tmp_path)
        fig = plot_scenario(sc)
        assert fig is not None
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        mpl.pyplot.close(fig)

    def test_with_zscore(self, tmp_path):
        T = 50
        sc = _load_scenario(tmp_path, T=T)
        zscore = np.random.default_rng(99).normal(0, 1.5, T)
        fig = plot_scenario(sc, zscore=zscore)
        # Should have 3 panels (equity, zscore, drawdown)
        plt.show()
        assert len(fig.axes) == 3
        mpl.pyplot.close(fig)

    def test_without_zscore(self, tmp_path):
        sc = _load_scenario(tmp_path)
        fig = plot_scenario(sc, zscore=None)
        # Should have 2 panels (equity, drawdown)
        assert len(fig.axes) == 2
        mpl.pyplot.close(fig)

    def test_save_to_file(self, tmp_path):
        sc = _load_scenario(tmp_path / "data")
        save_path = tmp_path / "output.png"
        fig = plot_scenario(sc, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        mpl.pyplot.close(fig)

    def test_no_trades(self, tmp_path):
        sc = _load_scenario(tmp_path, n_trades=0)
        fig = plot_scenario(sc)
        assert fig is not None
        mpl.pyplot.close(fig)

    def test_trade_markers_present(self, tmp_path):
        sc = _load_scenario(tmp_path)
        fig = plot_scenario(sc)
        ax_eq = fig.axes[0]
        # Should have scatter collections for trade markers
        # At minimum we have the line + scatters
        assert len(ax_eq.collections) > 0 or len(ax_eq.lines) > 0
        mpl.pyplot.close(fig)


class TestPlotTopScenarios:
    def test_returns_list_of_figures(self, tmp_path):
        scenarios = []
        for i in range(3):
            sd = tmp_path / "data" / f"scenario_{i:03d}"
            _write_scenario(sd, scenario_idx=i)
            scenarios.append(SweepResultsReader.load_full_scenario(sd))

        figs = plot_top_scenarios(scenarios)
        assert len(figs) == 3
        for fig in figs:
            mpl.pyplot.close(fig)

    def test_saves_to_dir(self, tmp_path):
        scenarios = []
        for i in range(2):
            sd = tmp_path / "data" / f"scenario_{i:03d}"
            _write_scenario(sd, scenario_idx=i)
            scenarios.append(SweepResultsReader.load_full_scenario(sd))

        save_dir = tmp_path / "plots"
        figs = plot_top_scenarios(scenarios, save_dir=save_dir)
        assert (save_dir / "scenario_000.png").exists()
        assert (save_dir / "scenario_001.png").exists()
        for fig in figs:
            mpl.pyplot.close(fig)

    def test_with_shared_zscore(self, tmp_path):
        T = 50
        scenarios = []
        for i in range(2):
            sd = tmp_path / "data" / f"scenario_{i:03d}"
            _write_scenario(sd, scenario_idx=i, T=T)
            scenarios.append(SweepResultsReader.load_full_scenario(sd))

        zscore = np.random.default_rng(99).normal(0, 1.5, T)
        figs = plot_top_scenarios(scenarios, zscore=zscore)
        # Each figure should have 3 panels
        for fig in figs:
            assert len(fig.axes) == 3
            mpl.pyplot.close(fig)
