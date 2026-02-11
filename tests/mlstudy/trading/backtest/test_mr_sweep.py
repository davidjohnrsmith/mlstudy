"""Tests for the MR backtest parameter sweep module."""

from __future__ import annotations

import numpy as np
import pytest

from mlstudy.trading.backtest.mean_reversion.engine import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.sweep import (
    SweepResult,
    SweepScenario,
    make_scenarios,
    rank_results,
    run_sweep,
    summary_table,
)
from mlstudy.trading.backtest.metrics import BacktestMetrics


# =========================================================================
# Helpers
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


def _make_scripted_inputs(
    T: int = 60,
    *,
    entry_bar: int = 5,
    tp_bar: int = 20,
    sl_bar: int = 40,
    base_book_size: float = 1000.0,
):
    """Minimal scripted dataset (same pattern as test_mr_backtest.py)."""
    N = 3
    ref_idx = 1
    hedge_ratios = np.array([-0.5, 1.0, -0.5], dtype=np.float64)

    dv01_vals = np.array([0.02, 0.045, 0.08], dtype=np.float64)
    dv01 = np.tile(dv01_vals, (T, 1))

    base_px = np.array([99.0, 98.0, 97.0], dtype=np.float64)
    mid_px = np.tile(base_px, (T, 1))
    rng = np.random.default_rng(42)
    mid_px += np.cumsum(rng.normal(0, 0.001, (T, N)), axis=0)

    half_spread = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    level2_off = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    base_sizes = np.full((T, N), base_book_size, dtype=np.float64)
    bid_px, bid_sz, ask_px, ask_sz = _make_book(
        mid_px, half_spread, level2_off, base_sizes
    )

    zscore = np.zeros(T, dtype=np.float64)
    package_yield_bps = np.zeros(T, dtype=np.float64)
    expected_yield_pnl_bps = np.full(T, 5.0, dtype=np.float64)

    # entry signal
    zscore[entry_bar] = 3.0
    zscore[entry_bar + 1 : tp_bar] = 2.5
    package_yield_bps[entry_bar] = 100.0
    package_yield_bps[entry_bar + 1 : tp_bar] = 100.0

    # TP trigger
    zscore[tp_bar] = -1.0
    package_yield_bps[tp_bar] = 94.0

    # re-entry + SL
    re_entry_bar = tp_bar + 4
    zscore[re_entry_bar] = 3.5
    package_yield_bps[re_entry_bar] = 100.0
    zscore[re_entry_bar + 1 : sl_bar] = 2.0
    package_yield_bps[re_entry_bar + 1 : sl_bar] = 100.0
    zscore[sl_bar] = 2.0
    package_yield_bps[sl_bar] = 106.0

    if sl_bar + 1 < T:
        zscore[sl_bar + 1 :] = 0.0
        package_yield_bps[sl_bar + 1 :] = 100.0

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
        ref_idx=ref_idx,
    )


def _base_cfg(ref_idx: int = 1) -> MRBacktestConfig:
    return MRBacktestConfig(
        target_notional_ref=100.0,
        ref_leg_idx=ref_idx,
        entry_z_threshold=2.0,
        take_profit_zscore_soft_threshold=0.5,
        take_profit_yield_change_soft_threshold=1.0,
        take_profit_yield_change_hard_threshold=3.0,
        stop_loss_yield_change_hard_threshold=5.0,
        max_holding_bars=0,
        expected_yield_pnl_bps_multiplier=1.0,
        entry_cost_premium_yield_bps=0.0,
        tp_cost_premium_yield_bps=0.0,
        tp_quarantine_bars=2,
        sl_quarantine_bars=3,
        time_quarantine_bars=0,
        max_levels_to_cross=2,
        size_haircut=1.0,
        validate_scope="ALL_LEGS",
        initial_capital=0.0,
        use_jit=False,
    )


def _market_data():
    """Return market data dict suitable for ``run_sweep``."""
    d = _make_scripted_inputs()
    return {k: v for k, v in d.items() if k != "ref_idx"}


# =========================================================================
# Tests
# =========================================================================


class TestMakeScenariosSingleParam:
    def test_count(self):
        base = _base_cfg()
        vals = [1.5, 2.0, 2.5, 3.0]
        scenarios = make_scenarios(base, {"entry_z_threshold": vals})
        assert len(scenarios) == len(vals)

    def test_names_and_tags(self):
        base = _base_cfg()
        vals = [1.5, 2.0]
        scenarios = make_scenarios(base, {"entry_z_threshold": vals})
        for sc, v in zip(scenarios, vals):
            assert sc.tags == {"entry_z_threshold": v}
            assert f"entry_z_threshold={v}" in sc.name

    def test_cfg_values(self):
        base = _base_cfg()
        vals = [1.5, 3.0]
        scenarios = make_scenarios(base, {"entry_z_threshold": vals})
        assert scenarios[0].cfg.entry_z_threshold == 1.5
        assert scenarios[1].cfg.entry_z_threshold == 3.0
        # Other fields unchanged
        assert scenarios[0].cfg.target_notional_ref == base.target_notional_ref


class TestMakeScenariosGrid:
    def test_cartesian_product_count(self):
        base = _base_cfg()
        grid = {
            "entry_z_threshold": [1.5, 2.0, 2.5],
            "stop_loss_yield_change_hard_threshold": [3.0, 5.0],
        }
        scenarios = make_scenarios(base, grid)
        assert len(scenarios) == 3 * 2

    def test_all_combos_present(self):
        base = _base_cfg()
        grid = {
            "entry_z_threshold": [1.5, 2.0],
            "tp_quarantine_bars": [0, 2],
        }
        scenarios = make_scenarios(base, grid)
        tag_combos = {(sc.tags["entry_z_threshold"], sc.tags["tp_quarantine_bars"]) for sc in scenarios}
        assert tag_combos == {(1.5, 0), (1.5, 2), (2.0, 0), (2.0, 2)}


class TestRunSweep:
    def test_returns_sweep_results(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 3.0]})
        results = run_sweep(scenarios, **_market_data())

        assert len(results) == 3
        for sr in results:
            assert isinstance(sr, SweepResult)
            assert isinstance(sr.scenario, SweepScenario)
            assert isinstance(sr.metrics, BacktestMetrics)
            assert sr.results is not None

    def test_parallel_matches_sequential(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.5]})
        md = _market_data()
        seq = run_sweep(scenarios, **md, parallel=False)
        par = run_sweep(scenarios, **md, parallel=True)

        assert len(seq) == len(par)
        for s, p in zip(seq, par):
            assert s.metrics.total_pnl == pytest.approx(p.metrics.total_pnl)
            assert s.metrics.sharpe_ratio == pytest.approx(p.metrics.sharpe_ratio)


class TestRankResults:
    def _make_results(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base,
            {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]},
        )
        return run_sweep(scenarios, **_market_data())

    def test_top_n(self):
        results = self._make_results()
        top3 = rank_results(results, metric="sharpe_ratio", top_n=3)
        assert len(top3) == 3

    def test_descending_order(self):
        results = self._make_results()
        ranked = rank_results(results, metric="total_pnl", top_n=5, ascending=False)
        pnls = [r.metrics.total_pnl for r in ranked]
        assert pnls == sorted(pnls, reverse=True)

    def test_ascending_order(self):
        results = self._make_results()
        ranked = rank_results(results, metric="max_drawdown", top_n=5, ascending=True)
        dds = [r.metrics.max_drawdown for r in ranked]
        assert dds == sorted(dds)

    def test_top_n_exceeds_length(self):
        results = self._make_results()
        ranked = rank_results(results, metric="sharpe_ratio", top_n=100)
        assert len(ranked) == len(results)

    def test_invalid_metric_raises(self):
        results = self._make_results()
        with pytest.raises(ValueError, match="Unknown metric"):
            rank_results(results, metric="nonexistent_field")


class TestSummaryTable:
    def test_columns(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0]})
        results = run_sweep(scenarios, **_market_data())
        df = summary_table(results)

        assert len(df) == 2
        assert "name" in df.columns
        assert "entry_z_threshold" in df.columns
        assert "sharpe_ratio" in df.columns
        assert "total_pnl" in df.columns
        assert "max_drawdown" in df.columns

    def test_tag_values_match(self):
        base = _base_cfg()
        vals = [1.5, 2.0]
        scenarios = make_scenarios(base, {"entry_z_threshold": vals})
        results = run_sweep(scenarios, **_market_data())
        df = summary_table(results)

        assert list(df["entry_z_threshold"]) == vals


class TestDifferentConfigsProduceDifferentResults:
    def test_varying_entry_threshold_gives_different_sharpe(self):
        base = _base_cfg()
        # Use thresholds that span across the signal level (3.0) —
        # low threshold enters, very high threshold never enters.
        scenarios = make_scenarios(
            base,
            {"entry_z_threshold": [1.0, 10.0]},
        )
        results = run_sweep(scenarios, **_market_data())
        sharpes = [r.metrics.sharpe_ratio for r in results]

        # threshold=10.0 should never enter (zscore peaks at 3.5),
        # so sharpe should be 0 vs. a non-zero sharpe for threshold=1.0.
        assert sharpes[0] != sharpes[1]
