"""Tests for the MR backtest parameter sweep module."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from mlstudy.trading.backtest.mean_reversion.engine import MRBacktestConfig
from mlstudy.trading.backtest.mean_reversion.sweep import (
    MetricsOnlyResult,
    SweepError,
    SweepResult,
    SweepScenario,
    SweepSummary,
    make_scenarios,
    rank_results,
    run_sweep,
    summary_table,
)
from mlstudy.trading.backtest.mean_reversion.sweep_rank import (
    MetricPreferenceRegistry,
    ParameterPreferenceRegistry,
    RankingPlan,
    rank_scenarios,
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
# Original tests
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


# =========================================================================
# New tests
# =========================================================================


class TestBackendConsistency:
    """serial / thread / process backends must produce identical metrics."""

    def test_serial_thread_same_metrics(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 3.0]})
        md = _market_data()

        serial = run_sweep(scenarios, **md, backend="serial")
        thread = run_sweep(scenarios, **md, backend="thread")

        assert len(serial) == len(thread)
        for s, t in zip(serial, thread):
            assert s.metrics.total_pnl == pytest.approx(t.metrics.total_pnl)
            assert s.metrics.sharpe_ratio == pytest.approx(t.metrics.sharpe_ratio)
            assert s.metrics.max_drawdown == pytest.approx(t.metrics.max_drawdown)

    def test_serial_process_same_metrics(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 3.0]})
        md = _market_data()

        serial = run_sweep(scenarios, **md, backend="serial")
        process = run_sweep(scenarios, **md, backend="process", n_workers=2)

        assert len(serial) == len(process)
        for s, p in zip(serial, process):
            assert s.metrics.total_pnl == pytest.approx(p.metrics.total_pnl)
            assert s.metrics.sharpe_ratio == pytest.approx(p.metrics.sharpe_ratio)

    def test_all_backends_same_ordering(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        serial = run_sweep(scenarios, **md, backend="serial")
        thread = run_sweep(scenarios, **md, backend="thread", n_workers=2)
        process = run_sweep(scenarios, **md, backend="process", n_workers=2)

        serial_names = [r.scenario.name for r in serial]
        thread_names = [r.scenario.name for r in thread]
        process_names = [r.scenario.name for r in process]

        assert serial_names == thread_names == process_names

    def test_invalid_backend_raises(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [2.0]})
        with pytest.raises(ValueError, match="Unknown backend"):
            run_sweep(scenarios, **_market_data(), backend="gpu")


class TestChunking:
    """Chunking must not change results; ordering must remain stable."""

    def test_chunk_size_1_matches_default(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 2.5]})
        md = _market_data()

        default = run_sweep(scenarios, **md, backend="serial")
        chunked = run_sweep(scenarios, **md, backend="serial", chunk_size=1)

        for d, c in zip(default, chunked):
            assert d.metrics.total_pnl == pytest.approx(c.metrics.total_pnl)

    def test_chunk_size_large_matches_default(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 2.5]})
        md = _market_data()

        default = run_sweep(scenarios, **md, backend="serial")
        chunked = run_sweep(scenarios, **md, backend="serial", chunk_size=1000)

        for d, c in zip(default, chunked):
            assert d.metrics.total_pnl == pytest.approx(c.metrics.total_pnl)

    def test_stable_ordering_with_thread_chunking(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        results = run_sweep(
            scenarios, **md, backend="thread", chunk_size=2, n_workers=3
        )
        names = [r.scenario.name for r in results]
        expected_names = [sc.name for sc in scenarios]
        assert names == expected_names

    def test_scenario_idx_assigned(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 2.5]})
        md = _market_data()

        results = run_sweep(scenarios, **md, backend="serial")
        for i, r in enumerate(results):
            assert r.scenario_idx == i


class TestMetricsOnlyMode:
    """mode="metrics_only" returns MetricsOnlyResult with correct values."""

    def test_returns_metrics_only_result(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 3.0]})
        md = _market_data()

        results = run_sweep(scenarios, **md, mode="metrics_only")

        assert len(results) == 3
        for r in results:
            assert isinstance(r, MetricsOnlyResult)

    def test_fields_exist(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [2.0]})
        md = _market_data()

        results = run_sweep(scenarios, **md, mode="metrics_only")
        r = results[0]

        assert hasattr(r, "scenario_idx")
        assert hasattr(r, "scenario")
        assert hasattr(r, "total_pnl")
        assert hasattr(r, "final_equity")
        assert hasattr(r, "n_trades")
        assert hasattr(r, "max_drawdown")
        assert hasattr(r, "sharpe_ratio")
        assert hasattr(r, "code_counts")
        assert isinstance(r.code_counts, dict)

    def test_values_match_full_mode(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 3.0]})
        md = _market_data()

        full = run_sweep(scenarios, **md, mode="full")
        metrics = run_sweep(scenarios, **md, mode="metrics_only")

        for f, m in zip(full, metrics):
            assert f.metrics.total_pnl == pytest.approx(m.total_pnl)
            assert f.metrics.sharpe_ratio == pytest.approx(m.sharpe_ratio)
            assert f.metrics.max_drawdown == pytest.approx(m.max_drawdown)
            assert f.metrics.n_trades == m.n_trades

    def test_metrics_only_rank(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        results = run_sweep(scenarios, **md, mode="metrics_only")
        ranked = rank_results(results, metric="total_pnl", top_n=3)

        assert len(ranked) == 3
        pnls = [r.total_pnl for r in ranked]
        assert pnls == sorted(pnls, reverse=True)

    def test_metrics_only_summary_table(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0]})
        md = _market_data()

        results = run_sweep(scenarios, **md, mode="metrics_only")
        df = summary_table(results)

        assert len(df) == 2
        assert "total_pnl" in df.columns
        assert "sharpe_ratio" in df.columns
        assert "final_equity" in df.columns

    def test_invalid_mode_raises(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [2.0]})
        with pytest.raises(ValueError, match="Unknown mode"):
            run_sweep(scenarios, **_market_data(), mode="turbo")


class TestTopKRerun:
    """keep_top_k_full rerun returns SweepSummary."""

    def test_returns_sweep_summary(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        result = run_sweep(scenarios, **md, mode="metrics_only", keep_top_k_full=2)

        assert isinstance(result, SweepSummary)
        assert len(result.all_metrics) == 5
        assert len(result.top_full) == 2

    def test_top_full_are_sweep_results(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.0, 2.0, 3.0]})
        md = _market_data()

        result = run_sweep(scenarios, **md, mode="metrics_only", keep_top_k_full=2)

        for sr in result.top_full:
            assert isinstance(sr, SweepResult)
            assert sr.results is not None
            assert isinstance(sr.metrics, BacktestMetrics)

    def test_top_full_ranked_by_pnl(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        result = run_sweep(scenarios, **md, mode="metrics_only", keep_top_k_full=3)

        pnls = [sr.metrics.total_pnl for sr in result.top_full]
        assert pnls == sorted(pnls, reverse=True)

    def test_top_k_exceeds_scenarios(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0]})
        md = _market_data()

        result = run_sweep(scenarios, **md, mode="metrics_only", keep_top_k_full=10)

        assert isinstance(result, SweepSummary)
        assert len(result.top_full) == 2  # only 2 scenarios exist

    def test_all_metrics_original_order(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        result = run_sweep(scenarios, **md, mode="metrics_only", keep_top_k_full=2)

        idxs = [m.scenario_idx for m in result.all_metrics]
        assert idxs == list(range(5))


class TestProcessBackend:
    """Process backend produces correct results."""

    def test_process_matches_serial(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0]})
        md = _market_data()

        serial = run_sweep(scenarios, **md, backend="serial")
        process = run_sweep(scenarios, **md, backend="process", n_workers=2)

        for s, p in zip(serial, process):
            assert s.metrics.total_pnl == pytest.approx(p.metrics.total_pnl)

    def test_process_with_chunking(self):
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        serial = run_sweep(scenarios, **md, backend="serial")
        process = run_sweep(
            scenarios, **md, backend="process", n_workers=2, chunk_size=2
        )

        assert len(serial) == len(process)
        for s, p in zip(serial, process):
            assert s.metrics.total_pnl == pytest.approx(p.metrics.total_pnl)

    def test_process_metrics_only(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0, 3.0]})
        md = _market_data()

        serial = run_sweep(scenarios, **md, mode="metrics_only")
        process = run_sweep(
            scenarios, **md, mode="metrics_only", backend="process", n_workers=2
        )

        for s, p in zip(serial, process):
            assert isinstance(p, MetricsOnlyResult)
            assert s.total_pnl == pytest.approx(p.total_pnl)


class TestPersistence:
    """save_top_full_dir writes expected files."""

    def test_creates_files(self, tmp_path):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.0, 2.0, 3.0]})
        md = _market_data()

        run_sweep(
            scenarios,
            **md,
            mode="metrics_only",
            keep_top_k_full=2,
            save_top_full_dir=tmp_path / "output",
        )

        output_dir = tmp_path / "output"
        assert output_dir.exists()
        assert (output_dir / "scenario_000").exists()
        assert (output_dir / "scenario_001").exists()

    def test_spec_json_content(self, tmp_path):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.0, 2.0]})
        md = _market_data()

        run_sweep(
            scenarios,
            **md,
            mode="metrics_only",
            keep_top_k_full=1,
            save_top_full_dir=tmp_path / "out",
        )

        spec_path = tmp_path / "out" / "scenario_000" / "spec.json"
        assert spec_path.exists()

        spec = json.loads(spec_path.read_text())
        assert "name" in spec
        assert "tags" in spec
        assert "config" in spec
        assert "metrics" in spec
        assert "scenario_idx" in spec

    def test_npy_files_loadable(self, tmp_path):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [2.0]})
        md = _market_data()

        run_sweep(
            scenarios,
            **md,
            mode="metrics_only",
            keep_top_k_full=1,
            save_top_full_dir=tmp_path / "out",
        )

        scenario_dir = tmp_path / "out" / "scenario_000"
        equity = np.load(scenario_dir / "equity.npy")
        assert equity.ndim == 1
        assert len(equity) > 0

        pnl = np.load(scenario_dir / "pnl.npy")
        assert pnl.ndim == 1

    def test_no_persistence_without_dir(self, tmp_path):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.0, 2.0]})
        md = _market_data()

        result = run_sweep(
            scenarios,
            **md,
            mode="metrics_only",
            keep_top_k_full=1,
        )

        assert isinstance(result, SweepSummary)
        # No files created
        assert not (tmp_path / "output").exists()


class TestBackwardCompat:
    """Existing API must keep working."""

    def test_existing_api_unchanged(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [1.5, 2.0]})
        md = _market_data()

        results = run_sweep(scenarios, **md)

        assert len(results) == 2
        for sr in results:
            assert isinstance(sr, SweepResult)

    def test_parallel_true_warns(self):
        base = _base_cfg()
        scenarios = make_scenarios(base, {"entry_z_threshold": [2.0]})
        md = _market_data()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_sweep(scenarios, **md, parallel=True)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_sweep_result_without_idx(self):
        """SweepResult can still be created without scenario_idx."""
        from mlstudy.trading.backtest.mean_reversion.analysis import (
            compute_performance_metrics,
        )
        from mlstudy.trading.backtest.mean_reversion.engine import run_backtest

        base = _base_cfg()
        sc = SweepScenario(name="test", cfg=base, tags={})
        md = _market_data()

        res = run_backtest(cfg=base, **md)
        metrics = compute_performance_metrics(res)

        sr = SweepResult(scenario=sc, results=res, metrics=metrics)
        assert sr.scenario_idx == -1  # default

    def test_empty_scenarios(self):
        md = _market_data()
        results = run_sweep([], **md)
        assert results == []


# =========================================================================
# Ranking tests
# =========================================================================


def _metrics_only_results() -> list[MetricsOnlyResult]:
    """Run a small sweep in metrics_only mode and return results."""
    base = _base_cfg()
    scenarios = make_scenarios(
        base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
    )
    md = _market_data()
    return run_sweep(scenarios, **md, mode="metrics_only")


class TestRanking:
    """Tests for the weighted-rank system."""

    def test_metric_registry_known(self):
        assert MetricPreferenceRegistry.direction("total_pnl") == +1
        assert MetricPreferenceRegistry.direction("sharpe_ratio") == +1
        assert MetricPreferenceRegistry.direction("std_daily_return") == -1
        assert MetricPreferenceRegistry.direction("max_drawdown_duration") == -1

    def test_metric_registry_unknown(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricPreferenceRegistry.direction("bogus_metric")

    def test_param_registry_known(self):
        assert ParameterPreferenceRegistry.direction("target_notional_ref") == +1
        assert ParameterPreferenceRegistry.direction("entry_z_threshold") == +1
        assert ParameterPreferenceRegistry.direction("max_holding_bars") == -1
        assert ParameterPreferenceRegistry.direction("size_haircut") == -1

    def test_param_registry_unknown(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParameterPreferenceRegistry.direction("nonexistent_param")

    def test_default_plan_matches_pnl_order(self):
        results = _metrics_only_results()
        ranked = rank_scenarios(results)

        pnls = [r.total_pnl for r in ranked]
        assert pnls == sorted(pnls, reverse=True)

    def test_custom_plan_sharpe(self):
        results = _metrics_only_results()
        plan = RankingPlan(primary_metrics=(("sharpe_ratio", 1.0),))
        ranked = rank_scenarios(results, plan)

        sharpes = [r.sharpe_ratio for r in ranked]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_multi_metric_plan(self):
        results = _metrics_only_results()
        plan = RankingPlan(
            primary_metrics=(("total_pnl", 0.5), ("sharpe_ratio", 0.5)),
        )
        ranked = rank_scenarios(results, plan)

        # The blended ranking should differ from a pure pnl ranking
        # when pnl and sharpe disagree on ordering.  At minimum, verify
        # it returns all results and the first is "good" by both metrics.
        assert len(ranked) == len(results)
        # Verify determinism: calling again gives same order
        ranked2 = rank_scenarios(results, plan)
        assert [r.scenario_idx for r in ranked] == [r.scenario_idx for r in ranked2]

    def test_param_stage_tiebreak(self):
        """When primary metrics tie, param stage should break the tie."""
        base = _base_cfg()
        # Two scenarios with identical config except entry_z_threshold.
        # With entry_z_threshold=10.0, neither scenario enters a trade,
        # so all metrics are identical — the param stage must break the tie.
        s1 = SweepScenario(
            name="s1",
            cfg=base.__class__(**{**base.__dict__, "entry_z_threshold": 10.0, "size_haircut": 0.5}),
            tags={"size_haircut": 0.5},
        )
        s2 = SweepScenario(
            name="s2",
            cfg=base.__class__(**{**base.__dict__, "entry_z_threshold": 10.0, "size_haircut": 0.9}),
            tags={"size_haircut": 0.9},
        )

        md = _market_data()
        r1 = run_sweep([s1], **md, mode="metrics_only")[0]
        r2 = run_sweep([s2], **md, mode="metrics_only")[0]

        # Rebuild with scenario_idx 0 and 1
        from dataclasses import replace
        r1 = replace(r1, scenario_idx=0, scenario=s1)
        r2 = replace(r2, scenario_idx=1, scenario=s2)

        plan = RankingPlan(
            primary_metrics=(("total_pnl", 1.0),),
            primary_params=(("size_haircut", 1.0),),
        )
        ranked = rank_scenarios([r1, r2], plan)

        # size_haircut direction is -1 (lower preferred), so s1 (0.5) wins
        assert ranked[0].scenario.name == "s1"
        assert ranked[1].scenario.name == "s2"

    def test_single_scenario(self):
        results = _metrics_only_results()[:1]
        ranked = rank_scenarios(results)
        assert len(ranked) == 1
        assert ranked[0].scenario_idx == results[0].scenario_idx

    def test_empty_list(self):
        ranked = rank_scenarios([])
        assert ranked == []

    def test_run_sweep_with_ranking_plan(self):
        """Integration: run_sweep accepts ranking_plan and top_full follows it."""
        base = _base_cfg()
        scenarios = make_scenarios(
            base, {"entry_z_threshold": [1.0, 1.5, 2.0, 2.5, 3.0]}
        )
        md = _market_data()

        plan = RankingPlan(primary_metrics=(("sharpe_ratio", 1.0),))
        result = run_sweep(
            scenarios, **md, mode="metrics_only", keep_top_k_full=3,
            ranking_plan=plan,
        )

        assert isinstance(result, SweepSummary)
        assert len(result.top_full) == 3

        # top_full ordering should follow the sharpe-based ranking plan
        top_idxs = [sr.scenario_idx for sr in result.top_full]

        # Get the sharpe-ranked order from all_metrics
        sharpe_ranked = rank_scenarios(result.all_metrics, plan)
        expected_idxs = [r.scenario_idx for r in sharpe_ranked[:3]]

        assert top_idxs == expected_idxs
