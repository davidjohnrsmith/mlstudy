"""Tests for MR backtest post-analysis and reporting.

Uses a synthetic MRBacktestResults to test analysis functions
without running the full backtester.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlstudy.trading.backtest.mean_reversion.single_backtest.results import MRBacktestResults
from mlstudy.trading.backtest.mean_reversion.single_backtest.state import (
    ActionCode, State, TradeType,
)
from mlstudy.trading.backtest.mean_reversion.analysis import (
    compute_code_distribution,
    compute_execution_quality,
    compute_exit_type_stats,
    compute_performance_metrics,
    compute_round_trips,
    compute_state_distribution,
    print_summary,
    to_dataframe,
)


# ---------------------------------------------------------------------------
# Fixture: synthetic MRBacktestResults
# ---------------------------------------------------------------------------


def _make_synthetic_results() -> MRBacktestResults:
    """Build a small, deterministic MRBacktestResults.

    Timeline (T=30, N=3):
      bars 0-4:   flat, ActionCode.NO_ACTION
      bar  5:     ActionCode.ENTRY_OK (long entry)
      bars 6-14:  in position (long), ActionCode.NO_ACTION_NO_SIGNAL
      bar  15:    ActionCode.EXIT_TP_OK
      bars 16-19: flat, ActionCode.NO_ACTION
      bar  20:    ActionCode.ENTRY_OK (long entry)
      bars 21-24: in position (long), ActionCode.NO_ACTION_NO_SIGNAL
      bar  25:    ActionCode.EXIT_SL_OK
      bars 26-29: flat, ActionCode.NO_ACTION
    """
    T, N = 30, 3
    rng = np.random.default_rng(123)

    # Per-bar arrays
    state = np.zeros(T, dtype=np.int32)
    codes = np.full(T, ActionCode.NO_ACTION, dtype=np.int32)
    holding = np.zeros(T, dtype=np.int32)
    positions = np.zeros((T, N), dtype=np.float64)

    # Trade 1: entry at bar 5, exit TP at bar 15
    state[5:16] = State.STATE_LONG
    codes[5] = ActionCode.ENTRY_OK
    codes[6:15] = ActionCode.NO_ACTION_NO_SIGNAL
    codes[15] = ActionCode.EXIT_TP_OK
    holding[5] = 1
    for t in range(6, 16):
        holding[t] = t - 5 + 1
    positions[5:16] = [10.0, -5.0, -5.0]

    # Trade 2: entry at bar 20, exit SL at bar 25
    state[20:26] = State.STATE_LONG
    codes[20] = ActionCode.ENTRY_OK
    codes[21:25] = ActionCode.NO_ACTION_NO_SIGNAL
    codes[25] = ActionCode.EXIT_SL_OK
    holding[20] = 1
    for t in range(21, 26):
        holding[t] = t - 20 + 1
    positions[20:26] = [10.0, -5.0, -5.0]

    # Equity: starts at 1000, drifts
    equity = np.full(T, 1000.0, dtype=np.float64)
    # Add some PnL in position periods
    pnl = np.zeros(T, dtype=np.float64)
    # Trade 1 wins: +0.5 per bar in position
    pnl[5:16] = 0.5
    # Trade 2 loses: -0.3 per bar in position
    pnl[20:26] = -0.3
    equity = 1000.0 + np.cumsum(pnl)
    cash = equity - np.sum(positions * 98.0, axis=1)  # rough cash

    # Per-trade arrays (4 trades: entry, exit_tp, entry, exit_sl)
    n_trades = 4
    tr_bar = np.array([5, 15, 20, 25], dtype=np.int64)
    tr_type = np.array([TradeType.TRADE_ENTRY, TradeType.TRADE_EXIT_TP, TradeType.TRADE_ENTRY, TradeType.TRADE_EXIT_SL], dtype=np.int32)
    tr_side = np.array([1, 1, 1, 1], dtype=np.int32)
    tr_sizes = np.array(
        [
            [10.0, -5.0, -5.0],
            [-10.0, 5.0, 5.0],
            [10.0, -5.0, -5.0],
            [-10.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )
    mids = np.array(
        [
            [99.0, 98.0, 97.0],
            [99.5, 97.5, 96.5],
            [99.0, 98.0, 97.0],
            [98.5, 98.5, 97.5],
        ],
        dtype=np.float64,
    )
    # VWAPs: slightly worse than mids (slippage)
    vwaps = mids.copy()
    vwaps[0] += np.array([0.01, -0.01, -0.01])  # buy higher, sell lower
    vwaps[1] += np.array([-0.01, 0.01, 0.01])
    vwaps[2] += np.array([0.01, -0.01, -0.01])
    vwaps[3] += np.array([-0.01, 0.01, 0.01])

    tr_cost = np.array([0.05, 0.04, 0.05, 0.06], dtype=np.float64)
    tr_code = np.array([ActionCode.ENTRY_OK, ActionCode.EXIT_TP_OK, ActionCode.ENTRY_OK, ActionCode.EXIT_SL_OK], dtype=np.int32)
    tr_pkg_yield = np.array([100.0, 94.0, 100.0, 106.0], dtype=np.float64)

    # gross_pnl: same as pnl but with execution costs added back on trade bars
    gross_pnl = pnl.copy()
    gross_pnl[5] += tr_cost[0]    # entry cost bar 5
    gross_pnl[15] += tr_cost[1]   # exit cost bar 15
    gross_pnl[20] += tr_cost[2]   # entry cost bar 20
    gross_pnl[25] += tr_cost[3]   # exit cost bar 25

    return MRBacktestResults(
        positions=positions,
        cash=cash,
        equity=equity,
        pnl=pnl,
        gross_pnl=gross_pnl,
        codes=codes,
        state=state,
        holding=holding,
        tr_bar=tr_bar,
        tr_type=tr_type,
        tr_side=tr_side,
        tr_sizes=tr_sizes,
        tr_risks=tr_sizes.copy(),  # synthetic risk = sizes (no dv01 scaling)
        tr_vwaps=vwaps,
        tr_mids=mids,
        tr_cost=tr_cost,
        tr_code=tr_code,
        tr_pkg_yield=tr_pkg_yield,
        n_trades=n_trades,
    )


@pytest.fixture
def res() -> MRBacktestResults:
    return _make_synthetic_results()


@pytest.fixture
def round_trips(res):
    return compute_round_trips(res)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToDataframe:
    def test_shape(self, res):
        df = to_dataframe(res)
        assert len(df) == 30
        expected_cols = {"equity", "pnl", "gross_pnl", "cumulative_pnl", "state", "code", "holding"}
        assert expected_cols.issubset(set(df.columns))

    def test_with_datetimes(self, res):
        import pandas as pd

        dts = pd.date_range("2024-01-01", periods=30, freq="h")
        df = to_dataframe(res, datetimes=dts)
        assert "datetime" in df.columns
        assert len(df) == 30


class TestComputePerformanceMetrics:
    def test_returns_backtest_metrics(self, res):
        from mlstudy.trading.backtest.metrics.metrics import BacktestMetrics

        metrics = compute_performance_metrics(res)
        assert isinstance(metrics, BacktestMetrics)

    def test_sharpe_is_finite(self, res):
        metrics = compute_performance_metrics(res)
        assert np.isfinite(metrics.sharpe_ratio)

    def test_total_pnl_matches(self, res):
        metrics = compute_performance_metrics(res)
        assert abs(metrics.total_pnl - float(np.sum(res.pnl))) < 1e-8


class TestRoundTrips:
    def test_count(self, round_trips):
        assert len(round_trips) == 2  # two round-trips

    def test_entry_exit_pairing(self, round_trips):
        rt = round_trips
        # First trip: bar 5 → 15
        assert rt.iloc[0]["entry_bar"] == 5
        assert rt.iloc[0]["exit_bar"] == 15
        assert rt.iloc[0]["exit_type"] == "tp"
        # Second trip: bar 20 → 25
        assert rt.iloc[1]["entry_bar"] == 20
        assert rt.iloc[1]["exit_bar"] == 25
        assert rt.iloc[1]["exit_type"] == "sl"

    def test_pnl_sign(self, round_trips):
        # First trade (TP) should be profitable, second (SL) should lose
        assert round_trips.iloc[0]["pnl"] > 0
        assert round_trips.iloc[1]["pnl"] < 0

    def test_holding_bars(self, round_trips):
        assert round_trips.iloc[0]["holding_bars"] == 10  # 15 - 5
        assert round_trips.iloc[1]["holding_bars"] == 5  # 25 - 20

    def test_costs(self, round_trips):
        rt = round_trips
        assert abs(rt.iloc[0]["entry_cost"] - 0.05) < 1e-8
        assert abs(rt.iloc[0]["exit_cost"] - 0.04) < 1e-8
        assert abs(rt.iloc[0]["total_cost"] - 0.09) < 1e-8


class TestCodeDistribution:
    def test_sums_to_T(self, res):
        dist = compute_code_distribution(res)
        assert sum(dist.values()) == 30  # T=30

    def test_contains_expected_codes(self, res):
        dist = compute_code_distribution(res)
        assert "ENTRY_OK" in dist
        assert "EXIT_TP_OK" in dist
        assert "NO_ACTION" in dist


class TestStateDistribution:
    def test_sums_to_one(self, res):
        dist = compute_state_distribution(res)
        assert abs(sum(dist.values()) - 1.0) < 1e-10

    def test_contains_all_states(self, res):
        dist = compute_state_distribution(res)
        assert "flat" in dist
        assert "long" in dist
        assert "short" in dist


class TestExitTypeStats:
    def test_breakdown(self, round_trips):
        stats = compute_exit_type_stats(round_trips)
        assert "tp" in stats.index
        assert "sl" in stats.index
        assert stats.loc["tp", "count"] == 1
        assert stats.loc["sl", "count"] == 1

    def test_columns(self, round_trips):
        stats = compute_exit_type_stats(round_trips)
        expected = {"count", "win_rate", "mean_pnl", "mean_holding_bars", "mean_cost"}
        assert expected.issubset(set(stats.columns))


class TestPrintSummary:
    def test_runs_without_error(self, res, capsys):
        print_summary(res)
        captured = capsys.readouterr()
        assert "MR Backtest Summary" in captured.out
        assert "Sharpe" in captured.out


class TestExecutionQuality:
    def test_returns_dataframe(self, res):
        eq = compute_execution_quality(res)
        assert not eq.empty
        assert "mean" in eq.columns


class TestPlots:
    @pytest.fixture(autouse=True)
    def _check_matplotlib(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_equity_curve(self, res):
        from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import plot_equity_curve
        import matplotlib.figure

        fig = plot_equity_curve(res)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_state_and_codes(self, res):
        from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import plot_state_and_codes
        import matplotlib.figure

        fig = plot_state_and_codes(res)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_trade_pnl(self, round_trips):
        from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import plot_trade_pnl
        import matplotlib.figure

        fig = plot_trade_pnl(round_trips)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_exit_breakdown(self, round_trips):
        from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import plot_exit_breakdown
        import matplotlib.figure

        fig = plot_exit_breakdown(round_trips)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_holding_distribution(self, round_trips):
        from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import plot_holding_distribution
        import matplotlib.figure

        fig = plot_holding_distribution(round_trips)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_slippage(self, res):
        from mlstudy.trading.backtest.mean_reversion.single_backtest.plots import plot_slippage
        import matplotlib.figure

        fig = plot_slippage(res)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# Metric enum & MetricsCalculator tests
# ---------------------------------------------------------------------------

import dataclasses
import pandas as pd

from mlstudy.trading.backtest.metrics.metrics import (
    BacktestMetrics,
    MetricCategory,
    compute_metrics,
)
from mlstudy.trading.backtest.metrics.metrics_calculator import MetricsCalculator
from mlstudy.trading.backtest.metrics.metrics_enum import Metric


def _make_bar_and_trade_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build small bar_df and trade_df suitable for MetricsCalculator."""
    rng = np.random.default_rng(99)
    n = 60
    pnl = rng.normal(0.001, 0.01, n)
    state = np.zeros(n, dtype=float)
    state[5:20] = 1.0
    state[30:45] = -1.0
    cumulative_pnl = np.cumsum(pnl)
    position_0 = state * 10.0

    bar_df = pd.DataFrame({
        "pnl": pnl,
        "cumulative_pnl": cumulative_pnl,
        "state": state,
        "position_0": position_0,
    })

    trade_df = pd.DataFrame({
        "entry_bar": [5, 30],
        "exit_bar": [19, 44],
        "side": [1, -1],
        "holding_bars": [14, 14],
        "exit_type": ["tp", "sl"],
        "total_cost": [0.05, 0.05],
        "pnl": [float(np.sum(pnl[5:20])), float(np.sum(pnl[30:45]))],
    })

    return bar_df, trade_df


class TestMetricEnum:
    def test_metric_enum_has_all_fields(self):
        """Every BacktestMetrics field has a corresponding Metric member."""
        field_names = {f.name for f in dataclasses.fields(BacktestMetrics)}
        enum_keys = {m.key for m in Metric}
        assert field_names == enum_keys

    def test_metric_categories(self):
        equity_names = {
            "total_pnl", "mean_daily_return", "std_daily_return",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "max_drawdown_duration", "calmar_ratio", "skewness",
            "kurtosis", "var_95", "cvar_95",
        }
        trade_names = {
            "turnover_annual", "avg_holding_period", "hit_rate",
            "profit_factor", "avg_win", "avg_loss", "win_loss_ratio",
            "n_trades", "pct_time_in_market",
        }
        for m in Metric:
            if m.key in equity_names:
                assert m.category == MetricCategory.EQUITY, m.key
            elif m.key in trade_names:
                assert m.category == MetricCategory.TRADE, m.key
            else:
                raise AssertionError(f"Unexpected metric {m.key}")

    def test_metric_from_name(self):
        m = Metric.from_key("sharpe_ratio")
        assert m is Metric.SHARPE_RATIO
        assert m.direction == +1

    def test_metric_from_name_unknown(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            Metric.from_key("nonexistent_metric")


class TestMetricsCalculator:
    def test_calculator_compute_all(self):
        """MetricsCalculator.compute_all() matches compute_metrics()."""
        bar_df, trade_df = _make_bar_and_trade_dfs()
        old = compute_metrics(bar_df, trade_df)
        new = MetricsCalculator(bar_df, trade_df).compute_all()
        for field in dataclasses.fields(BacktestMetrics):
            assert getattr(old, field.name) == pytest.approx(
                getattr(new, field.name)
            ), field.name

    def test_calculator_compute_equity(self):
        bar_df, trade_df = _make_bar_and_trade_dfs()
        result = MetricsCalculator(bar_df, trade_df).compute_equity()
        # Equity fields should be non-default
        assert result.sharpe_ratio != 0.0 or result.total_pnl != 0.0
        # Trade fields should be zero
        assert result.n_trades == 0
        assert result.turnover_annual == 0.0
        assert result.avg_holding_period == 0.0
        assert result.hit_rate == 0.0
        assert result.pct_time_in_market == 0.0

    def test_calculator_compute_trades(self):
        bar_df, trade_df = _make_bar_and_trade_dfs()
        result = MetricsCalculator(bar_df, trade_df).compute_trades()
        # Trade fields should be non-default
        assert result.n_trades > 0
        assert result.pct_time_in_market > 0.0
        # Equity fields should be zero
        assert result.total_pnl == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.max_drawdown_duration == 0

    def test_backward_compat_compute_metrics(self):
        """compute_metrics() works as a thin wrapper."""
        bar_df, trade_df = _make_bar_and_trade_dfs()
        result = compute_metrics(bar_df, trade_df)
        assert isinstance(result, BacktestMetrics)
        assert result.n_trades > 0
        assert result.total_pnl != 0.0
