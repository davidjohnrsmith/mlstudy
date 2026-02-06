"""Tests for regime diagnostics in backtest reports."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest.regime_diagnostics import (
    RegimeDiagnostics,
    compute_regime_diagnostics,
    plot_fly_with_regime,
    plot_zscore_positions_stops,
    print_regime_summary,
    save_regime_plots,
)
from mlstudy.trading.strategy.regime.regime import Regime
from mlstudy.trading.strategy.alpha.mean_reversion.signals import ExitReason


@pytest.fixture
def sample_pnl_df():
    """Create sample P&L DataFrame."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    np.random.seed(42)

    return pd.DataFrame({
        "datetime": dates,
        "signal": [int(x) for x in np.random.choice([-1, 0, 1], size=n)],
        "position": [int(x) for x in np.random.choice([-1, 0, 1], size=n)],
        "net_return": np.random.randn(n) * 0.01,
    })


@pytest.fixture
def sample_regime():
    """Create sample regime series."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Mix of regimes
    regimes = (
        [Regime.MEAN_REVERT] * 40 +
        [Regime.TREND] * 30 +
        [Regime.UNCERTAIN] * 30
    )
    return pd.Series(regimes, index=dates)


@pytest.fixture
def sample_exit_reasons():
    """Create sample exit reasons series."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    # Mostly no exit, some exits
    reasons = [ExitReason.NONE] * n
    reasons[20] = ExitReason.NORMAL_EXIT
    reasons[40] = ExitReason.STOP_LOSS
    reasons[60] = ExitReason.MAX_HOLD
    reasons[80] = ExitReason.NORMAL_EXIT

    return pd.Series(reasons, index=dates)


@pytest.fixture
def sample_fly_yield():
    """Create sample fly yield series."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    np.random.seed(42)

    # Mean-reverting process
    yields = [0.0]
    for _ in range(1, n):
        dy = 0.1 * (0 - yields[-1]) + 0.3 * np.random.randn()
        yields.append(yields[-1] + dy)

    return pd.Series(yields, index=dates, name="fly_yield")


@pytest.fixture
def sample_zscore():
    """Create sample z-score series."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    np.random.seed(42)

    return pd.Series(np.random.randn(n) * 2, index=dates, name="zscore")


class TestComputeRegimeDiagnostics:
    """Tests for compute_regime_diagnostics."""

    def test_returns_diagnostics(self, sample_pnl_df, sample_regime):
        """Should return RegimeDiagnostics object."""
        diag = compute_regime_diagnostics(sample_pnl_df, sample_regime)

        assert isinstance(diag, RegimeDiagnostics)

    def test_time_percentages_sum_to_one(self, sample_pnl_df, sample_regime):
        """Regime percentages should sum to 1."""
        diag = compute_regime_diagnostics(sample_pnl_df, sample_regime)

        total = diag.pct_mean_revert + diag.pct_trend + diag.pct_uncertain
        assert abs(total - 1.0) < 1e-10

    def test_expected_regime_percentages(self, sample_pnl_df, sample_regime):
        """Should compute correct regime percentages."""
        diag = compute_regime_diagnostics(sample_pnl_df, sample_regime)

        # Based on fixture: 40% MR, 30% trend, 30% uncertain
        assert abs(diag.pct_mean_revert - 0.40) < 0.01
        assert abs(diag.pct_trend - 0.30) < 0.01
        assert abs(diag.pct_uncertain - 0.30) < 0.01

    def test_counts_exit_types(self, sample_pnl_df, sample_regime, sample_exit_reasons):
        """Should count different exit types."""
        diag = compute_regime_diagnostics(
            sample_pnl_df, sample_regime, exit_reasons=sample_exit_reasons
        )

        # Based on fixture: 2 normal, 1 stop, 1 max_hold
        assert diag.n_normal_exits == 2
        assert diag.n_stops == 1
        assert diag.n_max_hold_exits == 1

    def test_to_dict(self, sample_pnl_df, sample_regime):
        """Should convert to dictionary."""
        diag = compute_regime_diagnostics(sample_pnl_df, sample_regime)
        d = diag.to_dict()

        assert isinstance(d, dict)
        assert "pct_mean_revert" in d
        assert "pnl_mean_revert" in d
        assert "sharpe_mean_revert" in d

    def test_handles_no_exit_reasons(self, sample_pnl_df, sample_regime):
        """Should work without exit_reasons."""
        diag = compute_regime_diagnostics(sample_pnl_df, sample_regime)

        assert diag.n_stops == 0
        assert diag.n_max_hold_exits == 0
        assert diag.n_normal_exits == 0


class TestPlotFlyWithRegime:
    """Tests for plot_fly_with_regime."""

    def test_returns_figure(self, sample_fly_yield, sample_regime):
        """Should return matplotlib figure."""
        pytest.importorskip("matplotlib")

        fig = plot_fly_with_regime(sample_fly_yield, sample_regime)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_provided_axes(self, sample_fly_yield, sample_regime):
        """Should work with provided axes."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fig, ax = plt.subplots()
        result = plot_fly_with_regime(sample_fly_yield, sample_regime, ax=ax)

        assert result is None  # Returns None when ax provided
        plt.close(fig)

    def test_handles_misaligned_indices(self, sample_fly_yield, sample_regime):
        """Should handle series with different indices."""
        pytest.importorskip("matplotlib")

        # Create regime with offset index
        offset_regime = sample_regime.iloc[10:90]

        fig = plot_fly_with_regime(sample_fly_yield, offset_regime)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotZscorePositionsStops:
    """Tests for plot_zscore_positions_stops."""

    def test_returns_figure(self, sample_zscore, sample_pnl_df):
        """Should return matplotlib figure."""
        pytest.importorskip("matplotlib")

        signal = sample_pnl_df.set_index("datetime")["signal"]
        fig = plot_zscore_positions_stops(sample_zscore, signal)

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_exit_reasons(self, sample_zscore, sample_pnl_df, sample_exit_reasons):
        """Should plot stop markers when exit_reasons provided."""
        pytest.importorskip("matplotlib")

        signal = sample_pnl_df.set_index("datetime")["signal"]
        fig = plot_zscore_positions_stops(
            sample_zscore, signal, exit_reasons=sample_exit_reasons
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_thresholds(self, sample_zscore, sample_pnl_df):
        """Should accept custom thresholds."""
        pytest.importorskip("matplotlib")

        signal = sample_pnl_df.set_index("datetime")["signal"]
        fig = plot_zscore_positions_stops(
            sample_zscore, signal,
            entry_z=1.5, exit_z=0.3, stop_z=3.0
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSaveRegimePlots:
    """Tests for save_regime_plots."""

    def test_saves_plots(self, tmp_path, sample_fly_yield, sample_zscore,
                         sample_pnl_df, sample_regime):
        """Should save plot files."""
        pytest.importorskip("matplotlib")

        signal = sample_pnl_df.set_index("datetime")["signal"]

        saved = save_regime_plots(
            output_path=tmp_path,
            fly_yield=sample_fly_yield,
            zscore=sample_zscore,
            signal=signal,
            regime=sample_regime,
        )

        assert "plot_fly_regime" in saved
        assert "plot_zscore_positions" in saved
        assert (tmp_path / "fly_regime_overlay.png").exists()
        assert (tmp_path / "zscore_positions_stops.png").exists()

    def test_with_exit_reasons(self, tmp_path, sample_fly_yield, sample_zscore,
                               sample_pnl_df, sample_regime, sample_exit_reasons):
        """Should work with exit_reasons."""
        pytest.importorskip("matplotlib")

        signal = sample_pnl_df.set_index("datetime")["signal"]

        saved = save_regime_plots(
            output_path=tmp_path,
            fly_yield=sample_fly_yield,
            zscore=sample_zscore,
            signal=signal,
            regime=sample_regime,
            exit_reasons=sample_exit_reasons,
        )

        assert len(saved) == 2


class TestPrintRegimeSummary:
    """Tests for print_regime_summary."""

    def test_prints_without_error(self, sample_pnl_df, sample_regime, capsys):
        """Should print formatted output."""
        diag = compute_regime_diagnostics(sample_pnl_df, sample_regime)
        print_regime_summary(diag)

        captured = capsys.readouterr()
        assert "REGIME DIAGNOSTICS" in captured.out
        assert "Time in Regime" in captured.out
        assert "P&L by Regime" in captured.out
        assert "Sharpe by Regime" in captured.out


class TestRegimeDiagnosticsIntegration:
    """Integration tests for regime diagnostics with report generation."""

    def test_generate_report_with_regime_data(
        self, tmp_path, sample_fly_yield, sample_zscore,
        sample_pnl_df, sample_regime
    ):
        """Should generate report with regime data."""
        pytest.importorskip("matplotlib")

        from mlstudy.trading.backtest.engine import BacktestConfig, BacktestResult
        from mlstudy.trading.backtest.report import RegimeData, generate_report

        # Create BacktestResult
        pnl_df = sample_pnl_df.copy()
        pnl_df["cumulative_pnl"] = pnl_df["net_return"].cumsum()
        result = BacktestResult(pnl_df=pnl_df, config=BacktestConfig())

        # Create RegimeData
        regime_data = RegimeData(
            fly_yield=sample_fly_yield,
            zscore=sample_zscore,
            regime=sample_regime,
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
        )

        # Generate report
        report = generate_report(
            result=result,
            output_dir=tmp_path,
            run_id="test_run",
            regime_data=regime_data,
        )

        # Check regime outputs exist
        assert "regime_diagnostics" in report
        assert "regime_diagnostics_json" in report
        assert "plot_fly_regime" in report
        assert "plot_zscore_positions" in report

        # Check files exist
        run_path = tmp_path / "test_run"
        assert (run_path / "regime_diagnostics.json").exists()
        assert (run_path / "fly_regime_overlay.png").exists()
        assert (run_path / "zscore_positions_stops.png").exists()

    def test_generate_report_without_regime_data(self, tmp_path, sample_pnl_df):
        """Should work without regime data (backward compatible)."""
        pytest.importorskip("matplotlib")

        from mlstudy.trading.backtest.engine import BacktestConfig, BacktestResult
        from mlstudy.trading.backtest.report import generate_report

        pnl_df = sample_pnl_df.copy()
        pnl_df["cumulative_pnl"] = pnl_df["net_return"].cumsum()
        result = BacktestResult(pnl_df=pnl_df, config=BacktestConfig())

        report = generate_report(
            result=result,
            output_dir=tmp_path,
            run_id="test_run",
        )

        # Should not have regime outputs
        assert "regime_diagnostics" not in report
        assert "plot_fly_regime" not in report
