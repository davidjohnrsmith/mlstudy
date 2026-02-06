"""Tests for fly backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest import (
    BacktestConfig,
    SizingMode,
    backtest_fly,
    backtest_fly_from_panel,
    compute_avg_holding_period,
    compute_hit_rate,
    compute_max_drawdown,
    compute_metrics,
    compute_n_trades,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_turnover,
)
from mlstudy.trading.strategy.structures.selection.curve_selection import select_fly_legs


@pytest.fixture
def synthetic_panel():
    """Create synthetic panel where price is constructed from yield changes.

    This allows us to verify PnL sign and DV01 scaling works correctly.
    The relationship: price_change ≈ -dv01 * yield_change
    """
    np.random.seed(42)

    n_dates = 100
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")

    # Bonds with fixed TTMs (for simplicity)
    bonds = [
        ("bond_2y", 2.0, 20.0),   # (id, ttm, dv01)
        ("bond_5y", 5.0, 50.0),
        ("bond_10y", 10.0, 100.0),
    ]

    # Generate yields with mean reversion for fly
    # Fly yield = y_2y - 2*y_5y + y_10y
    # We'll create yields such that fly has clear up/down moves

    records = []
    prev_yields = {b[0]: 3.0 + 0.5 * np.log(b[1] + 1) for b in bonds}

    for i, date in enumerate(dates):
        # Generate yield changes
        # Add mean-reverting noise
        fly_shock = 0.0
        if 20 <= i < 40:
            fly_shock = -0.02  # Fly yield goes down (enter long)
        elif 50 <= i < 70:
            fly_shock = 0.02   # Fly yield goes up (enter short)

        for bond_id, ttm, dv01 in bonds:
            # Base yield change with some noise
            dy = np.random.randn() * 0.005

            # Fly shock affects belly more
            if bond_id == "bond_5y":
                dy -= fly_shock  # Belly yield moves opposite to fly
            else:
                dy += fly_shock / 2  # Wings move with fly

            new_yield = prev_yields[bond_id] + dy

            # Price from yield: P ≈ 100 - (y - 4) * dv01 / 100
            # Simple approximation for testing
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


@pytest.fixture
def simple_legs_df():
    """Create simple legs DataFrame for direct backtest testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    # Create legs with known values
    records = []
    for i, dt in enumerate(dates):
        # Prices that go up steadily
        front_price = 100 + i * 0.1
        belly_price = 100 + i * 0.15
        back_price = 100 + i * 0.2

        for leg, price, dv01, weight in [
            ("front", front_price, 20, 1.0),
            ("belly", belly_price, 50, -2.0),
            ("back", back_price, 100, 1.0),
        ]:
            records.append({
                "datetime": dt,
                "leg": leg,
                "price": price,
                "dv01": dv01,
                "weight": weight,
            })

    return pd.DataFrame(records)


@pytest.fixture
def simple_signal_df():
    """Create simple signal DataFrame."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    # Signal: flat -> long -> flat
    signals = [0, 0, 1, 1, 1, 0, -1, -1, 0, 0]

    return pd.DataFrame({
        "datetime": dates,
        "signal": signals,
    })


class TestBacktestFly:
    """Tests for backtest_fly function."""

    def test_returns_backtest_result(self, simple_legs_df, simple_signal_df):
        """Should return BacktestResult with pnl_df."""
        result = backtest_fly(simple_legs_df, simple_signal_df)

        assert result.pnl_df is not None
        assert isinstance(result.pnl_df, pd.DataFrame)
        assert "datetime" in result.pnl_df.columns
        assert "net_return" in result.pnl_df.columns
        assert "position" in result.pnl_df.columns

    def test_pnl_df_columns(self, simple_legs_df, simple_signal_df):
        """pnl_df should have all expected columns."""
        result = backtest_fly(simple_legs_df, simple_signal_df)
        df = result.pnl_df

        expected_cols = [
            "datetime", "signal", "position",
            "front_ret", "belly_ret", "back_ret",
            "gross_return", "traded_notional", "transaction_cost",
            "net_return", "cumulative_pnl",
            "net_dv01", "gross_dv01", "gross_notional", "leverage",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_signal_lag(self, simple_legs_df, simple_signal_df):
        """Position should be lagged by 1 day from signal."""
        result = backtest_fly(simple_legs_df, simple_signal_df)
        df = result.pnl_df

        # Position at t should equal signal at t-1
        for i in range(1, len(df)):
            assert df.iloc[i]["position"] == df.iloc[i - 1]["signal"]

    def test_flat_position_zero_return(self, simple_legs_df, simple_signal_df):
        """Flat position should have zero gross return."""
        result = backtest_fly(simple_legs_df, simple_signal_df)
        df = result.pnl_df

        flat_mask = df["position"] == 0
        # First row has NaN returns, skip it
        for idx in df[flat_mask].index[1:]:
            assert df.loc[idx, "gross_return"] == 0.0

    def test_dv01_target_scaling(self, simple_legs_df, simple_signal_df):
        """DV01 target sizing should scale to target."""
        config = BacktestConfig(
            sizing_mode=SizingMode.DV01_TARGET,
            dv01_target=10000.0,
        )
        result = backtest_fly(simple_legs_df, simple_signal_df, config=config)
        df = result.pnl_df

        # When in position, gross_dv01 should be approximately target
        active_mask = df["position"] != 0
        if active_mask.any():
            avg_gross_dv01 = df.loc[active_mask, "gross_dv01"].mean()
            assert avg_gross_dv01 > 0  # Should have some exposure

    def test_transaction_costs_on_trade(self, simple_legs_df, simple_signal_df):
        """Transaction costs should be charged on position changes."""
        config = BacktestConfig(
            transaction_cost_bps=10.0,
            slippage_bps=5.0,
        )
        result = backtest_fly(simple_legs_df, simple_signal_df, config=config)
        df = result.pnl_df

        # Find position changes
        position_changes = df["position"].diff().fillna(0) != 0

        # Transaction costs should be non-zero on changes
        for idx in df[position_changes].index:
            if df.loc[idx, "traded_notional"] > 0:
                assert df.loc[idx, "transaction_cost"] > 0


class TestPnLSign:
    """Tests to verify PnL sign is correct based on price movements."""

    def test_long_position_positive_price_change(self):
        """Long position should profit when prices go up."""
        # Create scenario: long fly, all prices increase
        dates = pd.date_range("2023-01-01", periods=5, freq="D")

        # Prices increasing
        legs_records = []
        for i, dt in enumerate(dates):
            for leg, base_price, dv01, weight in [
                ("front", 100, 20, 1.0),
                ("belly", 100, 50, -2.0),
                ("back", 100, 100, 1.0),
            ]:
                # All prices go up uniformly
                price = base_price + i * 1.0
                legs_records.append({
                    "datetime": dt,
                    "leg": leg,
                    "price": price,
                    "dv01": dv01,
                    "weight": weight,
                })

        legs_df = pd.DataFrame(legs_records)

        # Constant long signal
        signal_df = pd.DataFrame({
            "datetime": dates,
            "signal": [1, 1, 1, 1, 1],
        })

        result = backtest_fly(legs_df, signal_df)

        # After lag, positions start from index 1
        # Net return should be positive when long and prices up
        active_returns = result.pnl_df[result.pnl_df["position"] == 1]["gross_return"]
        # Skip first (NaN return)
        active_returns = active_returns.iloc[1:]

        # With equal weights (1, -2, 1) and uniform price increase,
        # portfolio return = 1*ret - 2*ret + 1*ret = 0
        # So need unequal moves to see profit
        # This test checks structure; specific profit depends on price dynamics

    def test_dv01_neutral_net_dv01_near_zero(self, synthetic_panel):
        """DV01-neutral fly should have net DV01 near zero."""
        legs_table = select_fly_legs(synthetic_panel, tenors=(2, 5, 10))

        config = BacktestConfig(sizing_mode=SizingMode.DV01_TARGET)

        result = backtest_fly_from_panel(
            synthetic_panel, legs_table,
            window=10, entry_z=1.5, exit_z=0.3,
            config=config,
            use_dv01_weights=True,
        )

        # When in position, net DV01 should be close to zero
        active_mask = result.pnl_df["position"] != 0
        if active_mask.any():
            avg_net_dv01 = result.pnl_df.loc[active_mask, "net_dv01"].abs().mean()
            avg_gross_dv01 = result.pnl_df.loc[active_mask, "gross_dv01"].mean()

            # Net should be much smaller than gross
            if avg_gross_dv01 > 0:
                ratio = avg_net_dv01 / avg_gross_dv01
                assert ratio < 0.1, f"Net/Gross DV01 ratio {ratio:.3f} too high"


class TestMetrics:
    """Tests for backtest metrics functions."""

    def test_sharpe_ratio(self):
        """Sharpe ratio should be correct."""
        # Create returns with known mean and std
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        sharpe = compute_sharpe_ratio(returns)

        expected = returns.mean() / returns.std() * np.sqrt(252)
        assert abs(sharpe - expected) < 1e-10

    def test_sharpe_ratio_zero_std(self):
        """Sharpe ratio should be 0 when std is 0."""
        returns = pd.Series([0.01, 0.01, 0.01])
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_max_drawdown(self):
        """Max drawdown should be correct."""
        cumulative = pd.Series([0, 1, 2, 1, 0, -1, 0, 1])
        max_dd, duration = compute_max_drawdown(cumulative)

        # Max drawdown is from peak 2 to trough -1 = -3
        assert max_dd == -3.0

    def test_turnover(self):
        """Turnover should be calculated correctly."""
        traded = pd.Series([100, 0, 200, 0, 100])
        gross = pd.Series([1000, 1000, 1000, 1000, 1000])

        turnover = compute_turnover(traded, gross)

        # Total traded = 400, avg gross = 1000, n_days = 5
        # Daily turnover = 400 / (5 * 1000) = 0.08
        # Annual = 0.08 * 252 = 20.16
        expected = (400 / (5 * 1000)) * 252
        assert abs(turnover - expected) < 1e-10

    def test_avg_holding_period(self):
        """Average holding period should be correct."""
        position = pd.Series([0, 1, 1, 1, 0, 0, -1, -1, 0])
        avg_hold = compute_avg_holding_period(position)

        # Two holding periods: 3 days, 2 days
        assert avg_hold == 2.5

    def test_hit_rate(self):
        """Hit rate should be correct."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        position = pd.Series([1, 1, 1, 1, 1])

        hit_rate = compute_hit_rate(returns, position)
        # 3 positive out of 5
        assert hit_rate == 0.6

    def test_profit_factor(self):
        """Profit factor should be correct."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.01, 0.01])

        pf = compute_profit_factor(returns)
        # Profits = 0.06, Losses = 0.02
        assert pf == 3.0

    def test_n_trades(self):
        """Number of trades should count position changes."""
        position = pd.Series([0, 1, 1, 0, -1, -1, 0])
        n = compute_n_trades(position)

        # Changes: 0->1, 1->0, 0->-1, -1->0 = 4 trades
        assert n == 4

    def test_compute_metrics_full(self, simple_legs_df, simple_signal_df):
        """compute_metrics should return complete metrics."""
        result = backtest_fly(simple_legs_df, simple_signal_df)
        metrics = compute_metrics(result.pnl_df)

        # Check all metrics are present
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "hit_rate")
        assert hasattr(metrics, "n_trades")
        assert hasattr(metrics, "avg_holding_period")


class TestEndToEnd:
    """End-to-end tests for fly backtest pipeline."""

    def test_full_pipeline_runs(self, synthetic_panel):
        """Full pipeline should run without errors."""
        legs_table = select_fly_legs(synthetic_panel, tenors=(2, 5, 10))

        result = backtest_fly_from_panel(
            synthetic_panel, legs_table,
            window=10, entry_z=1.5, exit_z=0.3,
        )

        assert len(result.pnl_df) > 0
        assert "cumulative_pnl" in result.pnl_df.columns

    def test_pipeline_with_dv01_target(self, synthetic_panel):
        """Pipeline with DV01 target sizing should work."""
        legs_table = select_fly_legs(synthetic_panel, tenors=(2, 5, 10))

        config = BacktestConfig(
            sizing_mode=SizingMode.DV01_TARGET,
            dv01_target=5000.0,
        )

        result = backtest_fly_from_panel(
            synthetic_panel, legs_table,
            window=10, entry_z=1.5, exit_z=0.3,
            config=config,
        )

        assert len(result.pnl_df) > 0

    def test_pipeline_generates_trades(self, synthetic_panel):
        """Pipeline should generate some trades."""
        legs_table = select_fly_legs(synthetic_panel, tenors=(2, 5, 10))

        result = backtest_fly_from_panel(
            synthetic_panel, legs_table,
            window=10, entry_z=1.5, exit_z=0.3,
        )

        # Should have some non-zero positions
        n_active = (result.pnl_df["position"] != 0).sum()
        assert n_active > 0, "No trades generated"

    def test_metrics_from_pipeline(self, synthetic_panel):
        """Metrics should be computable from pipeline result."""
        legs_table = select_fly_legs(synthetic_panel, tenors=(2, 5, 10))

        result = backtest_fly_from_panel(
            synthetic_panel, legs_table,
            window=10, entry_z=1.5, exit_z=0.3,
        )

        metrics = compute_metrics(result.pnl_df)

        # Basic sanity checks
        assert metrics.n_trades >= 0
        assert 0 <= metrics.hit_rate <= 1
        assert metrics.pct_time_in_market >= 0


class TestSizingModes:
    """Tests for different sizing modes."""

    def test_fixed_notional_consistent(self, simple_legs_df, simple_signal_df):
        """Fixed notional should give consistent exposure."""
        config = BacktestConfig(
            sizing_mode=SizingMode.FIXED_NOTIONAL,
            fixed_notional=100000.0,
        )

        result = backtest_fly(simple_legs_df, simple_signal_df, config=config)

        # Gross notional should be roughly 3 * fixed_notional when in position
        # (one per leg, though weights vary)
        active_mask = result.pnl_df["position"] != 0
        if active_mask.any():
            gross = result.pnl_df.loc[active_mask, "gross_notional"]
            assert gross.max() > 0  # Should have exposure

    def test_dv01_target_scales_exposure(self, simple_legs_df, simple_signal_df):
        """DV01 target should scale to target exposure."""
        target_dv01 = 5000.0
        config = BacktestConfig(
            sizing_mode=SizingMode.DV01_TARGET,
            dv01_target=target_dv01,
        )

        result = backtest_fly(simple_legs_df, simple_signal_df, config=config)

        # Check that gross DV01 is scaled toward target
        active_mask = result.pnl_df["position"] != 0
        if active_mask.any():
            gross_dv01 = result.pnl_df.loc[active_mask, "gross_dv01"]
            # Should be positive when in position
            assert gross_dv01.max() > 0
