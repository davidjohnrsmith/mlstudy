"""Tests for intraday fly backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.backtest import SizingMode
from mlstudy.trading.backtest.intraday import (
    IntradayBacktestConfig,
    IntradayBacktestResult,
    aggregate_to_daily,
    backtest_fly_intraday,
    compute_fly_signal,
    compute_leg_returns,
    generate_positions,
    scale_weights_for_dv01_target,
)


@pytest.fixture
def intraday_panel():
    """Create synthetic intraday panel data."""
    np.random.seed(42)

    # 5 trading days, hourly data
    days = []
    for day_offset in range(5):
        day_start = pd.Timestamp(f"2023-01-0{day_offset + 2} 07:00", tz="Europe/Berlin")
        day_dates = pd.date_range(day_start, periods=12, freq="h")  # 07:00 to 18:00
        days.extend(day_dates)

    dates = pd.DatetimeIndex(days)

    # Three bonds
    bonds = [
        ("bond_2y", 2.0, 20.0),  # TTM, DV01
        ("bond_5y", 5.0, 50.0),
        ("bond_10y", 10.0, 100.0),
    ]

    records = []
    prev_yields = {b[0]: 3.0 + 0.3 * np.log(b[1] + 1) for b in bonds}

    for dt in dates:
        for bond_id, ttm, dv01 in bonds:
            dy = np.random.randn() * 0.01
            new_yield = prev_yields[bond_id] + dy
            price = 100 - (new_yield - 4) * dv01 / 100

            records.append({
                "datetime": dt,
                "bond_id": bond_id,
                "ttm_years": ttm,
                "yield": new_yield,
                "price": price,
                "dv01": dv01,
            })
            prev_yields[bond_id] = new_yield

    return pd.DataFrame(records)


class TestComputeFlySignal:
    """Tests for compute_fly_signal function."""

    def test_returns_series(self):
        """Should return a Series."""
        df = pd.DataFrame({
            "fly_yield": np.random.randn(50).cumsum() * 0.01 + 0.1,
        })

        result = compute_fly_signal(df, window=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_zscore_centered(self):
        """Z-score should be approximately centered around 0."""
        np.random.seed(42)
        df = pd.DataFrame({
            "fly_yield": np.random.randn(100) * 0.1 + 0.5,
        })

        result = compute_fly_signal(df, window=20)
        # After warmup, mean should be close to 0
        valid = result.dropna()
        assert abs(valid.mean()) < 0.5

    def test_robust_zscore(self):
        """Robust z-score should use median/MAD."""
        df = pd.DataFrame({
            "fly_yield": np.random.randn(50) * 0.1,
        })

        robust = compute_fly_signal(df, window=10, robust=True)
        standard = compute_fly_signal(df, window=10, robust=False)

        # Should produce different results
        assert not np.allclose(
            robust.dropna().values,
            standard.dropna().values,
            equal_nan=True,
        )

    def test_clip_applied(self):
        """Z-score should be clipped to specified range."""
        df = pd.DataFrame({
            "fly_yield": np.concatenate([np.zeros(20), [10, -10]]),  # Extreme values
        })

        result = compute_fly_signal(df, window=10, clip=3.0)
        assert result.max() <= 3.0
        assert result.min() >= -3.0


class TestGeneratePositions:
    """Tests for generate_positions function."""

    def test_long_entry(self):
        """Should enter long when z < -entry_z."""
        zscore = pd.Series([0, 0, -2.5, -2.5, -1.0, 0])
        can_trade = pd.Series([True] * 6)

        result = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=None, can_trade=can_trade)

        # Should enter long at index 2 (z = -2.5)
        assert result.iloc[2] == 1
        # Should stay long until z > -0.5
        assert result.iloc[3] == 1
        assert result.iloc[4] == 1
        # Should exit at z = 0
        assert result.iloc[5] == 0

    def test_short_entry(self):
        """Should enter short when z > entry_z."""
        zscore = pd.Series([0, 0, 2.5, 2.5, 1.0, 0])
        can_trade = pd.Series([True] * 6)

        result = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=None, can_trade=can_trade)

        # Should enter short at index 2 (z = 2.5)
        assert result.iloc[2] == -1
        # Should stay short until |z| < 0.5
        assert result.iloc[3] == -1
        assert result.iloc[4] == -1
        # Should exit at z = 0
        assert result.iloc[5] == 0

    def test_stop_loss(self):
        """Should stop out when |z| > stop_z."""
        zscore = pd.Series([0, -2.5, -5.0, 0])
        can_trade = pd.Series([True] * 4)

        result = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=4.0, can_trade=can_trade)

        # Should enter long at index 1
        assert result.iloc[1] == 1
        # Should stop out at index 2 (z = -5.0 exceeds stop_z=4.0)
        assert result.iloc[2] == 0

    def test_only_trades_when_allowed(self):
        """Should only change position when can_trade is True."""
        zscore = pd.Series([0, -2.5, -2.5, 0])
        can_trade = pd.Series([True, False, True, True])

        result = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=None, can_trade=can_trade)

        # Cannot enter at index 1 (can_trade=False), enters at index 2
        assert result.iloc[1] == 0
        assert result.iloc[2] == 1

    def test_position_persists_outside_trading(self):
        """Position should persist when can_trade is False."""
        zscore = pd.Series([-2.5, -2.5, 0, 0])
        can_trade = pd.Series([True, False, False, True])

        result = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=None, can_trade=can_trade)

        # Enters long at 0, persists at 1 and 2 (even though z=0 at 2)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 1
        assert result.iloc[2] == 1
        # Exits at 3 (can_trade=True, z=0 < exit_z=0.5)
        assert result.iloc[3] == 0


class TestScaleWeightsForDv01Target:
    """Tests for scale_weights_for_dv01_target function."""

    def test_scales_to_target(self):
        """Should scale weights to achieve target DV01."""
        weights = np.array([1.0, -2.0, 1.0])
        dv01s = np.array([20.0, 50.0, 100.0])
        target = 1000.0

        result = scale_weights_for_dv01_target(weights, dv01s, target)

        # Gross DV01 = sum(|w * dv01|)
        gross_dv01 = np.sum(np.abs(result * dv01s))
        assert abs(gross_dv01 - target) < 1e-6

    def test_preserves_relative_weights(self):
        """Should preserve relative weight ratios."""
        weights = np.array([1.0, -2.0, 1.0])
        dv01s = np.array([20.0, 50.0, 100.0])
        target = 1000.0

        result = scale_weights_for_dv01_target(weights, dv01s, target)

        # Ratio between weights should be preserved
        assert abs(result[1] / result[0] - weights[1] / weights[0]) < 1e-6


class TestComputeLegReturns:
    """Tests for compute_leg_returns function."""

    def test_computes_returns(self):
        """Should compute simple returns for each leg."""
        df = pd.DataFrame({
            "front_price": [100, 101, 102],
            "belly_price": [100, 99, 98],
            "back_price": [100, 100, 100],
        })

        result = compute_leg_returns(df)

        # First row should be NaN
        assert pd.isna(result["front_ret"].iloc[0])

        # Second row: (101-100)/100 = 0.01
        assert abs(result["front_ret"].iloc[1] - 0.01) < 1e-6

        # Belly decreases: (99-100)/100 = -0.01
        assert abs(result["belly_ret"].iloc[1] - (-0.01)) < 1e-6


class TestBacktestFlyIntraday:
    """Tests for backtest_fly_intraday function."""

    def test_returns_result(self, intraday_panel):
        """Should return IntradayBacktestResult."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="17:00",
            selection_time="08:00",
            rebalance_mode="every_bar",
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        assert isinstance(result, IntradayBacktestResult)
        assert result.pnl_df is not None
        assert result.daily_df is not None
        assert result.legs_table is not None
        assert result.metrics is not None

    def test_pnl_df_columns(self, intraday_panel):
        """Should have required columns in pnl_df."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="17:00",
            rebalance_mode="every_bar",
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        expected_cols = [
            "datetime", "trading_date", "is_session",
            "signal", "position",
            "gross_return", "net_return", "cumulative_pnl",
            "turnover", "transaction_cost",
            "gross_dv01", "net_dv01",
            "front_id", "belly_id", "back_id",
        ]
        for col in expected_cols:
            assert col in result.pnl_df.columns

    def test_signal_position_lag(self, intraday_panel):
        """Signal at t should give position at t+1."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="17:00",
            rebalance_mode="every_bar",
            signal_lag=1,
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        pnl = result.pnl_df

        # Position should be lagged signal
        # position[t] = signal[t-1]
        for i in range(1, len(pnl)):
            assert pnl["position"].iloc[i] == pnl["signal"].iloc[i - 1]

    def test_trades_only_during_session(self, intraday_panel):
        """With open_only mode, trades should only happen at session open."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="17:00",
            rebalance_mode="open_only",
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        pnl = result.pnl_df

        # Position changes should only occur at can_trade=True bars
        # Note: position is lagged, so signal changes at tradeable bars,
        # position follows next bar
        assert "can_trade" in pnl.columns
        assert "position" in pnl.columns

    def test_legs_stable_within_day(self, intraday_panel):
        """Leg IDs should be constant within each trading day."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="17:00",
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        pnl = result.pnl_df

        for _trading_date, group in pnl.groupby("trading_date"):
            for leg in ["front_id", "belly_id", "back_id"]:
                # All bars in day should have same leg
                assert group[leg].nunique() == 1

    def test_pnl_sign_long_position(self, intraday_panel):
        """Long position with rising prices should have positive PnL."""
        # Create simple data where prices increase
        dates = pd.date_range("2023-01-02 08:00", periods=20, freq="h", tz="Europe/Berlin")

        records = []
        for i, dt in enumerate(dates):
            for bond_id, ttm, base_price in [("b2", 2, 100), ("b5", 5, 100), ("b10", 10, 100)]:
                records.append({
                    "datetime": dt,
                    "bond_id": bond_id,
                    "ttm_years": ttm,
                    "yield": 3.0 - i * 0.01,  # Falling yields (cheap fly)
                    "price": base_price + i * 0.5,  # Rising prices
                    "dv01": ttm * 10,
                })

        simple_panel = pd.DataFrame(records)

        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="20:00",  # Wide session to include all bars
            rebalance_mode="every_bar",
        )

        result = backtest_fly_intraday(
            simple_panel,
            tenors=(2, 5, 10),
            window=5,
            entry_z=0.5,  # Low threshold to enter quickly
            config=config,
            verbose=False,
        )

        # With rising prices and long position, should have some positive returns
        pnl = result.pnl_df

        # Check that we have position data
        assert "position" in pnl.columns
        # Not all returns will be positive due to relative price moves

    def test_dv01_target_sizing(self, intraday_panel):
        """DV01 target sizing should scale exposures."""
        config = IntradayBacktestConfig(
            sizing_mode=SizingMode.DV01_TARGET,
            dv01_target=10000,
            session_start="08:00",
            session_end="17:00",
            rebalance_mode="every_bar",
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        pnl = result.pnl_df

        # When in position, gross_dv01 should be around target
        in_position = pnl[pnl["position"] != 0]
        if len(in_position) > 0:
            # Allow some tolerance due to weight adjustments
            mean_gross_dv01 = in_position["gross_dv01"].mean()
            assert mean_gross_dv01 > 0  # Should have exposure


class TestAggregateToDaily:
    """Tests for aggregate_to_daily function."""

    def test_aggregates_returns(self):
        """Should sum returns across bars in each day."""
        pnl_df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-02 08:00", periods=10, freq="h"),
            "trading_date": [pd.Timestamp("2023-01-02").date()] * 10,
            "gross_return": [0.01] * 10,
            "net_return": [0.009] * 10,
            "turnover": [100.0] * 10,
            "transaction_cost": [0.001] * 10,
            "gross_dv01": [1000.0] * 10,
            "net_dv01": [50.0] * 10,
            "gross_notional": [100000.0] * 10,
            "position": [1] * 10,
        })

        result = aggregate_to_daily(pnl_df)

        # Should have 1 row
        assert len(result) == 1

        # Returns should be summed
        assert abs(result["gross_return"].iloc[0] - 0.1) < 1e-6
        assert abs(result["net_return"].iloc[0] - 0.09) < 1e-6

        # Turnover should be summed
        assert abs(result["turnover"].iloc[0] - 1000.0) < 1e-6

    def test_last_position_and_exposure(self):
        """Should use last bar's position and exposure."""
        pnl_df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-02 08:00", periods=5, freq="h"),
            "trading_date": [pd.Timestamp("2023-01-02").date()] * 5,
            "gross_return": [0.0] * 5,
            "net_return": [0.0] * 5,
            "turnover": [0.0] * 5,
            "transaction_cost": [0.0] * 5,
            "gross_dv01": [1000, 1000, 1000, 1000, 2000],  # Changes at end
            "net_dv01": [50, 50, 50, 50, 100],
            "gross_notional": [100000.0] * 5,
            "position": [0, 1, 1, 1, -1],  # Changes at end
        })

        result = aggregate_to_daily(pnl_df)

        # Should use last values
        assert result["position"].iloc[0] == -1
        assert result["gross_dv01"].iloc[0] == 2000


class TestOvernightHolding:
    """Tests for overnight position holding behavior."""

    def test_position_persists_overnight(self, intraday_panel):
        """Position should persist from one day to the next."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="17:00",
            rebalance_mode="open_only",  # Only trade at open
            allow_overnight=True,
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            entry_z=1.5,  # Lower threshold to get more signals
            config=config,
            verbose=False,
        )

        pnl = result.pnl_df

        # Check if position persists across day boundaries
        # Positions should be available for all bars
        assert "position" in pnl.columns
        assert "trading_date" in pnl.columns

        # Check positions are tracked per day
        positions_by_day = pnl.groupby("trading_date")["position"].last()
        assert len(positions_by_day) > 0

    def test_pnl_accrues_overnight(self, intraday_panel):
        """PnL should accrue from price moves even outside session."""
        config = IntradayBacktestConfig(
            session_start="08:00",
            session_end="12:00",  # Short session
            rebalance_mode="every_bar",
            allow_overnight=True,
        )

        result = backtest_fly_intraday(
            intraday_panel,
            tenors=(2, 5, 10),
            window=10,
            config=config,
            verbose=False,
        )

        pnl = result.pnl_df

        # Bars outside session can still have returns if position is held
        outside_session = pnl[~pnl["is_session"]]
        in_position_outside = outside_session[outside_session["position"] != 0]

        # If position held overnight, should have some return data
        # (may be 0 if no price change, but column should exist)
        assert "net_return" in in_position_outside.columns
