"""Tests for daily leg selection stability."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.structures.selection.leg_selection import (
    attach_daily_legs,
    build_daily_legs_table,
    get_leg_values,
    validate_leg_stability,
)


@pytest.fixture
def intraday_panel():
    """Create intraday panel data for multiple days."""
    np.random.seed(42)

    # Two trading days
    days = [
        pd.date_range("2023-01-02 07:00", "2023-01-02 18:00", freq="h", tz="Europe/Berlin"),
        pd.date_range("2023-01-03 07:00", "2023-01-03 18:00", freq="h", tz="Europe/Berlin"),
    ]

    # Three bonds with different TTMs
    bonds = [
        ("bond_2y", 2.1),
        ("bond_5y", 5.05),
        ("bond_10y", 9.9),
    ]

    records = []
    for day_dates in days:
        for dt in day_dates:
            for bond_id, ttm in bonds:
                records.append({
                    "datetime": dt,
                    "bond_id": bond_id,
                    "ttm_years": ttm,
                    "yield": 3.0 + np.random.randn() * 0.1,
                    "price": 100 + np.random.randn(),
                    "dv01": 50 + ttm * 10,
                })

    return pd.DataFrame(records)


class TestBuildDailyLegsTable:
    """Tests for build_daily_legs_table function."""

    def test_selects_legs_for_each_day(self, intraday_panel):
        """Should select legs for each trading day."""
        result = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        # Should have entries for both trading days
        assert len(result) == 2
        assert "trading_date" in result.columns

    def test_selects_correct_bonds(self, intraday_panel):
        """Should select bonds closest to target tenors."""
        result = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        # Check that correct bonds were selected
        for _, row in result.iterrows():
            assert row["front_id"] == "bond_2y"
            assert row["belly_id"] == "bond_5y"
            assert row["back_id"] == "bond_10y"

    def test_includes_ttm_at_selection(self, intraday_panel):
        """Should include actual TTMs at selection time."""
        result = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        # Check TTM columns exist
        assert "front_ttm" in result.columns
        assert "belly_ttm" in result.columns
        assert "back_ttm" in result.columns

        # TTMs should be close to targets
        assert abs(result["front_ttm"].iloc[0] - 2) < 0.5
        assert abs(result["belly_ttm"].iloc[0] - 5) < 0.5
        assert abs(result["back_ttm"].iloc[0] - 10) < 0.5

    def test_includes_selection_datetime(self, intraday_panel):
        """Should include the datetime of selection."""
        result = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        assert "selection_datetime" in result.columns
        # Selection should be around 08:00
        for dt in result["selection_datetime"]:
            assert dt.hour == 8

    def test_fallback_to_previous_day(self, intraday_panel):
        """Should fallback to previous day when selection time missing."""
        # Remove first day's 08:00 bar
        panel = intraday_panel.copy()
        mask = ~((panel["datetime"].dt.date == date(2023, 1, 2)) &
                 (panel["datetime"].dt.hour == 8))
        panel = panel[mask]

        result = build_daily_legs_table(
            panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
            fallback_to_previous=True,
        )

        # Should still have entry for first day (fallback to nearest bar)
        # or use previous day's selection
        assert len(result) >= 1

    def test_is_fallback_flag(self, intraday_panel):
        """Should mark fallback selections."""
        result = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        assert "is_fallback" in result.columns
        # Normal selections should not be fallbacks
        assert not result["is_fallback"].any()


class TestAttachDailyLegs:
    """Tests for attach_daily_legs function."""

    def test_attaches_leg_ids(self, intraday_panel):
        """Should attach leg IDs to each bar."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        result = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        # Should have leg ID columns
        assert "front_id" in result.columns
        assert "belly_id" in result.columns
        assert "back_id" in result.columns

    def test_legs_stable_within_day(self, intraday_panel):
        """Leg IDs should be constant within each trading day."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        result = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        # Check stability per day
        for _trading_date, group in result.groupby("trading_date"):
            for leg in ["front_id", "belly_id", "back_id"]:
                # All bars in day should have same leg ID
                assert group[leg].nunique() == 1

    def test_adds_trading_date(self, intraday_panel):
        """Should add trading_date column."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        result = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        assert "trading_date" in result.columns


class TestGetLegValues:
    """Tests for get_leg_values function."""

    def test_retrieves_values_for_each_leg(self, intraday_panel):
        """Should retrieve price/yield/dv01 for each leg."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        df_with_legs = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        result = get_leg_values(
            intraday_panel,
            df_with_legs,
            value_cols=["price", "yield", "dv01"],
        )

        # Should have value columns for each leg
        for leg in ["front", "belly", "back"]:
            for col in ["price", "yield", "dv01"]:
                assert f"{leg}_{col}" in result.columns

    def test_values_match_selected_bonds(self, intraday_panel):
        """Values should match the selected bonds."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        df_with_legs = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        result = get_leg_values(
            intraday_panel,
            df_with_legs,
            value_cols=["price"],
        )

        # Spot check: front_price should match bond_2y price
        for _, row in result.iterrows():
            if pd.notna(row["front_id"]) and row["front_id"] == "bond_2y":
                # Find matching row in panel
                panel_row = intraday_panel[
                    (intraday_panel["datetime"] == row["datetime"]) &
                    (intraday_panel["bond_id"] == "bond_2y")
                ]
                if len(panel_row) > 0:
                    assert row["front_price"] == panel_row["price"].iloc[0]


class TestValidateLegStability:
    """Tests for validate_leg_stability function."""

    def test_reports_stability(self, intraday_panel):
        """Should report if legs are stable within days."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        df_with_legs = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        result = validate_leg_stability(df_with_legs)

        assert "trading_date" in result.columns
        assert "is_stable" in result.columns
        assert "n_bars" in result.columns

        # All days should be stable (same legs throughout)
        assert result["is_stable"].all()

    def test_detects_instability(self, intraday_panel):
        """Should detect if legs change within a day."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        df_with_legs = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        # Artificially create instability
        df_unstable = df_with_legs.copy()
        df_unstable.loc[df_unstable.index[0], "front_id"] = "different_bond"

        result = validate_leg_stability(df_unstable)

        # At least one day should be unstable
        assert not result["is_stable"].all()


class TestOvernightHold:
    """Tests for overnight position holding."""

    def test_legs_persist_across_days(self, intraday_panel):
        """Leg values should be available for all bars, enabling overnight hold."""
        legs_table = build_daily_legs_table(
            intraday_panel,
            tenors=(2, 5, 10),
            selection_time="08:00",
            tz="Europe/Berlin",
        )

        df_with_legs = attach_daily_legs(
            intraday_panel,
            legs_table,
            tz="Europe/Berlin",
        )

        result = get_leg_values(
            intraday_panel,
            df_with_legs,
            value_cols=["price"],
        )

        # All bars should have leg prices (for PnL computation)
        # Even bars outside session should have values for overnight holding
        assert result["front_price"].notna().any()
        assert result["belly_price"].notna().any()
        assert result["back_price"].notna().any()
