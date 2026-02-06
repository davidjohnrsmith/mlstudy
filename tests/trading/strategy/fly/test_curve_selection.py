"""Tests for curve trading bond selection utilities."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.structures.selection.curve_selection import (
    compute_fly_weights,
    get_fly_values,
    select_fly_legs,
    select_nearest_three_sorted_by_ttm,
    select_nearest_to_tenor,
    select_steepener_legs,
)


@pytest.fixture
def bond_panel():
    """Create bond panel with diverse TTMs for curve selection."""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    # Create bonds with different maturities
    bonds = [
        ("bond_2y_a", 1.8),
        ("bond_2y_b", 2.2),
        ("bond_3y", 3.0),
        ("bond_5y_a", 4.8),
        ("bond_5y_b", 5.2),
        ("bond_7y", 7.0),
        ("bond_10y_a", 9.5),
        ("bond_10y_b", 10.5),
        ("bond_20y", 20.0),
        ("bond_30y", 30.0),
    ]

    records = []
    for date in dates:
        for bond_id, base_ttm in bonds:
            # TTM decreases slightly each day
            day_offset = (date - dates[0]).days
            ttm = base_ttm - day_offset / 365.0

            # Synthetic yield curve (upward sloping)
            base_yield = 3.0 + 0.2 * np.log(ttm + 1) + np.random.randn() * 0.02
            price = 100 - (base_yield - 4) * 5
            dv01 = ttm * 10 + np.random.randn() * 2

            records.append({
                "datetime": date,
                "bond_id": bond_id,
                "ttm_years": ttm,
                "yield": base_yield,
                "price": price,
                "dv01": dv01,
            })

    return pd.DataFrame(records)


class TestSelectNearestToTenor:
    """Tests for select_nearest_to_tenor."""

    def test_selects_one_per_date(self, bond_panel):
        """Should select exactly one bond per date."""
        result = select_nearest_to_tenor(bond_panel, tenor_years=5.0)

        assert len(result) == 30  # One per date
        assert result["bond_id"].notna().all()

    def test_selects_closest_ttm(self, bond_panel):
        """Should select bond with TTM closest to target."""
        result = select_nearest_to_tenor(bond_panel, tenor_years=5.0)

        # Should mostly select 5y bonds
        assert result["bond_id"].str.contains("5y").mean() > 0.5

    def test_deviation_column(self, bond_panel):
        """Should include deviation from target."""
        result = select_nearest_to_tenor(bond_panel, tenor_years=5.0)

        assert "deviation" in result.columns
        # Deviation should be small for 5y selection
        assert result["deviation"].mean() < 1.0

    def test_max_deviation_filter(self, bond_panel):
        """Should filter by max deviation."""
        # Select 15y (no bond close to this)
        result = select_nearest_to_tenor(
            bond_panel, tenor_years=15.0, max_deviation=1.0
        )

        # Should have NaN bond_ids (no bonds within 1y of 15y)
        assert result["bond_id"].isna().any()


class TestSelectFlyLegs:
    """Tests for select_fly_legs."""

    def test_selects_three_legs(self, bond_panel):
        """Should select short, belly, and long legs."""
        result = select_fly_legs(bond_panel, tenors=(2, 5, 10))

        assert "short_bond" in result.columns
        assert "belly_bond" in result.columns
        assert "long_bond" in result.columns

    def test_leg_ttms_ordered(self, bond_panel):
        """Short TTM < Belly TTM < Long TTM."""
        result = select_fly_legs(bond_panel, tenors=(2, 5, 10))

        # TTMs should be ordered
        assert (result["short_ttm"] < result["belly_ttm"]).all()
        assert (result["belly_ttm"] < result["long_ttm"]).all()

    def test_custom_tenors(self, bond_panel):
        """Should work with custom tenor triplet."""
        result = select_fly_legs(bond_panel, tenors=(3, 7, 20))

        assert len(result) == 30
        # Check approximate TTMs
        assert result["short_ttm"].mean() < 5
        assert result["belly_ttm"].mean() < 10
        assert result["long_ttm"].mean() > 15

    def test_liquidity_filter(self, bond_panel):
        """Should filter by liquidity threshold."""
        # Use high DV01 threshold
        result = select_fly_legs(
            bond_panel, tenors=(2, 5, 10),
            min_liquidity_col="dv01",
            min_liquidity_threshold=50,
        )

        # Should still have results (most bonds have dv01 > 50)
        assert result["short_bond"].notna().any()

    def test_deviation_columns(self, bond_panel):
        """Should include deviation for each leg."""
        result = select_fly_legs(bond_panel, tenors=(2, 5, 10))

        assert "short_deviation" in result.columns
        assert "belly_deviation" in result.columns
        assert "long_deviation" in result.columns


class TestSelectNearestThreeSortedByTTM:
    """Tests for select_nearest_three_sorted_by_ttm."""

    def test_selects_three_bonds(self, bond_panel):
        """Should select exactly three bonds per date."""
        result = select_nearest_three_sorted_by_ttm(
            bond_panel, center_tenor=5.0, window_years=3.0
        )

        assert "bond_1" in result.columns
        assert "bond_2" in result.columns
        assert "bond_3" in result.columns

    def test_sorted_by_ttm(self, bond_panel):
        """Bonds should be sorted by TTM (bond_1 shortest)."""
        result = select_nearest_three_sorted_by_ttm(
            bond_panel, center_tenor=5.0, window_years=3.0
        )

        # TTMs should be sorted
        valid = result.dropna(subset=["ttm_1", "ttm_2", "ttm_3"])
        assert (valid["ttm_1"] <= valid["ttm_2"]).all()
        assert (valid["ttm_2"] <= valid["ttm_3"]).all()

    def test_within_window(self, bond_panel):
        """Selected bonds should be within window of center."""
        result = select_nearest_three_sorted_by_ttm(
            bond_panel, center_tenor=5.0, window_years=2.0
        )

        # All TTMs should be within 2 years of 5
        for ttm_col in ["ttm_1", "ttm_2", "ttm_3"]:
            valid = result[result[ttm_col].notna()]
            deviations = np.abs(valid[ttm_col] - 5.0)
            assert (deviations <= 2.0 + 0.1).all()  # Small tolerance

    def test_require_distinct(self, bond_panel):
        """Should return NaN when not enough bonds in window."""
        # Very narrow window around 15y (no bonds)
        result = select_nearest_three_sorted_by_ttm(
            bond_panel, center_tenor=15.0, window_years=1.0, require_distinct=True
        )

        # Should have NaN bonds
        assert result["bond_1"].isna().all()


class TestComputeFlyWeights:
    """Tests for compute_fly_weights."""

    def test_equal_weights(self, bond_panel):
        """Equal weights should be (1, -2, 1)."""
        legs = select_fly_legs(bond_panel, tenors=(2, 5, 10))
        weights = compute_fly_weights(legs, weight_method="equal")

        assert (weights["w_short"] == 1).all()
        assert (weights["w_belly"] == -2).all()
        assert (weights["w_long"] == 1).all()

    def test_duration_neutral_weights(self, bond_panel):
        """Duration neutral weights should sum to zero DV01."""
        legs = select_fly_legs(bond_panel, tenors=(2, 5, 10))

        # Get DV01 values for selected bonds
        dv01_lookup = bond_panel.set_index(["datetime", "bond_id"])["dv01"]

        dv01_short = pd.Series([
            dv01_lookup.loc[(row["datetime"], row["short_bond"])]
            for _, row in legs.iterrows()
        ])
        dv01_belly = pd.Series([
            dv01_lookup.loc[(row["datetime"], row["belly_bond"])]
            for _, row in legs.iterrows()
        ])
        dv01_long = pd.Series([
            dv01_lookup.loc[(row["datetime"], row["long_bond"])]
            for _, row in legs.iterrows()
        ])

        weights = compute_fly_weights(
            legs,
            weight_method="duration_neutral",
            dv01_short=dv01_short,
            dv01_belly=dv01_belly,
            dv01_long=dv01_long,
        )

        # Check duration neutrality
        total_dv01 = (
            weights["w_short"] * dv01_short +
            weights["w_belly"] * dv01_belly +
            weights["w_long"] * dv01_long
        )
        np.testing.assert_array_almost_equal(total_dv01, 0, decimal=10)


class TestGetFlyValues:
    """Tests for get_fly_values."""

    def test_computes_fly_yield(self, bond_panel):
        """Should compute butterfly yield spread."""
        legs = select_fly_legs(bond_panel, tenors=(2, 5, 10))
        fly_yield = get_fly_values(bond_panel, legs, value_col="yield")

        assert len(fly_yield) == 30
        assert fly_yield.name == "fly_yield"

    def test_fly_formula(self, bond_panel):
        """Fly value should equal short - 2*belly + long."""
        legs = select_fly_legs(bond_panel, tenors=(2, 5, 10))
        fly_yield = get_fly_values(bond_panel, legs, value_col="yield")

        # Manual calculation for first date
        row = legs.iloc[0]
        dt = row["datetime"]

        yield_lookup = bond_panel.set_index(["datetime", "bond_id"])["yield"]
        y_short = yield_lookup.loc[(dt, row["short_bond"])]
        y_belly = yield_lookup.loc[(dt, row["belly_bond"])]
        y_long = yield_lookup.loc[(dt, row["long_bond"])]

        expected = y_short - 2 * y_belly + y_long
        assert abs(fly_yield.iloc[0] - expected) < 1e-10


class TestSelectSteepenerLegs:
    """Tests for select_steepener_legs."""

    def test_selects_two_legs(self, bond_panel):
        """Should select short and long legs."""
        result = select_steepener_legs(bond_panel, short_tenor=2.0, long_tenor=10.0)

        assert "short_bond" in result.columns
        assert "long_bond" in result.columns
        assert "short_ttm" in result.columns
        assert "long_ttm" in result.columns

    def test_ttm_ordering(self, bond_panel):
        """Short TTM should be less than long TTM."""
        result = select_steepener_legs(bond_panel, short_tenor=2.0, long_tenor=10.0)

        assert (result["short_ttm"] < result["long_ttm"]).all()


class TestSelectionStability:
    """Tests for selection stability with varying TTMs."""

    @pytest.fixture
    def varying_ttm_panel(self):
        """Panel where bond TTMs change over time (roll-down effect)."""
        np.random.seed(123)

        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Bonds with initial TTMs that roll down
        bonds = [
            ("bond_a", 2.5),   # Starts near 2.5y
            ("bond_b", 5.3),   # Starts near 5y
            ("bond_c", 5.1),   # Also near 5y (competing)
            ("bond_d", 10.2),  # Starts near 10y
            ("bond_e", 9.8),   # Also near 10y (competing)
        ]

        records = []
        for date in dates:
            day_offset = (date - dates[0]).days
            for bond_id, initial_ttm in bonds:
                # TTM decreases by 1/365 per day
                ttm = initial_ttm - day_offset / 365.0

                records.append({
                    "datetime": date,
                    "bond_id": bond_id,
                    "ttm_years": ttm,
                    "yield": 4.0 + 0.1 * ttm + np.random.randn() * 0.01,
                    "price": 100 - ttm,
                })

        return pd.DataFrame(records)

    def test_selection_changes_with_rolldown(self, varying_ttm_panel):
        """Selection should change as bonds roll through tenor point."""
        result = select_nearest_to_tenor(varying_ttm_panel, tenor_years=5.0)

        # Early: bond_c (5.1y) should be selected
        # Later: bond_b (5.3y) may be closer after roll-down
        early_selection = result.iloc[0]["bond_id"]
        late_selection = result.iloc[-1]["bond_id"]

        # Both should be valid 5y candidates
        assert early_selection in ["bond_b", "bond_c"]
        assert late_selection in ["bond_b", "bond_c"]

    def test_selection_correctness(self, varying_ttm_panel):
        """Selected bond should always be the closest to target."""
        result = select_nearest_to_tenor(varying_ttm_panel, tenor_years=5.0)

        for _, row in result.iterrows():
            dt = row["datetime"]
            selected_ttm = row["ttm_years"]

            # Get all bonds on this date
            day_data = varying_ttm_panel[varying_ttm_panel["datetime"] == dt]

            # Verify selected is closest to target tenor
            for _, other in day_data.iterrows():
                other_deviation = abs(other["ttm_years"] - 5.0)
                selected_deviation = abs(selected_ttm - 5.0)
                assert selected_deviation <= other_deviation + 1e-10

    def test_fly_selection_stability(self, varying_ttm_panel):
        """Fly legs should maintain ordering despite roll-down."""
        result = select_fly_legs(varying_ttm_panel, tenors=(2.5, 5.0, 10.0))

        # Check ordering is maintained
        assert (result["short_ttm"] < result["belly_ttm"]).all()
        assert (result["belly_ttm"] < result["long_ttm"]).all()


class TestJoinability:
    """Tests for joining selection results back to panel data."""

    def test_legs_join_to_panel(self, bond_panel):
        """Legs output should join cleanly to original panel."""
        legs = select_fly_legs(bond_panel, tenors=(2, 5, 10))

        # Join short leg
        short_joined = legs.merge(
            bond_panel.rename(columns={"bond_id": "short_bond"}),
            on=["datetime", "short_bond"],
            how="left",
        )

        # Should have same number of rows
        assert len(short_joined) == len(legs)

        # Should have yield values from original
        assert "yield" in short_joined.columns
        assert short_joined["yield"].notna().all()

    def test_nearest_join_to_panel(self, bond_panel):
        """Nearest selection should join back to panel."""
        selected = select_nearest_to_tenor(bond_panel, tenor_years=5.0)

        # Join back
        joined = selected.merge(
            bond_panel,
            on=["datetime", "bond_id"],
            how="left",
        )

        # Should get full bond data
        assert len(joined) == len(selected)
        assert "yield" in joined.columns
        assert "price" in joined.columns

    def test_multi_leg_join(self, bond_panel):
        """Should be able to join all three legs to get full data."""
        legs = select_fly_legs(bond_panel, tenors=(2, 5, 10))

        # Create lookup
        bond_data = bond_panel.set_index(["datetime", "bond_id"])

        # Function to get leg values
        def get_leg_values(legs_df, leg_name, value_col):
            values = []
            for _, row in legs_df.iterrows():
                bond_id = row[f"{leg_name}_bond"]
                dt = row["datetime"]
                try:
                    val = bond_data.loc[(dt, bond_id), value_col]
                    values.append(val)
                except KeyError:
                    values.append(np.nan)
            return pd.Series(values)

        # Get yields for each leg
        legs["short_yield"] = get_leg_values(legs, "short", "yield")
        legs["belly_yield"] = get_leg_values(legs, "belly", "yield")
        legs["long_yield"] = get_leg_values(legs, "long", "yield")

        # All should be populated
        assert legs["short_yield"].notna().all()
        assert legs["belly_yield"].notna().all()
        assert legs["long_yield"].notna().all()

    def test_three_nearest_join(self, bond_panel):
        """Three nearest selection should join cleanly."""
        selected = select_nearest_three_sorted_by_ttm(
            bond_panel, center_tenor=5.0, window_years=4.0
        )

        # Verify we can lookup bond_1 data
        bond_data = bond_panel.set_index(["datetime", "bond_id"])

        yields = []
        for _, row in selected.iterrows():
            if pd.notna(row["bond_1"]):
                yields.append(bond_data.loc[(row["datetime"], row["bond_1"]), "yield"])
            else:
                yields.append(np.nan)

        # Most should have values
        assert pd.Series(yields).notna().mean() > 0.9
