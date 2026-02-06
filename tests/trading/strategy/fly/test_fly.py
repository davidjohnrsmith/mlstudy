"""Tests for butterfly (fly) trade construction."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.trading.strategy.structures.selection.curve_selection import select_fly_legs
from mlstudy.trading.strategy.structures.specs.fly.old.fly import (
    FlyResult,
    FlyWeights,
    build_fly,
    build_fly_legs_panel,
    build_fly_timeseries,
    compute_fly_carry,
    compute_fly_richness,
    compute_fly_value,
)


@pytest.fixture
def bond_panel():
    """Create bond panel with diverse TTMs and values."""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=50, freq="D")

    bonds = [
        ("bond_2y", 2.0),
        ("bond_5y", 5.0),
        ("bond_10y", 10.0),
        ("bond_20y", 20.0),
    ]

    records = []
    for date in dates:
        day_offset = (date - dates[0]).days
        for bond_id, base_ttm in bonds:
            ttm = base_ttm - day_offset / 365.0

            # Upward sloping yield curve with noise
            base_yield = 3.0 + 0.3 * np.log(ttm + 1) + np.random.randn() * 0.02
            price = 100 - (base_yield - 4) * 5 + np.random.randn() * 0.1
            dv01 = ttm * 10 + np.random.randn() * 1

            records.append({
                "datetime": date,
                "bond_id": bond_id,
                "ttm_years": ttm,
                "yield": base_yield,
                "price": price,
                "dv01": dv01,
            })

    return pd.DataFrame(records)


@pytest.fixture
def legs_table(bond_panel):
    """Pre-computed legs table."""
    return select_fly_legs(bond_panel, tenors=(2, 5, 10))


class TestFlyWeights:
    """Tests for FlyWeights dataclass."""

    def test_equal_weights(self):
        """Equal weights should be (1, -2, 1)."""
        w = FlyWeights.equal()

        assert w.front == 1.0
        assert w.belly == -2.0
        assert w.back == 1.0

    def test_as_tuple(self):
        """Should convert to tuple."""
        w = FlyWeights(front=1.0, belly=-2.0, back=1.0)
        assert w.as_tuple() == (1.0, -2.0, 1.0)

    def test_as_array(self):
        """Should convert to numpy array."""
        w = FlyWeights(front=1.0, belly=-2.0, back=1.0)
        arr = w.as_array()

        np.testing.assert_array_equal(arr, [1.0, -2.0, 1.0])

    def test_from_dv01_neutral(self):
        """DV01-neutral weights should sum to zero DV01."""
        # DV01s: 2Y=20, 5Y=50, 10Y=100
        w = FlyWeights.from_dv01(dv01_front=20, dv01_belly=50, dv01_back=100)

        # Check net DV01 is zero
        net_dv01 = w.front * 20 + w.belly * 50 + w.back * 100
        assert abs(net_dv01) < 1e-10

    def test_from_dv01_belly_negative(self):
        """Belly weight should be negative (short belly)."""
        w = FlyWeights.from_dv01(dv01_front=20, dv01_belly=50, dv01_back=100)
        assert w.belly < 0

    def test_from_dv01_wings_positive(self):
        """Wing weights should be positive (long wings)."""
        w = FlyWeights.from_dv01(dv01_front=20, dv01_belly=50, dv01_back=100)
        assert w.front > 0
        assert w.back > 0


class TestComputeFlyValue:
    """Tests for compute_fly_value."""

    def test_scalar_values(self):
        """Should compute fly from scalar values."""
        # yield_2y=3.0, yield_5y=3.5, yield_10y=4.0
        fly = compute_fly_value(3.0, 3.5, 4.0)

        # 1*3.0 - 2*3.5 + 1*4.0 = 3.0 - 7.0 + 4.0 = 0.0
        assert abs(fly - 0.0) < 1e-10

    def test_array_values(self):
        """Should compute fly from arrays."""
        front = np.array([3.0, 3.1, 3.2])
        belly = np.array([3.5, 3.6, 3.7])
        back = np.array([4.0, 4.1, 4.2])

        fly = compute_fly_value(front, belly, back)

        expected = front - 2 * belly + back
        np.testing.assert_array_almost_equal(fly, expected)

    def test_custom_weights(self):
        """Should use custom weights."""
        fly = compute_fly_value(3.0, 3.5, 4.0, weights=(2, -3, 1))

        # 2*3.0 - 3*3.5 + 1*4.0 = 6.0 - 10.5 + 4.0 = -0.5
        assert abs(fly - (-0.5)) < 1e-10

    def test_fly_weights_object(self):
        """Should accept FlyWeights object."""
        w = FlyWeights(front=1.5, belly=-2.5, back=1.0)
        fly = compute_fly_value(3.0, 3.5, 4.0, weights=w)

        expected = 1.5 * 3.0 - 2.5 * 3.5 + 1.0 * 4.0
        assert abs(fly - expected) < 1e-10


class TestBuildFlyTimeseries:
    """Tests for build_fly_timeseries."""

    def test_returns_dataframe(self, bond_panel, legs_table):
        """Should return DataFrame with expected columns."""
        fly_df = build_fly_timeseries(bond_panel, legs_table, value_col="yield")

        assert isinstance(fly_df, pd.DataFrame)
        assert "datetime" in fly_df.columns
        assert "fly_value" in fly_df.columns
        assert "front_bond" in fly_df.columns
        assert "belly_bond" in fly_df.columns
        assert "back_bond" in fly_df.columns

    def test_row_count_matches_legs(self, bond_panel, legs_table):
        """Should have one row per date in legs table."""
        fly_df = build_fly_timeseries(bond_panel, legs_table, value_col="yield")
        assert len(fly_df) == len(legs_table)

    def test_fly_value_formula(self, bond_panel, legs_table):
        """Fly value should match formula: front - 2*belly + back."""
        fly_df = build_fly_timeseries(bond_panel, legs_table, value_col="yield")

        # Check formula for each row
        for _, row in fly_df.iterrows():
            expected = row["front_value"] - 2 * row["belly_value"] + row["back_value"]
            assert abs(row["fly_value"] - expected) < 1e-10

    def test_includes_weights(self, bond_panel, legs_table):
        """Should include weights columns."""
        fly_df = build_fly_timeseries(bond_panel, legs_table, value_col="yield")

        assert "w_front" in fly_df.columns
        assert "w_belly" in fly_df.columns
        assert "w_back" in fly_df.columns

    def test_dv01_neutral_weights(self, bond_panel, legs_table):
        """Should compute DV01-neutral weights when requested."""
        fly_df = build_fly_timeseries(
            bond_panel, legs_table, value_col="yield", use_dv01_weights=True
        )

        # Check that weights exist and vary
        assert fly_df["w_front"].notna().all()
        assert fly_df["w_belly"].notna().all()


class TestBuildFlyLegsPanel:
    """Tests for build_fly_legs_panel."""

    def test_returns_dataframe(self, bond_panel, legs_table):
        """Should return DataFrame with expected columns."""
        legs_df = build_fly_legs_panel(bond_panel, legs_table)

        assert isinstance(legs_df, pd.DataFrame)
        assert "datetime" in legs_df.columns
        assert "bond_id" in legs_df.columns
        assert "leg" in legs_df.columns
        assert "weight" in legs_df.columns

    def test_three_legs_per_date(self, bond_panel, legs_table):
        """Should have 3 legs per date."""
        legs_df = build_fly_legs_panel(bond_panel, legs_table)

        legs_per_date = legs_df.groupby("datetime").size()
        assert (legs_per_date == 3).all()

    def test_leg_names(self, bond_panel, legs_table):
        """Should have front, belly, back leg names."""
        legs_df = build_fly_legs_panel(bond_panel, legs_table)

        leg_names = set(legs_df["leg"].unique())
        assert leg_names == {"front", "belly", "back"}

    def test_includes_values(self, bond_panel, legs_table):
        """Should include value columns."""
        legs_df = build_fly_legs_panel(
            bond_panel, legs_table, value_cols=["yield", "price", "dv01"]
        )

        assert "yield" in legs_df.columns
        assert "price" in legs_df.columns
        assert "dv01" in legs_df.columns

    def test_dv01_neutral_net_dv01(self, bond_panel, legs_table):
        """Net DV01 should be ~0 with DV01-neutral weights."""
        legs_df = build_fly_legs_panel(
            bond_panel, legs_table, use_dv01_weights=True, value_cols=["yield", "dv01"]
        )

        # Compute weighted DV01 per date
        legs_df["weighted_dv01"] = legs_df["weight"] * legs_df["dv01"]
        net_dv01_per_date = legs_df.groupby("datetime")["weighted_dv01"].sum()

        # Net DV01 should be close to zero
        assert net_dv01_per_date.abs().mean() < 1.0


class TestBuildFly:
    """Tests for build_fly convenience function."""

    def test_returns_fly_result(self, bond_panel, legs_table):
        """Should return FlyResult object."""
        result = build_fly(bond_panel, legs_table)

        assert isinstance(result, FlyResult)
        assert isinstance(result.fly_df, pd.DataFrame)
        assert isinstance(result.legs_df, pd.DataFrame)

    def test_get_fly_yield(self, bond_panel, legs_table):
        """Should extract fly yield series."""
        result = build_fly(bond_panel, legs_table)
        fly_yield = result.get_fly_yield()

        assert isinstance(fly_yield, pd.Series)
        assert len(fly_yield) == len(legs_table)

    def test_get_leg_values(self, bond_panel, legs_table):
        """Should extract leg values."""
        result = build_fly(bond_panel, legs_table, value_cols=["yield", "price"])

        front_yield = result.get_leg_values("front", "yield")
        assert isinstance(front_yield, pd.Series)
        assert len(front_yield) == len(legs_table)

    def test_get_net_dv01(self, bond_panel, legs_table):
        """Should compute net DV01."""
        result = build_fly(bond_panel, legs_table, use_dv01_weights=True)
        net_dv01 = result.get_net_dv01()

        # Should be close to zero
        assert net_dv01.abs().mean() < 1.0

    def test_compute_weighted_pnl(self, bond_panel, legs_table):
        """Should compute weighted PnL."""
        result = build_fly(bond_panel, legs_table, value_cols=["yield", "price", "dv01"])
        pnl = result.compute_weighted_pnl(price_col="price")

        assert isinstance(pnl, pd.DataFrame)
        assert "total" in pnl.columns
        assert "front" in pnl.columns
        assert "belly" in pnl.columns
        assert "back" in pnl.columns


class TestFlyAnalytics:
    """Tests for fly analytics functions."""

    def test_compute_fly_carry(self, bond_panel, legs_table):
        """Should compute fly carry."""
        result = build_fly(bond_panel, legs_table)
        carry = compute_fly_carry(result, roll_days=10)

        assert isinstance(carry, pd.Series)
        assert carry.name == "fly_carry"
        # First 10 values should be NaN
        assert carry.iloc[:10].isna().all()

    def test_compute_fly_richness(self, bond_panel, legs_table):
        """Should compute fly z-score."""
        result = build_fly(bond_panel, legs_table)
        zscore = compute_fly_richness(result, lookback=10)

        assert isinstance(zscore, pd.Series)
        assert zscore.name == "fly_zscore"
        # Z-score should be roughly bounded
        valid = zscore.dropna()
        assert valid.abs().mean() < 3.0  # Reasonable z-scores


class TestOutputFormats:
    """Tests for output DataFrame formats."""

    def test_fly_df_columns(self, bond_panel, legs_table):
        """fly_df should have specified columns."""
        result = build_fly(bond_panel, legs_table)
        fly_df = result.fly_df

        required_cols = [
            "datetime", "fly_value",
            "front_bond", "belly_bond", "back_bond",
            "front_value", "belly_value", "back_value",
            "w_front", "w_belly", "w_back",
        ]
        for col in required_cols:
            assert col in fly_df.columns, f"Missing column: {col}"

    def test_legs_df_columns(self, bond_panel, legs_table):
        """legs_df should have specified columns."""
        result = build_fly(bond_panel, legs_table, value_cols=["yield", "price", "dv01"])
        legs_df = result.legs_df

        required_cols = ["datetime", "bond_id", "leg", "weight", "yield", "price", "dv01"]
        for col in required_cols:
            assert col in legs_df.columns, f"Missing column: {col}"

    def test_fly_df_joinable(self, bond_panel, legs_table):
        """fly_df should be joinable back to panel."""
        result = build_fly(bond_panel, legs_table)
        fly_df = result.fly_df

        # Join front leg back to panel
        joined = fly_df.merge(
            bond_panel.rename(columns={"bond_id": "front_bond"}),
            on=["datetime", "front_bond"],
            how="left",
        )

        assert len(joined) == len(fly_df)
        assert joined["yield"].notna().all()

    def test_legs_df_pivotable(self, bond_panel, legs_table):
        """legs_df should be pivotable by leg."""
        result = build_fly(bond_panel, legs_table, value_cols=["yield"])
        legs_df = result.legs_df

        pivoted = legs_df.pivot(index="datetime", columns="leg", values="yield")

        assert pivoted.shape[1] == 3
        assert set(pivoted.columns) == {"front", "belly", "back"}
