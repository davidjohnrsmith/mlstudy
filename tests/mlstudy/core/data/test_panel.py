"""Tests for panel data utilities."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.core.data.panel import (
    fill_panel_gaps,
    get_panel_summary,
    pivot_by_bond,
    unpivot_to_panel,
    validate_panel,
)


@pytest.fixture
def bond_panel():
    """Create sample bond panel data."""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    bonds = [f"bond_{i:02d}" for i in range(10)]

    records = []
    for bond in bonds:
        base_yield = 4.0 + np.random.randn() * 0.5
        for i, date in enumerate(dates):
            # Random walk yield
            if i > 0:
                base_yield += np.random.randn() * 0.02
            records.append({
                "datetime": date,
                "bond_id": bond,
                "yield": base_yield,
                "price": 100 - (base_yield - 4) * 5,
                "dv01": 50 + np.random.randn() * 5,
                "ttm_years": 5 + np.random.randn() * 2,
            })

    return pd.DataFrame(records)


@pytest.fixture
def bond_panel_with_issues():
    """Create panel with validation issues."""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    bonds = ["bond_01", "bond_02", "bond_03"]

    records = []
    for bond in bonds:
        for date in dates:
            records.append({
                "datetime": date,
                "bond_id": bond,
                "yield": 4.0 + np.random.randn() * 0.1,
            })

    df = pd.DataFrame(records)

    # Add duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    return df


class TestValidatePanel:
    """Tests for validate_panel."""

    def test_valid_panel(self, bond_panel):
        """Should pass validation for clean panel."""
        result = validate_panel(bond_panel)

        assert result.is_valid
        assert result.n_dates == 50
        assert result.n_bonds == 10
        assert result.duplicate_keys == 0
        assert len(result.non_monotonic_bonds) == 0

    def test_detects_duplicates(self, bond_panel_with_issues):
        """Should detect duplicate keys."""
        result = validate_panel(bond_panel_with_issues)

        assert not result.is_valid
        assert result.duplicate_keys > 0

    def test_detects_non_monotonic(self, bond_panel):
        """Should detect non-monotonic datetime."""
        df = bond_panel.copy()
        # Shuffle dates for one bond
        mask = df["bond_id"] == "bond_00"
        shuffled_dates = df.loc[mask, "datetime"].sample(frac=1).values
        df.loc[mask, "datetime"] = shuffled_dates

        result = validate_panel(df)

        assert not result.is_valid
        assert "bond_00" in result.non_monotonic_bonds

    def test_computes_missing_rate(self, bond_panel):
        """Should compute missing value rate."""
        df = bond_panel.copy()
        # Add some missing values
        df.loc[df.index[:10], "yield"] = np.nan

        result = validate_panel(df, value_cols=["yield"])

        assert result.missing_rate > 0
        # 10 missing out of 500 total
        expected_rate = 10 / 500
        assert abs(result.missing_rate - expected_rate) < 0.01

    def test_result_string(self, bond_panel):
        """Should produce readable string output."""
        result = validate_panel(bond_panel)
        result_str = str(result)

        assert "PASSED" in result_str
        assert "Dates: 50" in result_str
        assert "Bonds: 10" in result_str


class TestPivotByBond:
    """Tests for pivot_by_bond."""

    def test_pivot_shape(self, bond_panel):
        """Should create wide format with correct shape."""
        pivoted = pivot_by_bond(bond_panel, value_col="yield")

        assert pivoted.shape == (50, 10)  # 50 dates x 10 bonds
        assert isinstance(pivoted.index, pd.DatetimeIndex)

    def test_pivot_values(self, bond_panel):
        """Should preserve values correctly."""
        pivoted = pivot_by_bond(bond_panel, value_col="yield")

        # Check a specific value
        sample = bond_panel[
            (bond_panel["datetime"] == bond_panel["datetime"].iloc[0]) &
            (bond_panel["bond_id"] == "bond_00")
        ]["yield"].iloc[0]

        assert pivoted.loc[pivoted.index[0], "bond_00"] == sample


class TestUnpivotToPanel:
    """Tests for unpivot_to_panel."""

    def test_roundtrip(self, bond_panel):
        """Should recover original data after pivot/unpivot."""
        pivoted = pivot_by_bond(bond_panel, value_col="yield")
        unpivoted = unpivot_to_panel(pivoted, value_name="yield")

        # Sort both for comparison
        orig = bond_panel[["datetime", "bond_id", "yield"]].sort_values(
            ["datetime", "bond_id"]
        ).reset_index(drop=True)
        recovered = unpivoted.sort_values(["datetime", "bond_id"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            orig[["datetime", "bond_id"]],
            recovered[["datetime", "bond_id"]],
        )
        np.testing.assert_array_almost_equal(
            orig["yield"].values,
            recovered["yield"].values,
        )


class TestFillPanelGaps:
    """Tests for fill_panel_gaps."""

    def test_fills_gaps(self):
        """Should fill date gaps within bonds."""
        # Create panel with gaps
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05"]),
            "bond_id": ["bond_01"] * 3,
            "yield": [4.0, 4.1, 4.2],
        })

        filled = fill_panel_gaps(df, freq="D", method="ffill")

        # Should have 5 dates now
        assert len(filled[filled["bond_id"] == "bond_01"]) == 5

    def test_ffill_preserves_values(self):
        """Forward fill should carry values forward."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-01", "2023-01-03"]),
            "bond_id": ["bond_01"] * 2,
            "yield": [4.0, 4.2],
        })

        filled = fill_panel_gaps(df, freq="D", method="ffill")
        filled = filled.sort_values("datetime")

        # Jan 2 should have Jan 1's value
        jan2 = filled[filled["datetime"] == pd.Timestamp("2023-01-02")]["yield"].iloc[0]
        assert jan2 == 4.0


class TestGetPanelSummary:
    """Tests for get_panel_summary."""

    def test_summary_columns(self, bond_panel):
        """Should include expected columns."""
        summary = get_panel_summary(bond_panel, value_cols=["yield"])

        assert "n_obs" in summary.columns
        assert "first_date" in summary.columns
        assert "last_date" in summary.columns
        assert "yield_mean" in summary.columns
        assert "yield_std" in summary.columns

    def test_summary_per_bond(self, bond_panel):
        """Should have one row per bond."""
        summary = get_panel_summary(bond_panel)

        assert len(summary) == 10
        assert summary.index.tolist() == sorted(bond_panel["bond_id"].unique())

    def test_obs_count(self, bond_panel):
        """Should correctly count observations."""
        summary = get_panel_summary(bond_panel)

        # Each bond should have 50 observations
        assert (summary["n_obs"] == 50).all()
