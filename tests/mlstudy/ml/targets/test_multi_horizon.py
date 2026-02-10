"""Tests for multi-horizon prediction and uncertainty quantification."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.ml.targets import (
    MultiHorizonTargetGenerator,
    extract_horizon,
    get_horizon_columns,
    make_forward_change_target,
    make_multi_horizon_targets,
)


@pytest.fixture
def panel_df():
    """Create synthetic panel data with weak predictable component."""
    np.random.seed(42)

    n_days = 200
    n_bonds = 20
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    records = []
    for bond_id in range(n_bonds):
        # Random walk with drift based on bond characteristics
        bond_drift = 0.001 * (bond_id % 5 - 2)  # Different drifts
        bond_vol = 0.02 + 0.01 * (bond_id % 3)  # Different volatilities

        # Starting yield
        yield_level = 4.0 + np.random.randn() * 0.5

        for i, date in enumerate(dates):
            # Random walk for yield
            if i > 0:
                yield_level += bond_drift + bond_vol * np.random.randn()

            # Features that are weakly predictive
            feat_momentum = np.random.randn() * 0.5
            feat_value = np.random.randn() * 0.3
            feat_carry = bond_drift * 10 + np.random.randn() * 0.2

            records.append({
                "datetime": date,
                "bond_id": f"bond_{bond_id:02d}",
                "hedged_yield": yield_level,
                "feat_momentum": feat_momentum,
                "feat_value": feat_value,
                "feat_carry": feat_carry,
            })

    df = pd.DataFrame(records)
    return df


class TestMakeForwardChangeTarget:
    """Tests for make_forward_change_target."""

    def test_returns_series(self, panel_df):
        """Should return a Series."""
        target = make_forward_change_target(
            panel_df, value_col="hedged_yield", horizon_steps=5, group_col="bond_id"
        )
        assert isinstance(target, pd.Series)

    def test_correct_name(self, panel_df):
        """Should have correct column name."""
        target = make_forward_change_target(
            panel_df, value_col="hedged_yield", horizon_steps=5, group_col="bond_id"
        )
        assert target.name == "target_5"

    def test_custom_name(self, panel_df):
        """Should use custom column name."""
        target = make_forward_change_target(
            panel_df, value_col="hedged_yield", horizon_steps=5,
            group_col="bond_id", target_col="custom_target"
        )
        assert target.name == "custom_target"

    def test_nan_at_end_of_groups(self, panel_df):
        """Should have NaN at the end of each group."""
        target = make_forward_change_target(
            panel_df, value_col="hedged_yield", horizon_steps=5, group_col="bond_id"
        )
        # Each group should have 5 NaN values at the end
        n_bonds = panel_df["bond_id"].nunique()
        n_nan = target.isna().sum()
        assert n_nan == n_bonds * 5

    def test_change_calculation(self, panel_df):
        """Should correctly calculate forward change."""
        df = panel_df.copy()
        target = make_forward_change_target(
            df, value_col="hedged_yield", horizon_steps=1, group_col="bond_id"
        )

        # Check a specific group
        bond_data = df[df["bond_id"] == "bond_00"].copy()
        bond_data = bond_data.sort_values("datetime")
        target_bond = target[bond_data.index]

        # First value should be second yield minus first yield
        expected = bond_data["hedged_yield"].iloc[1] - bond_data["hedged_yield"].iloc[0]
        actual = target_bond.iloc[0]
        np.testing.assert_almost_equal(actual, expected)


class TestMakeMultiHorizonTargets:
    """Tests for make_multi_horizon_targets."""

    def test_creates_target_columns(self, panel_df):
        """Should create target columns for each horizon."""
        horizons = [1, 5, 10, 20]
        result = make_multi_horizon_targets(
            panel_df, value_col="hedged_yield", horizons=horizons, group_col="bond_id"
        )

        for h in horizons:
            assert f"target_{h}" in result.columns

    def test_preserves_original_columns(self, panel_df):
        """Should preserve original columns."""
        result = make_multi_horizon_targets(
            panel_df, value_col="hedged_yield", horizons=[1, 5], group_col="bond_id"
        )

        for col in panel_df.columns:
            assert col in result.columns

    def test_nan_count_matches_horizon(self, panel_df):
        """NaN count should match horizon * n_groups."""
        horizons = [1, 5, 10]
        result = make_multi_horizon_targets(
            panel_df, value_col="hedged_yield", horizons=horizons, group_col="bond_id"
        )

        n_bonds = panel_df["bond_id"].nunique()
        for h in horizons:
            n_nan = result[f"target_{h}"].isna().sum()
            assert n_nan == n_bonds * h

    def test_method_return(self, panel_df):
        """Should calculate percentage returns."""
        result = make_multi_horizon_targets(
            panel_df, value_col="hedged_yield", horizons=[1], group_col="bond_id",
            method="return"
        )

        # Check a specific calculation
        bond_data = panel_df[panel_df["bond_id"] == "bond_00"].sort_values("datetime")
        expected = (bond_data["hedged_yield"].iloc[1] / bond_data["hedged_yield"].iloc[0]) - 1

        result_bond = result[result["bond_id"] == "bond_00"].sort_values("datetime")
        actual = result_bond["target_1"].iloc[0]

        np.testing.assert_almost_equal(actual, expected)


class TestMultiHorizonTargetGenerator:
    """Tests for MultiHorizonTargetGenerator."""

    def test_fit_transform(self, panel_df):
        """Should add target columns."""
        gen = MultiHorizonTargetGenerator(
            horizons=[1, 5, 10],
            value_col="hedged_yield",
            group_col="bond_id",
        )
        result = gen.fit_transform(panel_df)

        assert "target_1" in result.columns
        assert "target_5" in result.columns
        assert "target_10" in result.columns

    def test_get_X_y(self, panel_df):
        """Should return aligned X and y_dict."""
        gen = MultiHorizonTargetGenerator(
            horizons=[1, 5],
            value_col="hedged_yield",
            group_col="bond_id",
        )
        df = gen.fit_transform(panel_df)
        feature_cols = ["feat_momentum", "feat_value", "feat_carry"]

        X, y_dict = gen.get_X_y(df, feature_cols, dropna=True)

        assert X.shape[1] == 3  # 3 features
        assert 1 in y_dict
        assert 5 in y_dict
        assert len(y_dict[1]) == X.shape[0]
        assert len(y_dict[5]) == X.shape[0]

    def test_get_X_y_stacked(self, panel_df):
        """Should return stacked data with horizon indicator."""
        gen = MultiHorizonTargetGenerator(
            horizons=[1, 5],
            value_col="hedged_yield",
            group_col="bond_id",
        )
        df = gen.fit_transform(panel_df)
        feature_cols = ["feat_momentum", "feat_value", "feat_carry"]

        X_stacked, y_stacked, horizons_stacked = gen.get_X_y_stacked(df, feature_cols, dropna=True)

        # Stacked should be 2x the single horizon
        X, y_dict = gen.get_X_y(df, feature_cols, dropna=True)
        assert X_stacked.shape[0] == 2 * X.shape[0]
        assert len(y_stacked) == 2 * len(y_dict[1])
        assert set(horizons_stacked) == {1, 5}


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_horizon_columns(self, panel_df):
        """Should return sorted horizon columns."""
        df = make_multi_horizon_targets(
            panel_df, value_col="hedged_yield", horizons=[5, 1, 10], group_col="bond_id"
        )
        cols = get_horizon_columns(df)

        assert cols == ["target_1", "target_5", "target_10"]

    def test_extract_horizon(self):
        """Should extract horizon from column name."""
        assert extract_horizon("target_5") == 5
        assert extract_horizon("target_20") == 20
        assert extract_horizon("pred_100") == 100
