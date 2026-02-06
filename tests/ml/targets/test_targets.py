"""Tests for target variable generation."""

import numpy as np
import pandas as pd
import pytest

from mlstudy.ml.targets.returns import (
    make_forward_direction_target,
    make_forward_return_target,
)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Simple DataFrame for testing."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "close": [100, 102, 101, 105, 103, 108, 106, 110, 109, 112],
        }
    )


@pytest.fixture
def grouped_df() -> pd.DataFrame:
    """DataFrame with multiple groups."""
    n = 10
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "datetime": list(dates) * 2,
            "asset": ["A"] * n + ["B"] * n,
            "close": list(range(100, 100 + n)) + list(range(50, 50 + n)),
        }
    )


class TestMakeForwardReturnTarget:
    """Tests for make_forward_return_target."""

    def test_returns_dataframe(self, simple_df: pd.DataFrame) -> None:
        """Should return a DataFrame."""
        result = make_forward_return_target(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_column_name(self, simple_df: pd.DataFrame) -> None:
        """Should have expected column name."""
        result = make_forward_return_target(simple_df, horizon_steps=1)
        assert "forward_log_return_1" in result.columns

    def test_custom_column_name(self, simple_df: pd.DataFrame) -> None:
        """Should use custom column name when provided."""
        result = make_forward_return_target(simple_df, target_col="my_target")
        assert "my_target" in result.columns

    def test_nan_at_end(self, simple_df: pd.DataFrame) -> None:
        """Last rows should have NaN (no future data)."""
        result = make_forward_return_target(simple_df, horizon_steps=1)
        assert pd.isna(result.iloc[-1, 0])

    def test_nan_count_matches_horizon(self, simple_df: pd.DataFrame) -> None:
        """Number of NaN should equal horizon_steps."""
        horizon = 3
        result = make_forward_return_target(simple_df, horizon_steps=horizon)
        nan_count = result.iloc[:, 0].isna().sum()
        assert nan_count == horizon

    def test_log_return_calculation(self) -> None:
        """Log return should be computed correctly."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=3, freq="1h"),
                "close": [100, 110, 100],  # 10% up, then ~9% down
            }
        )
        result = make_forward_return_target(df, horizon_steps=1, log_return=True)

        # log(110/100) = log(1.1) ≈ 0.0953
        expected = np.log(110 / 100)
        assert abs(result.iloc[0, 0] - expected) < 1e-10

    def test_simple_return_calculation(self) -> None:
        """Simple return should be computed correctly."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=3, freq="1h"),
                "close": [100, 110, 100],
            }
        )
        result = make_forward_return_target(df, horizon_steps=1, log_return=False)

        # (110 - 100) / 100 = 0.1
        assert result.iloc[0, 0] == 0.1

    def test_multi_step_horizon(self, simple_df: pd.DataFrame) -> None:
        """Multi-step horizon should look further ahead."""
        result = make_forward_return_target(simple_df, horizon_steps=3)
        # Should have 3 NaN at the end
        assert result.iloc[-3:, 0].isna().all()
        assert result.iloc[:-3, 0].notna().all()

    def test_grouped_returns(self, grouped_df: pd.DataFrame) -> None:
        """Should compute returns within each group."""
        result = make_forward_return_target(
            grouped_df, group_col="asset", horizon_steps=1
        )

        # Each group should have 1 NaN (at the end of that group)
        # After sorting by group and datetime, last row of each group is NaN
        combined = grouped_df.copy()
        combined["target"] = result.iloc[:, 0]
        combined = combined.sort_values(["asset", "datetime"])

        for asset in ["A", "B"]:
            asset_data = combined[combined["asset"] == asset]
            # Last row should be NaN
            assert pd.isna(asset_data["target"].iloc[-1])
            # Others should be valid
            assert asset_data["target"].iloc[:-1].notna().all()


class TestMakeForwardDirectionTarget:
    """Tests for make_forward_direction_target."""

    def test_returns_binary(self, simple_df: pd.DataFrame) -> None:
        """Should return binary values (0 or 1)."""
        result = make_forward_direction_target(simple_df, horizon_steps=1)
        valid = result.iloc[:, 0].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_direction_matches_return_sign(self) -> None:
        """Direction should match sign of return."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=4, freq="1h"),
                "close": [100, 110, 105, 108],  # up, down, up
            }
        )
        result = make_forward_direction_target(df, horizon_steps=1)

        assert result.iloc[0, 0] == 1  # 100 -> 110 = up
        assert result.iloc[1, 0] == 0  # 110 -> 105 = down
        assert result.iloc[2, 0] == 1  # 105 -> 108 = up
