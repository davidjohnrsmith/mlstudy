"""Tests for intraday session utilities."""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from mlstudy.core.data.session import (
    add_session_flags,
    compute_trading_date,
    filter_session,
    get_session_boundaries,
    is_rebalance_bar,
    parse_time,
)


class TestParseTime:
    """Tests for parse_time function."""

    def test_parse_hhmm(self):
        """Should parse HH:MM format."""
        result = parse_time("07:30")
        assert result == time(7, 30)

    def test_parse_hhmmss(self):
        """Should parse HH:MM:SS format."""
        result = parse_time("07:30:45")
        assert result == time(7, 30, 45)

    def test_pass_through_time(self):
        """Should pass through time objects."""
        t = time(9, 0)
        result = parse_time(t)
        assert result == t

    def test_invalid_format_raises(self):
        """Should raise on invalid format."""
        with pytest.raises(ValueError):
            parse_time("invalid")


class TestFilterSession:
    """Tests for filter_session function."""

    @pytest.fixture
    def intraday_df(self):
        """Create intraday test data."""
        dates = pd.date_range(
            "2023-01-02 06:00",
            "2023-01-02 20:00",
            freq="30min",
            tz="Europe/Berlin",
        )
        return pd.DataFrame({
            "datetime": dates,
            "price": range(len(dates)),
        })

    def test_filters_to_session(self, intraday_df):
        """Should filter to session hours only."""
        result = filter_session(
            intraday_df,
            start="07:30",
            end="17:00",
            tz="Europe/Berlin",
        )

        # Check all times are within session
        times = result["datetime"].dt.time
        assert all(times >= time(7, 30))
        assert all(times <= time(17, 0))

    def test_excludes_outside_session(self, intraday_df):
        """Should exclude bars outside session."""
        result = filter_session(
            intraday_df,
            start="07:30",
            end="17:00",
            tz="Europe/Berlin",
        )

        # 06:00, 06:30, 07:00 should be excluded (before 07:30)
        # 17:30, 18:00, ..., 20:00 should be excluded (after 17:00)
        assert len(result) < len(intraday_df)

    def test_empty_result_if_no_session_bars(self):
        """Should return empty if no bars in session."""
        df = pd.DataFrame({
            "datetime": pd.date_range(
                "2023-01-02 18:00",
                "2023-01-02 23:00",
                freq="h",
                tz="Europe/Berlin",
            ),
            "price": [1, 2, 3, 4, 5, 6],
        })

        result = filter_session(df, start="07:30", end="17:00", tz="Europe/Berlin")
        assert len(result) == 0


class TestAddSessionFlags:
    """Tests for add_session_flags function."""

    @pytest.fixture
    def multi_day_df(self):
        """Create multi-day intraday data."""
        dates1 = pd.date_range(
            "2023-01-02 06:00",
            "2023-01-02 20:00",
            freq="h",
            tz="Europe/Berlin",
        )
        dates2 = pd.date_range(
            "2023-01-03 06:00",
            "2023-01-03 20:00",
            freq="h",
            tz="Europe/Berlin",
        )
        dates = dates1.append(dates2)
        return pd.DataFrame({
            "datetime": dates,
            "price": range(len(dates)),
        })

    def test_adds_is_session_flag(self, multi_day_df):
        """Should add is_session column."""
        result = add_session_flags(
            multi_day_df,
            start="07:30",
            end="17:00",
            tz="Europe/Berlin",
        )

        assert "is_session" in result.columns
        # 08:00, 09:00, ..., 17:00 should be in session (10 hours)
        # For each day: 6, 7, 8, ..., 17, 18, 19, 20 = 15 bars
        # Session: 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 = 10 bars
        session_count = result["is_session"].sum()
        assert session_count == 20  # 10 per day * 2 days

    def test_adds_trading_date(self, multi_day_df):
        """Should add trading_date column."""
        result = add_session_flags(
            multi_day_df,
            start="07:30",
            end="17:00",
            tz="Europe/Berlin",
        )

        assert "trading_date" in result.columns
        # Should have 2 unique trading dates
        assert result["trading_date"].nunique() == 2

    def test_adds_open_close_flags(self, multi_day_df):
        """Should add is_open_bar and is_close_bar flags."""
        result = add_session_flags(
            multi_day_df,
            start="07:30",
            end="17:00",
            tz="Europe/Berlin",
        )

        assert "is_open_bar" in result.columns
        assert "is_close_bar" in result.columns

        # Should have 2 open bars (one per day)
        assert result["is_open_bar"].sum() == 2
        # Should have 2 close bars (one per day)
        assert result["is_close_bar"].sum() == 2

    def test_open_bar_is_first_session_bar(self, multi_day_df):
        """Open bar should be the first session bar of the day."""
        result = add_session_flags(
            multi_day_df,
            start="07:30",
            end="17:00",
            tz="Europe/Berlin",
        )

        open_bars = result[result["is_open_bar"]]
        for _, row in open_bars.iterrows():
            # First session bar should be 08:00 (first hourly bar >= 07:30)
            assert row["datetime"].hour == 8


class TestIsRebalanceBar:
    """Tests for is_rebalance_bar function."""

    @pytest.fixture
    def session_df(self):
        """Create DataFrame with session flags."""
        dates = pd.date_range(
            "2023-01-02 07:00",
            "2023-01-02 18:00",
            freq="h",
            tz="Europe/Berlin",
        )
        df = pd.DataFrame({"datetime": dates, "price": range(len(dates))})
        return add_session_flags(df, start="08:00", end="17:00", tz="Europe/Berlin")

    def test_open_only_mode(self, session_df):
        """Should only flag open bar for open_only mode."""
        result = is_rebalance_bar(session_df, rebalance_mode="open_only")

        # Should only have 1 rebalance bar (open)
        assert result.sum() == 1
        assert result[session_df["is_open_bar"]].all()

    def test_close_only_mode(self, session_df):
        """Should only flag close bar for close_only mode."""
        result = is_rebalance_bar(session_df, rebalance_mode="close_only")

        # Should only have 1 rebalance bar (close)
        assert result.sum() == 1
        assert result[session_df["is_close_bar"]].all()

    def test_every_bar_mode(self, session_df):
        """Should flag all session bars for every_bar mode."""
        result = is_rebalance_bar(session_df, rebalance_mode="every_bar")

        # Should equal number of session bars
        assert result.sum() == session_df["is_session"].sum()

    def test_every_n_bars_mode(self, session_df):
        """Should flag every N session bars."""
        result = is_rebalance_bar(
            session_df,
            rebalance_mode="every_n_bars",
            n_bars=3,
        )

        # Session bars: 08:00, 09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00, 17:00
        # Every 3: 08:00 (0), 11:00 (3), 14:00 (6), 17:00 (9) = 4 bars
        assert result.sum() == 4

    def test_raises_without_session_flags(self):
        """Should raise if is_session column missing."""
        df = pd.DataFrame({"datetime": pd.date_range("2023-01-01", periods=5)})

        with pytest.raises(ValueError, match="is_session"):
            is_rebalance_bar(df, rebalance_mode="open_only")


class TestComputeTradingDate:
    """Tests for compute_trading_date function."""

    def test_during_session(self):
        """Bars during session should have same trading date."""
        dt = pd.Timestamp("2023-01-02 10:00", tz="Europe/Berlin")
        result = compute_trading_date(dt, session_start="07:30", tz="Europe/Berlin")

        from datetime import date
        assert result == date(2023, 1, 2)

    def test_before_session_start(self):
        """Bars before session start should have previous trading date."""
        dt = pd.Timestamp("2023-01-02 06:00", tz="Europe/Berlin")
        result = compute_trading_date(dt, session_start="07:30", tz="Europe/Berlin")

        from datetime import date
        assert result == date(2023, 1, 1)

    def test_series_input(self):
        """Should handle Series input."""
        dates = pd.Series([
            pd.Timestamp("2023-01-02 06:00", tz="Europe/Berlin"),
            pd.Timestamp("2023-01-02 10:00", tz="Europe/Berlin"),
        ])
        result = compute_trading_date(dates, session_start="07:30", tz="Europe/Berlin")

        from datetime import date
        assert result.iloc[0] == date(2023, 1, 1)  # Before session -> prev day
        assert result.iloc[1] == date(2023, 1, 2)  # During session -> same day


class TestGetSessionBoundaries:
    """Tests for get_session_boundaries function."""

    def test_returns_boundaries(self):
        """Should return session open and close times."""
        dates = pd.date_range(
            "2023-01-02 07:00",
            "2023-01-02 18:00",
            freq="h",
            tz="Europe/Berlin",
        )
        df = pd.DataFrame({"datetime": dates, "price": range(len(dates))})

        result = get_session_boundaries(
            df,
            start="08:00",
            end="17:00",
            tz="Europe/Berlin",
        )

        assert "trading_date" in result.columns
        assert "session_open" in result.columns
        assert "session_close" in result.columns
        assert len(result) == 1  # One trading day
