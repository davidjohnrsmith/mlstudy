"""Tests for portfolio interface types and adapters."""

from __future__ import annotations

import pandas as pd
import pytest

from mlstudy.trading.portfolio.aggregate import (
    AggregationResult,
    SizingRules,
    aggregate_signal_batch,
    compute_position_changes,
    compute_turnover,
    create_sizing_rules_with_weighting,
    signals_to_bond_targets,
)
from mlstudy.trading.portfolio.backtest import (
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    RebalanceRule,
    Trade,
    compute_costs_summary,
    compute_pnl_from_prices,
    run_portfolio_backtest,
    run_portfolio_backtest_from_targets,
    simulate_rebalance,
)
from mlstudy.trading.portfolio.signal_adapter import (
    LegWeight,
    StrategySignal,
    StrategySignalBatch,
    batch_signals_by_timestamp,
    create_fly_legs,
    filter_signals_by_direction,
    fly_name_to_strategy_id,
    get_active_signals,
    signal_df_to_strategy_signals,

)
from mlstudy.trading.portfolio.portfolio_types import (
    PortfolioTarget,
    signals_to_dataframe,
)

from mlstudy.trading.portfolio.weighting import (
    WeightingConfig,
    WeightingMethod,
    apply_weight_caps,
    compute_equal_weights,
    compute_inverse_vol_weights,
    compute_strategy_weights,
    validate_weights,
)


class TestLegWeight:
    """Tests for LegWeight dataclass."""

    def test_basic_creation(self):
        """Should create leg weight with basic attributes."""
        leg = LegWeight(bond_id="UST_2Y", weight=1.0)
        assert leg.bond_id == "UST_2Y"
        assert leg.weight == 1.0
        assert leg.tenor is None

    def test_with_tenor(self):
        """Should store optional tenor."""
        leg = LegWeight(bond_id="UST_5Y", weight=-2.0, tenor=5.0)
        assert leg.tenor == 5.0

    def test_repr(self):
        """Should have readable repr."""
        leg = LegWeight(bond_id="UST_10Y", weight=1.0, tenor=10.0)
        assert "UST_10Y" in repr(leg)
        assert "+1.00" in repr(leg)


class TestStrategySignal:
    """Tests for StrategySignal dataclass."""

    @pytest.fixture
    def sample_legs(self):
        """Create sample fly legs."""
        return [
            LegWeight("UST_2Y", 1.0, tenor=2.0),
            LegWeight("UST_5Y", -2.0, tenor=5.0),
            LegWeight("UST_10Y", 1.0, tenor=10.0),
        ]

    def test_basic_creation(self, sample_legs):
        """Should create signal with basic attributes."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            strategy_id="fly_2y5y10y",
            legs=sample_legs,
            direction=1,
        )

        assert signal.strategy_id == "fly_2y5y10y"
        assert signal.direction == 1
        assert len(signal.legs) == 3

    def test_direction_validation(self, sample_legs):
        """Should validate direction is -1, 0, or 1."""
        with pytest.raises(ValueError, match="direction must be"):
            StrategySignal(
                timestamp=pd.Timestamp("2024-01-15"),
                strategy_id="test",
                legs=sample_legs,
                direction=2,
            )

    def test_confidence_validation(self, sample_legs):
        """Should validate confidence is in [0, 1]."""
        with pytest.raises(ValueError, match="confidence must be"):
            StrategySignal(
                timestamp=pd.Timestamp("2024-01-15"),
                strategy_id="test",
                legs=sample_legs,
                direction=1,
                confidence=1.5,
            )

    def test_is_flat(self, sample_legs):
        """Should correctly identify flat signals."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=sample_legs,
            direction=0,
        )
        assert signal.is_flat
        assert not signal.is_long
        assert not signal.is_short

    def test_is_long(self, sample_legs):
        """Should correctly identify long signals."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=sample_legs,
            direction=1,
        )
        assert signal.is_long
        assert not signal.is_flat
        assert not signal.is_short

    def test_is_short(self, sample_legs):
        """Should correctly identify short signals."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=sample_legs,
            direction=-1,
        )
        assert signal.is_short
        assert not signal.is_flat
        assert not signal.is_long

    def test_leg_ids(self, sample_legs):
        """Should return list of leg IDs."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=sample_legs,
            direction=1,
        )
        assert signal.leg_ids == ["UST_2Y", "UST_5Y", "UST_10Y"]

    def test_weights_dict(self, sample_legs):
        """Should return weights as dict."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=sample_legs,
            direction=1,
        )
        weights = signal.weights_dict
        assert weights["UST_2Y"] == 1.0
        assert weights["UST_5Y"] == -2.0
        assert weights["UST_10Y"] == 1.0

    def test_scaled_weights(self, sample_legs):
        """Should scale weights by direction and multiplier."""
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=sample_legs,
            direction=-1,  # Short
        )
        scaled = signal.scaled_weights(scale=2.0)

        # Direction=-1 flips signs, scale=2 doubles
        assert scaled["UST_2Y"] == -2.0  # 1.0 * -1 * 2
        assert scaled["UST_5Y"] == 4.0  # -2.0 * -1 * 2
        assert scaled["UST_10Y"] == -2.0  # 1.0 * -1 * 2

    def test_to_dict_from_dict(self, sample_legs):
        """Should serialize and deserialize correctly."""
        original = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            strategy_id="fly_2y5y10y",
            legs=sample_legs,
            direction=1,
            signal_value=-2.5,
            target_gross_dv01=10000,
            confidence=0.8,
            metadata={"entry_reason": "mean_reversion"},
        )

        d = original.to_dict()
        reconstructed = StrategySignal.from_dict(d)

        assert reconstructed.strategy_id == original.strategy_id
        assert reconstructed.direction == original.direction
        assert reconstructed.signal_value == original.signal_value
        assert reconstructed.target_gross_dv01 == original.target_gross_dv01
        assert reconstructed.confidence == original.confidence
        assert len(reconstructed.legs) == len(original.legs)


class TestPortfolioTarget:
    """Tests for PortfolioTarget dataclass."""

    def test_basic_creation(self):
        """Should create portfolio target."""
        target = PortfolioTarget(
            timestamp=pd.Timestamp("2024-01-15"),
            positions={"UST_2Y": 1.5, "UST_5Y": -3.0, "UST_10Y": 1.5},
        )

        assert target.n_instruments == 3
        assert target.get_position("UST_2Y") == 1.5
        assert target.get_position("UNKNOWN") == 0.0

    def test_instruments_list(self):
        """Should list all instruments."""
        target = PortfolioTarget(
            timestamp=pd.Timestamp("2024-01-15"),
            positions={"A": 1, "B": 2, "C": 3},
        )
        assert set(target.instruments) == {"A", "B", "C"}

    def test_to_dataframe(self):
        """Should convert to DataFrame."""
        target = PortfolioTarget(
            timestamp=pd.Timestamp("2024-01-15"),
            positions={"UST_2Y": 1.0, "UST_5Y": -2.0},
        )
        df = target.to_dataframe()

        assert len(df) == 2
        assert "bond_id" in df.columns
        assert "weight" in df.columns


class TestStrategySignalBatch:
    """Tests for StrategySignalBatch dataclass."""

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for same timestamp."""
        ts = pd.Timestamp("2024-01-15 08:00")
        legs = [LegWeight("A", 1), LegWeight("B", -2), LegWeight("C", 1)]
        return [
            StrategySignal(timestamp=ts, strategy_id="strat_1", legs=legs, direction=1),
            StrategySignal(timestamp=ts, strategy_id="strat_2", legs=legs, direction=-1),
        ]

    def test_basic_creation(self, sample_signals):
        """Should create batch from signals."""
        batch = StrategySignalBatch(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            signals=sample_signals,
        )

        assert len(batch) == 2
        assert "strat_1" in batch.strategy_ids

    def test_get_signal(self, sample_signals):
        """Should retrieve signal by strategy ID."""
        batch = StrategySignalBatch(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            signals=sample_signals,
        )

        signal = batch.get_signal("strat_1")
        assert signal is not None
        assert signal.direction == 1

        missing = batch.get_signal("nonexistent")
        assert missing is None

    def test_add_signal(self, sample_signals):
        """Should add signal to batch."""
        batch = StrategySignalBatch(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            signals=sample_signals,
        )

        legs = [LegWeight("D", 1)]
        new_signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            strategy_id="strat_3",
            legs=legs,
            direction=0,
        )
        batch.add_signal(new_signal)

        assert len(batch) == 3

    def test_add_signal_timestamp_mismatch(self, sample_signals):
        """Should reject signal with different timestamp."""
        batch = StrategySignalBatch(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            signals=sample_signals,
        )

        legs = [LegWeight("D", 1)]
        bad_signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-16 08:00"),  # Different date
            strategy_id="strat_3",
            legs=legs,
            direction=0,
        )

        with pytest.raises(ValueError, match="timestamp"):
            batch.add_signal(bad_signal)


class TestFlyNameToStrategyId:
    """Tests for fly_name_to_strategy_id function."""

    def test_year_tenors(self):
        """Should format year tenors correctly."""
        assert fly_name_to_strategy_id(2, 5, 10) == "fly_2y5y10y"
        assert fly_name_to_strategy_id(5, 10, 30) == "fly_5y10y30y"

    def test_month_tenors(self):
        """Should format sub-year tenors as months."""
        assert fly_name_to_strategy_id(0.5, 2, 5) == "fly_6m2y5y"

    def test_custom_prefix(self):
        """Should use custom prefix."""
        assert fly_name_to_strategy_id(2, 5, 10, prefix="strat") == "strat_2y5y10y"


class TestCreateFlyLegs:
    """Tests for create_fly_legs function."""

    def test_default_weights(self):
        """Should create legs with default weights."""
        legs = create_fly_legs("A", "B", "C")

        assert len(legs) == 3
        assert legs[0].weight == 1.0
        assert legs[1].weight == -2.0
        assert legs[2].weight == 1.0

    def test_custom_weights(self):
        """Should use custom weights."""
        legs = create_fly_legs("A", "B", "C", front_weight=0.5, belly_weight=-1.0, back_weight=0.5)

        assert legs[0].weight == 0.5
        assert legs[1].weight == -1.0
        assert legs[2].weight == 0.5

    def test_with_tenors(self):
        """Should store tenors."""
        legs = create_fly_legs(
            "A", "B", "C",
            front_tenor=2.0, belly_tenor=5.0, back_tenor=10.0,
        )

        assert legs[0].tenor == 2.0
        assert legs[1].tenor == 5.0
        assert legs[2].tenor == 10.0


class TestSignalDfToStrategySignals:
    """Tests for signal_df_to_strategy_signals function."""

    @pytest.fixture
    def sample_signal_df(self):
        """Create sample signal DataFrame."""
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-15 08:00", periods=5, freq="h"),
            "signal": [0, 1, 1, 0, -1],
            "zscore": [-1.5, -2.5, -2.0, 0.3, 2.5],
            "strength": [0.3, 0.6, 0.5, 0.1, 0.6],
        })

    def test_converts_to_signals(self, sample_signal_df):
        """Should convert DataFrame to list of signals."""
        signals = signal_df_to_strategy_signals(
            sample_signal_df,
            front_id="UST_2Y",
            belly_id="UST_5Y",
            back_id="UST_10Y",
            tenors=(2, 5, 10),
        )

        assert len(signals) == 5
        assert all(isinstance(s, StrategySignal) for s in signals)

    def test_direction_matches(self, sample_signal_df):
        """Signal direction should match input."""
        signals = signal_df_to_strategy_signals(
            sample_signal_df,
            front_id="A", belly_id="B", back_id="C",
        )

        assert signals[0].direction == 0
        assert signals[1].direction == 1
        assert signals[4].direction == -1

    def test_strategy_id_from_tenors(self, sample_signal_df):
        """Should generate strategy ID from tenors."""
        signals = signal_df_to_strategy_signals(
            sample_signal_df,
            front_id="A", belly_id="B", back_id="C",
            tenors=(2, 5, 10),
        )

        assert signals[0].strategy_id == "fly_2y5y10y"

    def test_includes_zscore(self, sample_signal_df):
        """Should include z-score as signal_value."""
        signals = signal_df_to_strategy_signals(
            sample_signal_df,
            front_id="A", belly_id="B", back_id="C",
        )

        assert signals[1].signal_value == -2.5

    def test_includes_confidence(self, sample_signal_df):
        """Should include strength as confidence."""
        signals = signal_df_to_strategy_signals(
            sample_signal_df,
            front_id="A", belly_id="B", back_id="C",
        )

        assert signals[1].confidence == 0.6


class TestBatchSignalsByTimestamp:
    """Tests for batch_signals_by_timestamp function."""

    def test_groups_by_timestamp(self):
        """Should group signals by timestamp."""
        legs = [LegWeight("A", 1)]
        signals = [
            StrategySignal(pd.Timestamp("2024-01-15 08:00"), "s1", legs, direction=1),
            StrategySignal(pd.Timestamp("2024-01-15 08:00"), "s2", legs, direction=-1),
            StrategySignal(pd.Timestamp("2024-01-15 09:00"), "s1", legs, direction=0),
        ]

        batches = batch_signals_by_timestamp(signals)

        assert len(batches) == 2
        assert len(batches[0]) == 2  # Two signals at 08:00
        assert len(batches[1]) == 1  # One signal at 09:00


class TestFilterFunctions:
    """Tests for signal filtering functions."""

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals."""
        legs = [LegWeight("A", 1)]
        return [
            StrategySignal(pd.Timestamp("2024-01-15"), "s1", legs, direction=1),
            StrategySignal(pd.Timestamp("2024-01-15"), "s2", legs, direction=-1),
            StrategySignal(pd.Timestamp("2024-01-15"), "s3", legs, direction=0),
        ]

    def test_filter_by_direction(self, sample_signals):
        """Should filter by direction."""
        long_only = filter_signals_by_direction(sample_signals, direction=1)
        assert len(long_only) == 1
        assert long_only[0].strategy_id == "s1"

    def test_get_active_signals(self, sample_signals):
        """Should return non-flat signals."""
        active = get_active_signals(sample_signals)
        assert len(active) == 2
        assert all(not s.is_flat for s in active)


class TestSignalsToDataframe:
    """Tests for signals_to_dataframe function."""

    def test_converts_to_dataframe(self):
        """Should convert signals to DataFrame."""
        legs = [
            LegWeight("A", 1.0, tenor=2.0),
            LegWeight("B", -2.0, tenor=5.0),
        ]
        signals = [
            StrategySignal(
                pd.Timestamp("2024-01-15"),
                "test",
                legs,
                direction=1,
                signal_value=-2.0,
            ),
        ]

        df = signals_to_dataframe(signals)

        assert len(df) == 1
        assert "strategy_id" in df.columns
        assert "direction" in df.columns
        assert "signal_value" in df.columns
        assert "leg_0_id" in df.columns
        assert df["leg_0_id"].iloc[0] == "A"


class TestSerialization:
    """Tests for serialization (to_dict/from_dict) methods."""

    def test_strategy_signal_roundtrip_minimal(self):
        """Should serialize and deserialize minimal signal."""
        legs = [LegWeight("UST_5Y", -2.0)]
        original = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=legs,
            direction=0,
        )

        d = original.to_dict()
        reconstructed = StrategySignal.from_dict(d)

        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.strategy_id == original.strategy_id
        assert reconstructed.direction == original.direction
        assert reconstructed.signal_value is None
        assert reconstructed.target_gross_dv01 is None
        assert reconstructed.confidence is None
        assert reconstructed.metadata == {}

    def test_strategy_signal_roundtrip_full(self):
        """Should serialize and deserialize signal with all fields."""
        legs = [
            LegWeight("UST_2Y", 1.0, tenor=2.0),
            LegWeight("UST_5Y", -2.0, tenor=5.0),
            LegWeight("UST_10Y", 1.0, tenor=10.0),
        ]
        original = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15 08:30:00"),
            strategy_id="fly_2y5y10y",
            legs=legs,
            direction=-1,
            signal_value=2.5,
            target_gross_dv01=15000.0,
            confidence=0.75,
            metadata={"zscore": 2.5, "entry_reason": "mean_reversion"},
        )

        d = original.to_dict()
        reconstructed = StrategySignal.from_dict(d)

        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.strategy_id == original.strategy_id
        assert reconstructed.direction == original.direction
        assert reconstructed.signal_value == original.signal_value
        assert reconstructed.target_gross_dv01 == original.target_gross_dv01
        assert reconstructed.confidence == original.confidence
        assert reconstructed.metadata == original.metadata
        assert len(reconstructed.legs) == 3
        assert reconstructed.legs[0].bond_id == "UST_2Y"
        assert reconstructed.legs[0].tenor == 2.0
        assert reconstructed.legs[1].weight == -2.0

    def test_strategy_signal_to_dict_structure(self):
        """Should produce expected dict structure."""
        legs = [LegWeight("A", 1.0, tenor=2.0)]
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="test",
            legs=legs,
            direction=1,
        )

        d = signal.to_dict()

        assert "timestamp" in d
        assert "strategy_id" in d
        assert "legs" in d
        assert "direction" in d
        assert isinstance(d["legs"], list)
        assert d["legs"][0] == ("A", 1.0, 2.0)

    def test_strategy_signal_from_dict_without_tenor(self):
        """Should handle legs without tenor in serialized form."""
        d = {
            "timestamp": pd.Timestamp("2024-01-15"),
            "strategy_id": "test",
            "legs": [("A", 1.0), ("B", -2.0)],  # No tenor
            "direction": 1,
        }

        signal = StrategySignal.from_dict(d)

        assert len(signal.legs) == 2
        assert signal.legs[0].tenor is None
        assert signal.legs[1].tenor is None

    def test_portfolio_target_to_dict(self):
        """Should serialize PortfolioTarget."""
        target = PortfolioTarget(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            positions={"UST_2Y": 1.5, "UST_5Y": -3.0, "UST_10Y": 1.5},
            strategy_contributions={
                "fly_2y5y10y": {"UST_2Y": 1.0, "UST_5Y": -2.0, "UST_10Y": 1.0},
                "fly_5y10y30y": {"UST_5Y": -1.0, "UST_10Y": 0.5, "UST_30Y": 0.5},
            },
            total_gross_dv01=20000.0,
            total_net_dv01=500.0,
            metadata={"risk_limit": "ok"},
        )

        d = target.to_dict()

        assert d["timestamp"] == pd.Timestamp("2024-01-15 08:00")
        assert d["positions"]["UST_2Y"] == 1.5
        assert d["positions"]["UST_5Y"] == -3.0
        assert "fly_2y5y10y" in d["strategy_contributions"]
        assert d["total_gross_dv01"] == 20000.0
        assert d["total_net_dv01"] == 500.0
        assert d["metadata"]["risk_limit"] == "ok"

    def test_signal_batch_to_dataframe(self):
        """Should convert batch to DataFrame."""
        ts = pd.Timestamp("2024-01-15 08:00")
        legs = [LegWeight("A", 1), LegWeight("B", -2), LegWeight("C", 1)]
        signals = [
            StrategySignal(ts, "strat_1", legs, direction=1, signal_value=-2.0),
            StrategySignal(ts, "strat_2", legs, direction=-1, signal_value=1.5),
        ]
        batch = StrategySignalBatch(timestamp=ts, signals=signals)

        df = batch.to_dataframe()

        assert len(df) == 2
        assert list(df["strategy_id"]) == ["strat_1", "strat_2"]
        assert list(df["direction"]) == [1, -1]
        assert list(df["signal_value"]) == [-2.0, 1.5]

    def test_multiple_signals_to_dataframe(self):
        """Should convert list of signals to DataFrame with leg details."""
        legs1 = [LegWeight("A", 1.0, 2.0), LegWeight("B", -2.0, 5.0)]
        legs2 = [LegWeight("X", 0.5), LegWeight("Y", -1.0), LegWeight("Z", 0.5)]

        signals = [
            StrategySignal(
                pd.Timestamp("2024-01-15"),
                "strat_1",
                legs1,
                direction=1,
                target_gross_dv01=10000,
            ),
            StrategySignal(
                pd.Timestamp("2024-01-16"),
                "strat_2",
                legs2,
                direction=-1,
                confidence=0.8,
            ),
        ]

        df = signals_to_dataframe(signals)

        assert len(df) == 2
        assert df.iloc[0]["n_legs"] == 2
        assert df.iloc[1]["n_legs"] == 3
        assert df.iloc[0]["leg_ids"] == "A,B"
        assert df.iloc[1]["leg_ids"] == "X,Y,Z"
        assert df.iloc[0]["leg_0_tenor"] == 2.0
        assert pd.isna(df.iloc[1]["leg_0_tenor"])


class TestMinimalExample:
    """Minimal end-to-end example demonstrating portfolio interface usage."""

    def test_minimal_fly_signal_workflow(self):
        """Demonstrate creating signals for a single fly strategy."""
        # 1. Create fly legs for 2y5y10y butterfly
        legs = create_fly_legs(
            front_id="UST_2Y",
            belly_id="UST_5Y",
            back_id="UST_10Y",
            front_tenor=2.0,
            belly_tenor=5.0,
            back_tenor=10.0,
        )

        # Verify standard fly weights
        assert legs[0].weight == 1.0  # Long front
        assert legs[1].weight == -2.0  # Short belly
        assert legs[2].weight == 1.0  # Long back

        # 2. Create a strategy signal (fly is cheap, go long)
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            strategy_id=fly_name_to_strategy_id(2, 5, 10),
            legs=legs,
            direction=1,  # Long the fly
            signal_value=-2.5,  # Z-score indicates cheapness
            target_gross_dv01=10000,
        )

        assert signal.strategy_id == "fly_2y5y10y"
        assert signal.is_long
        assert not signal.is_flat

        # 3. Get scaled weights (direction=1 means weights unchanged)
        weights = signal.scaled_weights()
        assert weights["UST_2Y"] == 1.0
        assert weights["UST_5Y"] == -2.0
        assert weights["UST_10Y"] == 1.0

    def test_minimal_multi_strategy_workflow(self):
        """Demonstrate combining signals from multiple fly strategies."""
        ts = pd.Timestamp("2024-01-15 08:00")

        # Strategy 1: 2y5y10y fly - long signal
        legs_2510 = create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y")
        signal_2510 = StrategySignal(
            timestamp=ts,
            strategy_id="fly_2y5y10y",
            legs=legs_2510,
            direction=1,
            signal_value=-2.0,
            target_gross_dv01=10000,
        )

        # Strategy 2: 5y10y30y fly - short signal
        legs_51030 = create_fly_legs("UST_5Y", "UST_10Y", "UST_30Y")
        signal_51030 = StrategySignal(
            timestamp=ts,
            strategy_id="fly_5y10y30y",
            legs=legs_51030,
            direction=-1,
            signal_value=1.8,
            target_gross_dv01=8000,
        )

        # Strategy 3: 2y5y10y fly - flat (no signal)
        signal_flat = StrategySignal(
            timestamp=ts,
            strategy_id="fly_3y7y10y",
            legs=create_fly_legs("UST_3Y", "UST_7Y", "UST_10Y"),
            direction=0,
        )

        # Batch signals by timestamp
        all_signals = [signal_2510, signal_51030, signal_flat]
        batches = batch_signals_by_timestamp(all_signals)

        assert len(batches) == 1  # All same timestamp
        assert len(batches[0]) == 3

        # Filter to active signals only
        active = get_active_signals(all_signals)
        assert len(active) == 2
        assert all(not s.is_flat for s in active)

        # Filter by direction
        long_only = filter_signals_by_direction(all_signals, direction=1)
        assert len(long_only) == 1
        assert long_only[0].strategy_id == "fly_2y5y10y"

    def test_minimal_signal_df_conversion(self):
        """Demonstrate converting a signal DataFrame to StrategySignal list."""
        # Simulate output from a z-score signal generator
        signal_df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-15 08:00", periods=3, freq="h"),
            "signal": [0, 1, -1],  # Flat, Long, Short
            "zscore": [0.5, -2.2, 2.1],
            "strength": [0.2, 0.7, 0.65],
        })

        # Convert to StrategySignal list
        signals = signal_df_to_strategy_signals(
            signal_df,
            front_id="UST_2Y",
            belly_id="UST_5Y",
            back_id="UST_10Y",
            tenors=(2, 5, 10),
            target_gross_dv01=10000,
        )

        assert len(signals) == 3
        assert signals[0].is_flat
        assert signals[1].is_long
        assert signals[2].is_short

        # Check signal values and confidence
        assert signals[1].signal_value == -2.2
        assert signals[1].confidence == 0.7
        assert signals[1].target_gross_dv01 == 10000

        # Convert to DataFrame for analysis
        df = signals_to_dataframe(signals)
        assert len(df) == 3
        assert "strategy_id" in df.columns
        assert all(df["strategy_id"] == "fly_2y5y10y")

    def test_minimal_serialization_workflow(self):
        """Demonstrate serializing and deserializing signals."""
        # Create a signal
        original = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15 08:00"),
            strategy_id="fly_2y5y10y",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y", front_tenor=2, belly_tenor=5, back_tenor=10),
            direction=1,
            signal_value=-2.5,
            target_gross_dv01=10000,
            confidence=0.8,
            metadata={"entry_reason": "mean_reversion", "zscore": -2.5},
        )

        # Serialize to dict (could be saved to JSON/database)
        serialized = original.to_dict()

        # Verify serialized structure
        assert isinstance(serialized, dict)
        assert serialized["strategy_id"] == "fly_2y5y10y"
        assert serialized["direction"] == 1
        assert len(serialized["legs"]) == 3

        # Deserialize back
        restored = StrategySignal.from_dict(serialized)

        # Verify roundtrip
        assert restored.strategy_id == original.strategy_id
        assert restored.direction == original.direction
        assert restored.signal_value == original.signal_value
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata
        assert len(restored.legs) == len(original.legs)


class TestAggregation:
    """Tests for signals_to_bond_targets aggregation."""

    def test_single_strategy_basic(self):
        """Should aggregate a single strategy signal."""
        legs = create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y")
        signal = StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="fly_2y5y10y",
            legs=legs,
            direction=1,
            target_gross_dv01=10000,
        )

        result = signals_to_bond_targets([signal])

        assert isinstance(result, AggregationResult)
        assert len(result.target_dv01) == 3
        # Fly weights [1, -2, 1], gross = 4, so each unit = 10000/4 = 2500
        assert result.target_dv01["UST_2Y"] == pytest.approx(2500)
        assert result.target_dv01["UST_5Y"] == pytest.approx(-5000)
        assert result.target_dv01["UST_10Y"] == pytest.approx(2500)
        assert result.gross_dv01 == pytest.approx(10000)
        assert result.net_dv01 == pytest.approx(0)  # Fly is DV01 neutral

    def test_two_flies_netting_reduces_overlap(self):
        """Two flies sharing a bond should net overlapping exposures.

        fly_2y5y10y: long 2Y, short 5Y, long 10Y
        fly_5y10y30y: long 5Y, short 10Y, long 30Y

        With equal weighting, the 5Y and 10Y exposures partially offset.
        """
        ts = pd.Timestamp("2024-01-15")

        # Fly 1: 2y5y10y - long the fly (buy wings, sell belly)
        signal_1 = StrategySignal(
            timestamp=ts,
            strategy_id="fly_2y5y10y",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y"),
            direction=1,
            target_gross_dv01=10000,
        )

        # Fly 2: 5y10y30y - long the fly
        signal_2 = StrategySignal(
            timestamp=ts,
            strategy_id="fly_5y10y30y",
            legs=create_fly_legs("UST_5Y", "UST_10Y", "UST_30Y"),
            direction=1,
            target_gross_dv01=10000,
        )

        # Aggregate with equal weights
        result = signals_to_bond_targets(
            [signal_1, signal_2],
            SizingRules(strategy_weights="equal"),
        )

        # Each fly gets 50% weight
        # fly_2y5y10y: 2Y=+1250, 5Y=-2500, 10Y=+1250
        # fly_5y10y30y: 5Y=+1250, 10Y=-2500, 30Y=+1250
        # After netting:
        #   2Y = +1250
        #   5Y = -2500 + 1250 = -1250 (reduced!)
        #   10Y = +1250 - 2500 = -1250 (flipped!)
        #   30Y = +1250

        assert result.target_dv01["UST_2Y"] == pytest.approx(1250)
        assert result.target_dv01["UST_5Y"] == pytest.approx(-1250)  # Netted down
        assert result.target_dv01["UST_10Y"] == pytest.approx(-1250)  # Netted and flipped
        assert result.target_dv01["UST_30Y"] == pytest.approx(1250)

        # Gross is less than sum of individual grosses due to netting
        assert result.gross_dv01 == pytest.approx(5000)  # |1250| + |1250| + |1250| + |1250|

        # Verify strategy contributions are tracked
        assert "fly_2y5y10y" in result.strategy_contributions
        assert "fly_5y10y30y" in result.strategy_contributions

    def test_two_flies_opposite_directions_full_netting(self):
        """Same fly, opposite directions should fully offset shared bonds."""
        ts = pd.Timestamp("2024-01-15")

        signal_long = StrategySignal(
            timestamp=ts,
            strategy_id="fly_long",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y"),
            direction=1,
            target_gross_dv01=10000,
        )

        signal_short = StrategySignal(
            timestamp=ts,
            strategy_id="fly_short",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y"),
            direction=-1,
            target_gross_dv01=10000,
        )

        result = signals_to_bond_targets([signal_long, signal_short])

        # Equal weights, opposite directions = full offset
        assert result.target_dv01["UST_2Y"] == pytest.approx(0)
        assert result.target_dv01["UST_5Y"] == pytest.approx(0)
        assert result.target_dv01["UST_10Y"] == pytest.approx(0)
        assert result.gross_dv01 == pytest.approx(0)

    def test_flat_signals_excluded(self):
        """Flat signals should not contribute."""
        ts = pd.Timestamp("2024-01-15")
        legs = create_fly_legs("A", "B", "C")

        signal_active = StrategySignal(ts, "active", legs, direction=1, target_gross_dv01=10000)
        signal_flat = StrategySignal(ts, "flat", legs, direction=0)

        result = signals_to_bond_targets([signal_active, signal_flat])

        # Only active signal contributes
        assert result.n_strategies == 1
        assert "active" in result.strategy_contributions
        assert "flat" not in result.strategy_contributions

    def test_empty_signals_returns_empty(self):
        """Empty signal list should return empty result."""
        result = signals_to_bond_targets([])

        assert len(result.target_dv01) == 0
        assert result.gross_dv01 == 0
        assert result.net_dv01 == 0


class TestSizingRulesConstraints:
    """Tests for SizingRules constraints."""

    @pytest.fixture
    def sample_signal(self):
        """Create a sample fly signal."""
        return StrategySignal(
            timestamp=pd.Timestamp("2024-01-15"),
            strategy_id="fly_2y5y10y",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y"),
            direction=1,
            target_gross_dv01=20000,  # Gross = 20k
        )

    def test_per_bond_dv01_cap(self, sample_signal):
        """Should cap individual bond DV01."""
        # Without cap, 5Y would have -10000 DV01
        rules = SizingRules(per_bond_dv01_cap=3000)
        result = signals_to_bond_targets([sample_signal], rules)

        # 5Y capped at -3000 (was -10000)
        assert result.target_dv01["UST_5Y"] == pytest.approx(-3000)
        # Wings capped at 3000 (was 5000)
        assert result.target_dv01["UST_2Y"] == pytest.approx(3000)
        assert result.target_dv01["UST_10Y"] == pytest.approx(3000)
        assert "per_bond_dv01_cap" in result.constraints_applied

    def test_gross_dv01_budget(self, sample_signal):
        """Should scale down to meet gross budget."""
        rules = SizingRules(gross_dv01_budget=10000)
        result = signals_to_bond_targets([sample_signal], rules)

        # Original gross was 20000, scaled to 10000 (factor = 0.5)
        assert result.gross_dv01 == pytest.approx(10000)
        assert result.scale_factor == pytest.approx(0.5)
        assert "gross_dv01_budget" in result.constraints_applied

        # Individual positions scaled by 0.5
        assert result.target_dv01["UST_5Y"] == pytest.approx(-5000)

    def test_gross_budget_not_binding(self, sample_signal):
        """Budget not binding if gross is under budget."""
        rules = SizingRules(gross_dv01_budget=50000)
        result = signals_to_bond_targets([sample_signal], rules)

        assert result.scale_factor == 1.0
        assert "gross_dv01_budget" not in result.constraints_applied

    def test_net_dv01_target(self):
        """Should adjust positions to hit net target."""
        ts = pd.Timestamp("2024-01-15")

        # Fly is DV01 neutral (net = 0), but we want net = 1000
        signal = StrategySignal(
            timestamp=ts,
            strategy_id="fly",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y"),
            direction=1,
            target_gross_dv01=10000,
        )

        rules = SizingRules(net_dv01_target=1000)
        result = signals_to_bond_targets([signal], rules)

        assert result.net_dv01 == pytest.approx(1000)
        assert "net_dv01_target" in result.constraints_applied

    def test_combined_constraints(self, sample_signal):
        """Multiple constraints applied in order."""
        rules = SizingRules(
            per_bond_dv01_cap=8000,
            gross_dv01_budget=15000,
        )
        result = signals_to_bond_targets([sample_signal], rules)

        # First per_bond_cap clips 5Y from -10000 to -8000
        # Then gross_budget scales if needed
        assert abs(result.target_dv01["UST_5Y"]) <= 8000
        assert result.gross_dv01 <= 15000


class TestStrategyWeights:
    """Tests for strategy weighting options."""

    @pytest.fixture
    def two_signals(self):
        """Create two fly signals."""
        ts = pd.Timestamp("2024-01-15")
        return [
            StrategySignal(
                timestamp=ts,
                strategy_id="strat_A",
                legs=create_fly_legs("A", "B", "C"),
                direction=1,
                target_gross_dv01=10000,
            ),
            StrategySignal(
                timestamp=ts,
                strategy_id="strat_B",
                legs=create_fly_legs("X", "Y", "Z"),
                direction=1,
                target_gross_dv01=10000,
            ),
        ]

    def test_equal_weights(self, two_signals):
        """Equal weighting divides budget equally."""
        rules = SizingRules(strategy_weights="equal")
        result = signals_to_bond_targets(two_signals, rules)

        # Each strategy gets 50%, so gross contribution = 5000 each
        # Total gross = 10000
        assert result.gross_dv01 == pytest.approx(10000)

    def test_dict_weights(self, two_signals):
        """Dict weights apply specified ratios."""
        rules = SizingRules(strategy_weights={"strat_A": 0.8, "strat_B": 0.2})
        result = signals_to_bond_targets(two_signals, rules)

        # strat_A: 8000, strat_B: 2000
        contrib_A = sum(abs(v) for v in result.strategy_contributions["strat_A"].values())
        contrib_B = sum(abs(v) for v in result.strategy_contributions["strat_B"].values())

        assert contrib_A == pytest.approx(8000)
        assert contrib_B == pytest.approx(2000)

    def test_dict_weights_missing_strategy(self, two_signals):
        """Missing strategy in dict weights gets zero."""
        rules = SizingRules(strategy_weights={"strat_A": 1.0})  # strat_B not listed
        result = signals_to_bond_targets(two_signals, rules)

        assert "strat_A" in result.strategy_contributions
        # strat_B has zero weight, not in contributions
        assert result.strategy_contributions.get("strat_B") is None or \
               sum(abs(v) for v in result.strategy_contributions.get("strat_B", {}).values()) == 0

    def test_callable_weights(self, two_signals):
        """Callable weights for time-varying allocation."""

        def weight_fn(strategy_id, timestamp):
            if strategy_id == "strat_A":
                return 0.7
            return 0.3

        rules = SizingRules(strategy_weights=weight_fn)
        result = signals_to_bond_targets(two_signals, rules)

        contrib_A = sum(abs(v) for v in result.strategy_contributions["strat_A"].values())
        contrib_B = sum(abs(v) for v in result.strategy_contributions["strat_B"].values())

        assert contrib_A == pytest.approx(7000)
        assert contrib_B == pytest.approx(3000)


class TestConfidenceFiltering:
    """Tests for confidence-based signal filtering."""

    def test_min_confidence_filter(self):
        """Should exclude signals below confidence threshold."""
        ts = pd.Timestamp("2024-01-15")
        legs = create_fly_legs("A", "B", "C")

        signals = [
            StrategySignal(ts, "high_conf", legs, direction=1, confidence=0.8, target_gross_dv01=10000),
            StrategySignal(ts, "low_conf", legs, direction=1, confidence=0.3, target_gross_dv01=10000),
            StrategySignal(ts, "no_conf", legs, direction=1, target_gross_dv01=10000),  # None confidence
        ]

        rules = SizingRules(min_signal_confidence=0.5)
        result = signals_to_bond_targets(signals, rules)

        # high_conf and no_conf included, low_conf excluded
        assert "high_conf" in result.strategy_contributions
        assert "no_conf" in result.strategy_contributions
        assert "low_conf" in result.excluded_strategies


class TestAggregationResult:
    """Tests for AggregationResult methods."""

    def test_to_portfolio_target(self):
        """Should convert to PortfolioTarget."""
        target_dv01 = pd.Series({"A": 1000, "B": -2000, "C": 1000})
        result = AggregationResult(
            target_dv01=target_dv01,
            strategy_contributions={"strat_1": {"A": 1000, "B": -2000, "C": 1000}},
            gross_dv01=4000,
            net_dv01=0,
            scale_factor=0.8,
            constraints_applied=["gross_dv01_budget"],
        )

        pt = result.to_portfolio_target(pd.Timestamp("2024-01-15"))

        assert isinstance(pt, PortfolioTarget)
        assert pt.positions["A"] == 1000
        assert pt.total_gross_dv01 == 4000
        assert pt.total_net_dv01 == 0
        assert pt.metadata["scale_factor"] == 0.8

    def test_n_bonds(self):
        """Should count non-zero bonds."""
        result = AggregationResult(
            target_dv01=pd.Series({"A": 1000, "B": 0, "C": -500}),
        )
        assert result.n_bonds == 2


class TestTurnoverCalculation:
    """Tests for turnover and position change utilities."""

    def test_compute_turnover(self):
        """Should compute total absolute change."""
        current = pd.Series({"A": 1000, "B": -2000, "C": 500})
        previous = pd.Series({"A": 500, "B": -1500, "C": 500})

        turnover = compute_turnover(current, previous)

        # |1000-500| + |-2000-(-1500)| + |500-500| = 500 + 500 + 0 = 1000
        assert turnover == pytest.approx(1000)

    def test_compute_turnover_new_position(self):
        """Should handle new positions not in previous."""
        current = pd.Series({"A": 1000, "B": 2000})
        previous = pd.Series({"A": 500})

        turnover = compute_turnover(current, previous)

        # |1000-500| + |2000-0| = 500 + 2000 = 2500
        assert turnover == pytest.approx(2500)

    def test_compute_turnover_closed_position(self):
        """Should handle positions closed (in previous, not current)."""
        current = pd.Series({"A": 1000})
        previous = pd.Series({"A": 500, "B": 2000})

        turnover = compute_turnover(current, previous)

        # |1000-500| + |0-2000| = 500 + 2000 = 2500
        assert turnover == pytest.approx(2500)

    def test_compute_position_changes(self):
        """Should compute detailed position changes."""
        current = pd.Series({"A": 1000, "B": -2000})
        previous = pd.Series({"A": 500, "B": -1000})

        df = compute_position_changes(current, previous)

        assert df.loc["A", "previous"] == 500
        assert df.loc["A", "current"] == 1000
        assert df.loc["A", "change"] == 500
        assert df.loc["B", "change"] == -1000


class TestAggregateSignalBatch:
    """Tests for aggregate_signal_batch convenience function."""

    def test_returns_portfolio_target(self):
        """Should return PortfolioTarget directly."""
        ts = pd.Timestamp("2024-01-15")
        signal = StrategySignal(
            timestamp=ts,
            strategy_id="test",
            legs=create_fly_legs("A", "B", "C"),
            direction=1,
            target_gross_dv01=10000,
        )

        target = aggregate_signal_batch([signal])

        assert isinstance(target, PortfolioTarget)
        assert target.timestamp == ts
        assert "A" in target.positions

    def test_empty_signals_raises(self):
        """Should raise on empty signal list."""
        with pytest.raises(ValueError, match="No signals"):
            aggregate_signal_batch([])


class TestSimulateRebalance:
    """Tests for simulate_rebalance function."""

    def test_basic_rebalance(self):
        """Should generate trades to move from current to target."""
        current = pd.Series({"A": 1000, "B": -2000, "C": 1000})
        target = pd.Series({"A": 1500, "B": -1500, "C": 500})
        ts = pd.Timestamp("2024-01-15")

        trades, new_holdings = simulate_rebalance(current, target, ts)

        assert len(trades) == 3
        # Check trade directions
        trade_dict = {t.bond_id: t.dv01_change for t in trades}
        assert trade_dict["A"] == pytest.approx(500)  # Buy 500 DV01
        assert trade_dict["B"] == pytest.approx(500)  # Buy back 500 DV01
        assert trade_dict["C"] == pytest.approx(-500)  # Sell 500 DV01

        # Check new holdings match target
        assert new_holdings["A"] == pytest.approx(1500)
        assert new_holdings["B"] == pytest.approx(-1500)
        assert new_holdings["C"] == pytest.approx(500)

    def test_rebalance_with_costs(self):
        """Should compute transaction costs."""
        current = pd.Series({"A": 0})
        target = pd.Series({"A": 10000})
        ts = pd.Timestamp("2024-01-15")

        config = PortfolioBacktestConfig(cost_bps=1.0)
        trades, _ = simulate_rebalance(current, target, ts, config=config)

        assert len(trades) == 1
        # Notional = 10000 * 10000 = 100M, cost = 100M * 1bp = 10000
        assert trades[0].cost == pytest.approx(10000)

    def test_min_trade_dv01_filter(self):
        """Should skip trades below minimum."""
        current = pd.Series({"A": 1000, "B": 1005})  # Small diff on B
        target = pd.Series({"A": 2000, "B": 1000})
        ts = pd.Timestamp("2024-01-15")

        rule = RebalanceRule(min_trade_dv01=100)
        trades, new_holdings = simulate_rebalance(
            current, target, ts, rebalance_rule=rule
        )

        # Only A should trade (1000 change), B skipped (5 change < 100)
        assert len(trades) == 1
        assert trades[0].bond_id == "A"
        assert new_holdings["B"] == 1005  # Unchanged

    def test_new_position(self):
        """Should handle new positions not in current."""
        current = pd.Series({"A": 1000})
        target = pd.Series({"A": 1000, "B": 2000})
        ts = pd.Timestamp("2024-01-15")

        trades, new_holdings = simulate_rebalance(current, target, ts)

        assert len(trades) == 1
        assert trades[0].bond_id == "B"
        assert trades[0].dv01_change == 2000
        assert new_holdings["B"] == 2000

    def test_close_position(self):
        """Should handle closing positions."""
        current = pd.Series({"A": 1000, "B": 2000})
        target = pd.Series({"A": 1000})  # Close B
        ts = pd.Timestamp("2024-01-15")

        trades, new_holdings = simulate_rebalance(current, target, ts)

        assert len(trades) == 1
        assert trades[0].bond_id == "B"
        assert trades[0].dv01_change == -2000
        assert new_holdings["B"] == 0


class TestComputePnlFromPrices:
    """Tests for compute_pnl_from_prices function."""

    def test_basic_pnl(self):
        """Should compute P&L from price changes."""
        holdings = pd.Series({"A": 1000, "B": -500})
        prices_t0 = pd.Series({"A": 100.0, "B": 100.0})
        prices_t1 = pd.Series({"A": 101.0, "B": 99.0})  # A up 1%, B down 1%

        result = compute_pnl_from_prices(holdings, prices_t0, prices_t1)

        # A: long 1000 DV01, price up 1% = +100bps, P&L = 1000 * 100 / 100 = 1000
        # B: short 500 DV01, price down 1% = -100bps, P&L = -500 * -100 / 100 = 500
        assert result["pnl_by_bond"]["A"] == pytest.approx(1000)
        assert result["pnl_by_bond"]["B"] == pytest.approx(500)
        assert result["total_pnl"] == pytest.approx(1500)

    def test_pnl_with_loss(self):
        """Should compute negative P&L correctly."""
        holdings = pd.Series({"A": 1000})
        prices_t0 = pd.Series({"A": 100.0})
        prices_t1 = pd.Series({"A": 99.0})  # A down 1%

        result = compute_pnl_from_prices(holdings, prices_t0, prices_t1)

        # Long position loses when price drops
        assert result["total_pnl"] == pytest.approx(-1000)

    def test_zero_holdings(self):
        """Should return zero P&L for zero holdings."""
        holdings = pd.Series({"A": 0})
        prices_t0 = pd.Series({"A": 100.0})
        prices_t1 = pd.Series({"A": 110.0})

        result = compute_pnl_from_prices(holdings, prices_t0, prices_t1)

        assert result["total_pnl"] == pytest.approx(0)


class TestRunPortfolioBacktest:
    """Tests for run_portfolio_backtest function."""

    @pytest.fixture
    def synthetic_data(self):
        """Create small synthetic panel for testing."""
        # 5 timestamps, 3 bonds
        timestamps = pd.date_range("2024-01-15 08:00", periods=5, freq="h")
        bonds = ["UST_2Y", "UST_5Y", "UST_10Y"]

        # Build price panel (long format)
        records = []
        base_prices = {"UST_2Y": 100.0, "UST_5Y": 100.0, "UST_10Y": 100.0}

        for i, ts in enumerate(timestamps):
            for bond in bonds:
                # Simple price evolution
                price = base_prices[bond] * (1 + 0.001 * i * (1 if bond != "UST_5Y" else -1))
                records.append({
                    "datetime": ts,
                    "bond_id": bond,
                    "price": price,
                })

        price_panel = pd.DataFrame(records)

        # Build target DV01 (rebalance at first and third timestamps)
        target_records = [
            # First rebalance: establish fly position
            {"datetime": timestamps[0], "bond_id": "UST_2Y", "target_dv01": 2500},
            {"datetime": timestamps[0], "bond_id": "UST_5Y", "target_dv01": -5000},
            {"datetime": timestamps[0], "bond_id": "UST_10Y", "target_dv01": 2500},
            # Second rebalance: reduce position
            {"datetime": timestamps[2], "bond_id": "UST_2Y", "target_dv01": 1250},
            {"datetime": timestamps[2], "bond_id": "UST_5Y", "target_dv01": -2500},
            {"datetime": timestamps[2], "bond_id": "UST_10Y", "target_dv01": 1250},
        ]
        target_dv01 = pd.DataFrame(target_records)

        return price_panel, target_dv01, timestamps

    def test_basic_backtest(self, synthetic_data):
        """Should run backtest and produce results."""
        price_panel, target_dv01, timestamps = synthetic_data

        result = run_portfolio_backtest(
            price_panel=price_panel,
            target_dv01=target_dv01,
        )

        assert isinstance(result, PortfolioBacktestResult)
        assert len(result.pnl_df) == 5  # One row per timestamp
        assert "gross_pnl" in result.pnl_df.columns
        assert "net_pnl" in result.pnl_df.columns
        assert "cumulative_pnl" in result.pnl_df.columns
        assert "gross_dv01" in result.pnl_df.columns

    def test_backtest_with_costs(self, synthetic_data):
        """Should deduct transaction costs."""
        price_panel, target_dv01, timestamps = synthetic_data

        config = PortfolioBacktestConfig(cost_bps=1.0)
        result = run_portfolio_backtest(
            price_panel=price_panel,
            target_dv01=target_dv01,
            config=config,
        )

        # Should have costs from two rebalances
        assert result.total_cost > 0
        assert len(result.trades) > 0

    def test_holdings_tracked(self, synthetic_data):
        """Should track holdings over time."""
        price_panel, target_dv01, timestamps = synthetic_data

        result = run_portfolio_backtest(
            price_panel=price_panel,
            target_dv01=target_dv01,
        )

        assert result.holdings_df is not None
        assert len(result.holdings_df) == 5

        # Check first row has initial positions
        first_row = result.holdings_df.iloc[0]
        assert first_row["dv01_UST_2Y"] == pytest.approx(2500)
        assert first_row["dv01_UST_5Y"] == pytest.approx(-5000)

        # Check third row (after second rebalance)
        third_row = result.holdings_df.iloc[2]
        assert third_row["dv01_UST_2Y"] == pytest.approx(1250)
        assert third_row["dv01_UST_5Y"] == pytest.approx(-2500)

    def test_gross_dv01_tracked(self, synthetic_data):
        """Should track gross DV01 exposure."""
        price_panel, target_dv01, timestamps = synthetic_data

        result = run_portfolio_backtest(
            price_panel=price_panel,
            target_dv01=target_dv01,
        )

        # First period: gross = |2500| + |-5000| + |2500| = 10000
        assert result.pnl_df.iloc[0]["gross_dv01"] == pytest.approx(10000)

        # After second rebalance: gross = |1250| + |-2500| + |1250| = 5000
        assert result.pnl_df.iloc[2]["gross_dv01"] == pytest.approx(5000)

    def test_net_dv01_tracked(self, synthetic_data):
        """Should track net DV01 exposure."""
        price_panel, target_dv01, timestamps = synthetic_data

        result = run_portfolio_backtest(
            price_panel=price_panel,
            target_dv01=target_dv01,
        )

        # Fly is DV01 neutral: 2500 - 5000 + 2500 = 0
        assert result.pnl_df.iloc[0]["net_dv01"] == pytest.approx(0)

    def test_summary_computed(self, synthetic_data):
        """Should compute summary statistics."""
        price_panel, target_dv01, timestamps = synthetic_data

        config = PortfolioBacktestConfig(cost_bps=1.0)
        result = run_portfolio_backtest(
            price_panel=price_panel,
            target_dv01=target_dv01,
            config=config,
        )

        assert "total_pnl" in result.summary
        assert "total_cost" in result.summary
        assert "n_trades" in result.summary
        assert "n_rebalances" in result.summary


class TestRunPortfolioBacktestFromTargets:
    """Tests for run_portfolio_backtest_from_targets convenience function."""

    def test_from_portfolio_targets(self):
        """Should run backtest from PortfolioTarget list."""
        # Create price panel
        timestamps = pd.date_range("2024-01-15 08:00", periods=3, freq="h")
        records = []
        for ts in timestamps:
            records.append({"datetime": ts, "bond_id": "A", "price": 100.0})
            records.append({"datetime": ts, "bond_id": "B", "price": 100.0})
        price_panel = pd.DataFrame(records)

        # Create portfolio targets
        targets = [
            PortfolioTarget(
                timestamp=timestamps[0],
                positions={"A": 1000, "B": -1000},
            ),
            PortfolioTarget(
                timestamp=timestamps[1],
                positions={"A": 500, "B": -500},
            ),
        ]

        result = run_portfolio_backtest_from_targets(
            price_panel=price_panel,
            portfolio_targets=targets,
        )

        assert isinstance(result, PortfolioBacktestResult)
        assert len(result.trades) > 0


class TestComputeCostsSummary:
    """Tests for compute_costs_summary function."""

    def test_by_bond(self):
        """Should summarize costs by bond."""
        trades = [
            Trade(pd.Timestamp("2024-01-15"), "A", 1000, cost=100),
            Trade(pd.Timestamp("2024-01-15"), "B", -500, cost=50),
            Trade(pd.Timestamp("2024-01-16"), "A", 500, cost=50),
        ]

        df = compute_costs_summary(trades, by="bond")

        assert df.loc["A", "cost"] == 150
        assert df.loc["B", "cost"] == 50
        assert df.loc["A", "turnover_dv01"] == 1500  # |1000| + |500|

    def test_by_timestamp(self):
        """Should summarize costs by timestamp."""
        trades = [
            Trade(pd.Timestamp("2024-01-15"), "A", 1000, cost=100),
            Trade(pd.Timestamp("2024-01-15"), "B", -500, cost=50),
            Trade(pd.Timestamp("2024-01-16"), "A", 500, cost=25),
        ]

        df = compute_costs_summary(trades, by="timestamp")

        assert df.loc[pd.Timestamp("2024-01-15"), "cost"] == 150
        assert df.loc[pd.Timestamp("2024-01-16"), "cost"] == 25


class TestPortfolioBacktestEndToEnd:
    """End-to-end test: signals -> aggregate -> backtest."""

    def test_full_pipeline(self):
        """Should run complete pipeline from signals to P&L."""
        # 1. Create price panel
        timestamps = pd.date_range("2024-01-15 08:00", periods=5, freq="h")
        bonds = ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]

        records = []
        for i, ts in enumerate(timestamps):
            for bond in bonds:
                # Price changes: 2Y up, 5Y down, 10Y up, 30Y down
                direction = 1 if bond in ["UST_2Y", "UST_10Y"] else -1
                price = 100.0 * (1 + 0.001 * i * direction)
                records.append({"datetime": ts, "bond_id": bond, "price": price})

        price_panel = pd.DataFrame(records)

        # 2. Create fly signals
        ts = timestamps[0]
        signal_2510 = StrategySignal(
            timestamp=ts,
            strategy_id="fly_2y5y10y",
            legs=create_fly_legs("UST_2Y", "UST_5Y", "UST_10Y"),
            direction=1,
            target_gross_dv01=10000,
        )
        signal_51030 = StrategySignal(
            timestamp=ts,
            strategy_id="fly_5y10y30y",
            legs=create_fly_legs("UST_5Y", "UST_10Y", "UST_30Y"),
            direction=1,
            target_gross_dv01=10000,
        )

        # 3. Aggregate
        rules = SizingRules(
            strategy_weights="equal",
            gross_dv01_budget=15000,
        )
        target = aggregate_signal_batch([signal_2510, signal_51030], rules)

        # 4. Run backtest
        config = PortfolioBacktestConfig(cost_bps=0.5)
        result = run_portfolio_backtest_from_targets(
            price_panel=price_panel,
            portfolio_targets=[target],
            config=config,
        )

        # 5. Verify results
        assert result.total_pnl != 0  # Some P&L generated
        assert result.total_cost > 0  # Costs incurred
        assert len(result.trades) > 0  # Trades executed
        assert result.pnl_df["gross_dv01"].iloc[0] <= 15000  # Budget respected


class TestEqualWeights:
    """Tests for compute_equal_weights function."""

    def test_equal_weights_basic(self):
        """Should compute equal weights summing to 1."""
        weights = compute_equal_weights(["A", "B", "C"])

        assert len(weights) == 3
        assert weights["A"] == pytest.approx(1 / 3)
        assert weights["B"] == pytest.approx(1 / 3)
        assert weights["C"] == pytest.approx(1 / 3)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_equal_weights_single(self):
        """Single strategy gets weight 1."""
        weights = compute_equal_weights(["only_one"])

        assert weights["only_one"] == pytest.approx(1.0)

    def test_equal_weights_empty(self):
        """Empty list returns empty dict."""
        weights = compute_equal_weights([])

        assert weights == {}


class TestInverseVolWeights:
    """Tests for compute_inverse_vol_weights function."""

    def test_inverse_vol_basic(self):
        """Lower vol strategy should get higher weight."""
        # strat_A: high vol, strat_B: low vol
        pnl_df = pd.DataFrame({
            "strat_A": [100, -100, 100, -100, 100],  # High vol
            "strat_B": [10, -10, 10, -10, 10],  # Low vol (10x less)
        })

        weights, vols = compute_inverse_vol_weights(pnl_df, lookback=5)

        # strat_B should have higher weight (lower vol)
        assert weights["strat_B"] > weights["strat_A"]
        assert sum(weights.values()) == pytest.approx(1.0)

        # Volatilities should be captured
        assert vols["strat_A"] > vols["strat_B"]

    def test_inverse_vol_equal_vol(self):
        """Equal vol should give equal weights."""
        pnl_df = pd.DataFrame({
            "strat_A": [10, -10, 10, -10, 10],
            "strat_B": [10, -10, 10, -10, 10],  # Same vol
        })

        weights, _ = compute_inverse_vol_weights(pnl_df, lookback=5)

        assert weights["strat_A"] == pytest.approx(0.5)
        assert weights["strat_B"] == pytest.approx(0.5)

    def test_inverse_vol_floor(self):
        """Should use floor vol for constant P&L."""
        pnl_df = pd.DataFrame({
            "strat_A": [100, -100, 100, -100, 100],
            "strat_B": [0, 0, 0, 0, 0],  # Zero vol
        })

        weights, vols = compute_inverse_vol_weights(pnl_df, lookback=5, floor_vol=0.01)

        # Should not error, strat_B gets floor vol
        assert vols["strat_B"] == pytest.approx(0.01)
        assert sum(weights.values()) == pytest.approx(1.0)


class TestApplyWeightCaps:
    """Tests for apply_weight_caps function."""

    def test_cap_applied(self):
        """Should cap weights above max."""
        weights = {"A": 0.7, "B": 0.2, "C": 0.1}

        capped, was_binding = apply_weight_caps(weights, max_weight=0.5)

        assert capped["A"] == pytest.approx(0.5)  # Capped
        assert was_binding
        # Remaining weight redistributed to B and C
        assert sum(capped.values()) == pytest.approx(1.0)

    def test_floor_applied(self):
        """Should floor weights below min."""
        weights = {"A": 0.8, "B": 0.15, "C": 0.05}

        capped, was_binding = apply_weight_caps(weights, min_weight=0.1)

        assert capped["C"] >= 0.1  # Floored
        assert was_binding
        assert sum(capped.values()) == pytest.approx(1.0)

    def test_no_cap_needed(self):
        """Should not modify weights within bounds."""
        weights = {"A": 0.4, "B": 0.35, "C": 0.25}

        capped, was_binding = apply_weight_caps(weights, min_weight=0.1, max_weight=0.5)

        assert capped == weights
        assert not was_binding

    def test_cap_and_floor_together(self):
        """Should apply both cap and floor."""
        weights = {"A": 0.8, "B": 0.15, "C": 0.05}

        capped, was_binding = apply_weight_caps(
            weights, min_weight=0.1, max_weight=0.5
        )

        assert capped["A"] <= 0.5
        assert capped["C"] >= 0.1
        assert was_binding
        assert sum(capped.values()) == pytest.approx(1.0)

    def test_weights_sum_to_one_after_cap(self):
        """Weights should always sum to 1 after capping."""
        weights = {"A": 0.6, "B": 0.3, "C": 0.1}

        capped, _ = apply_weight_caps(weights, max_weight=0.4)

        assert sum(capped.values()) == pytest.approx(1.0)


class TestComputeStrategyWeights:
    """Tests for compute_strategy_weights function."""

    def test_equal_method(self):
        """Should compute equal weights."""
        config = WeightingConfig(method=WeightingMethod.EQUAL)
        result = compute_strategy_weights(["A", "B", "C"], config)

        assert result.weights["A"] == pytest.approx(1 / 3)
        assert result.method == "equal"
        assert result.weight_sum == pytest.approx(1.0)

    def test_inverse_vol_method(self):
        """Should compute inverse vol weights."""
        pnl_df = pd.DataFrame({
            "strat_A": [100, -100, 100, -100, 100],
            "strat_B": [10, -10, 10, -10, 10],
        })

        config = WeightingConfig(
            method=WeightingMethod.INVERSE_VOL,
            vol_lookback=5,
        )
        result = compute_strategy_weights(
            ["strat_A", "strat_B"],
            config,
            strategy_pnl=pnl_df,
        )

        assert result.weights["strat_B"] > result.weights["strat_A"]
        assert result.volatilities is not None
        assert result.method == "inverse_vol"
        assert result.weight_sum == pytest.approx(1.0)

    def test_inverse_vol_with_cap(self):
        """Should apply cap to inverse vol weights."""
        pnl_df = pd.DataFrame({
            "strat_A": [100, -100, 100, -100, 100],  # High vol
            "strat_B": [1, -1, 1, -1, 1],  # Very low vol (would dominate)
        })

        config = WeightingConfig(
            method=WeightingMethod.INVERSE_VOL,
            vol_lookback=5,
            max_weight=0.6,  # Cap at 60%
        )
        result = compute_strategy_weights(
            ["strat_A", "strat_B"],
            config,
            strategy_pnl=pnl_df,
        )

        assert result.weights["strat_B"] <= 0.6
        assert result.caps_applied
        assert result.weight_sum == pytest.approx(1.0)

    def test_custom_method(self):
        """Should use custom weights."""
        config = WeightingConfig(
            method=WeightingMethod.CUSTOM,
            custom_weights={"A": 0.7, "B": 0.3},
        )
        result = compute_strategy_weights(["A", "B"], config)

        assert result.weights["A"] == pytest.approx(0.7)
        assert result.weights["B"] == pytest.approx(0.3)
        assert result.method == "custom"

    def test_missing_pnl_raises(self):
        """Should raise if inverse_vol requested without P&L."""
        config = WeightingConfig(method=WeightingMethod.INVERSE_VOL)

        with pytest.raises(ValueError, match="strategy_pnl required"):
            compute_strategy_weights(["A", "B"], config)


class TestValidateWeights:
    """Tests for validate_weights function."""

    def test_valid_weights(self):
        """Should pass for valid weights."""
        is_valid, errors = validate_weights({"A": 0.5, "B": 0.3, "C": 0.2})

        assert is_valid
        assert errors == []

    def test_weights_not_sum_to_one(self):
        """Should fail if weights don't sum to 1."""
        is_valid, errors = validate_weights({"A": 0.5, "B": 0.3})

        assert not is_valid
        assert any("sum to" in e for e in errors)

    def test_negative_weights(self):
        """Should fail for negative weights."""
        is_valid, errors = validate_weights({"A": 1.2, "B": -0.2})

        assert not is_valid
        assert any("Negative" in e for e in errors)


class TestWeightingIntegration:
    """Integration tests for weighting with aggregation."""

    def test_create_sizing_rules_with_weighting(self):
        """Should create SizingRules from WeightingConfig."""
        pnl_df = pd.DataFrame({
            "fly_2y5y10y": [100, -50, 80, -30, 60],
            "fly_5y10y30y": [50, -25, 40, -15, 30],
        })

        config = WeightingConfig(
            method=WeightingMethod.INVERSE_VOL,
            vol_lookback=5,
            max_weight=0.7,
        )

        rules = create_sizing_rules_with_weighting(
            config,
            strategy_pnl=pnl_df,
            gross_dv01_budget=50000,
        )

        assert isinstance(rules, SizingRules)
        assert callable(rules.strategy_weights)
        assert rules.gross_dv01_budget == 50000

    def test_equal_weighting_shortcut(self):
        """Equal weighting should use string shortcut."""
        config = WeightingConfig(method=WeightingMethod.EQUAL)

        rules = create_sizing_rules_with_weighting(config)

        assert rules.strategy_weights == "equal"

    def test_aggregation_with_inverse_vol_weights(self):
        """Should aggregate using inverse vol weights."""
        ts = pd.Timestamp("2024-01-15")

        # Create strategy P&L (strat_B has lower vol)
        pnl_df = pd.DataFrame({
            "strat_A": [100, -100, 100, -100, 100],
            "strat_B": [20, -20, 20, -20, 20],
        })

        # Create signals
        legs_A = create_fly_legs("A1", "A2", "A3")
        legs_B = create_fly_legs("B1", "B2", "B3")

        signal_A = StrategySignal(
            timestamp=ts, strategy_id="strat_A", legs=legs_A,
            direction=1, target_gross_dv01=10000,
        )
        signal_B = StrategySignal(
            timestamp=ts, strategy_id="strat_B", legs=legs_B,
            direction=1, target_gross_dv01=10000,
        )

        # Create rules with inverse vol weighting
        config = WeightingConfig(
            method=WeightingMethod.INVERSE_VOL,
            vol_lookback=5,
        )
        rules = create_sizing_rules_with_weighting(config, strategy_pnl=pnl_df)

        # Aggregate
        result = signals_to_bond_targets([signal_A, signal_B], rules)

        # strat_B should have higher contribution (lower vol)
        contrib_A = sum(abs(v) for v in result.strategy_contributions["strat_A"].values())
        contrib_B = sum(abs(v) for v in result.strategy_contributions["strat_B"].values())

        assert contrib_B > contrib_A  # Lower vol gets higher weight


class TestWeightingSumToOne:
    """Verify weights sum to 1 under various conditions."""

    def test_equal_weights_sum_to_one(self):
        """Equal weights should sum to 1."""
        for n in [1, 2, 3, 5, 10]:
            strategy_ids = [f"strat_{i}" for i in range(n)]
            weights = compute_equal_weights(strategy_ids)
            assert sum(weights.values()) == pytest.approx(1.0), f"Failed for n={n}"

    def test_inverse_vol_weights_sum_to_one(self):
        """Inverse vol weights should sum to 1."""
        import numpy as np

        np.random.seed(42)
        pnl_df = pd.DataFrame({
            "A": np.random.randn(30) * 100,
            "B": np.random.randn(30) * 50,
            "C": np.random.randn(30) * 25,
        })

        weights, _ = compute_inverse_vol_weights(pnl_df, lookback=20)

        assert sum(weights.values()) == pytest.approx(1.0)

    def test_capped_weights_sum_to_one(self):
        """Capped weights should sum to 1."""
        # Various initial weight distributions
        test_cases = [
            {"A": 0.9, "B": 0.05, "C": 0.05},
            {"A": 0.4, "B": 0.4, "C": 0.2},
            {"A": 0.33, "B": 0.33, "C": 0.34},
        ]

        for weights in test_cases:
            capped, _ = apply_weight_caps(weights, min_weight=0.1, max_weight=0.5)
            assert sum(capped.values()) == pytest.approx(1.0), f"Failed for {weights}"

    def test_strategy_weights_always_sum_to_one(self):
        """compute_strategy_weights should always sum to 1."""
        pnl_df = pd.DataFrame({
            "A": [100, -50, 80, -30, 60, 40, -20, 50, -40, 30],
            "B": [20, -10, 16, -6, 12, 8, -4, 10, -8, 6],
            "C": [50, -25, 40, -15, 30, 20, -10, 25, -20, 15],
        })

        # Test all methods
        configs = [
            WeightingConfig(method=WeightingMethod.EQUAL),
            WeightingConfig(method=WeightingMethod.INVERSE_VOL, vol_lookback=5),
            WeightingConfig(method=WeightingMethod.INVERSE_VOL, max_weight=0.5),
            WeightingConfig(method=WeightingMethod.CUSTOM, custom_weights={"A": 0.5, "B": 0.3, "C": 0.2}),
        ]

        for config in configs:
            result = compute_strategy_weights(
                ["A", "B", "C"],
                config,
                strategy_pnl=pnl_df,
            )
            assert result.weight_sum == pytest.approx(1.0), f"Failed for {config.method}"
