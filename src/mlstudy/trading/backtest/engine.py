"""Fly backtest engine with yield-based signals and price-based PnL.

Signal generation uses fly yield z-scores, but PnL is computed from
actual price changes of the 3-leg portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class SizingMode(Enum):
    """Position sizing mode."""

    FIXED_NOTIONAL = "fixed_notional"
    DV01_TARGET = "dv01_target"


@dataclass
class BacktestConfig:
    """Configuration for fly backtest.

    Attributes:
        sizing_mode: How to size positions.
        fixed_notional: Notional per leg when sizing_mode=FIXED_NOTIONAL.
        dv01_target: Target portfolio DV01 when sizing_mode=DV01_TARGET.
        transaction_cost_bps: Transaction cost in bps on traded notional.
        slippage_bps: Bid-ask/slippage cost in bps (placeholder).
        signal_lag: Days to lag signal (1 = trade on next day's open).
    """

    sizing_mode: SizingMode = SizingMode.FIXED_NOTIONAL
    fixed_notional: float = 1_000_000.0
    dv01_target: float = 10_000.0  # $10k DV01 target
    transaction_cost_bps: float = 1.0  # 1 bp per trade
    slippage_bps: float = 0.5  # 0.5 bp slippage placeholder
    signal_lag: int = 1

    def total_cost_bps(self) -> float:
        """Total trading cost in bps."""
        return self.transaction_cost_bps + self.slippage_bps


@dataclass
class BacktestResult:
    """Result of fly backtest.

    Attributes:
        pnl_df: DataFrame with daily P&L, costs, exposures, positions.
        config: Backtest configuration used.
    """

    pnl_df: pd.DataFrame
    config: BacktestConfig


def scale_weights_to_dv01_target(
    weights: NDArray,
    dv01s: NDArray,
    target_dv01: float,
) -> NDArray:
    """Scale weights so portfolio gross DV01 equals target.

    Args:
        weights: Base weights [w_front, w_belly, w_back].
        dv01s: DV01 per unit for each leg [dv01_front, dv01_belly, dv01_back].
        target_dv01: Target gross portfolio DV01.

    Returns:
        Scaled weights.
    """
    current_gross_dv01 = np.sum(np.abs(weights * dv01s))
    if current_gross_dv01 < 1e-10:
        return weights
    scale = target_dv01 / current_gross_dv01
    return weights * scale


def compute_leg_notionals(
    weights: NDArray,
    prices: NDArray,
    position: int,
) -> NDArray:
    """Compute notional exposure for each leg.

    Args:
        weights: Position weights [w_front, w_belly, w_back].
        prices: Current prices [p_front, p_belly, p_back].
        position: Signal position (-1, 0, or 1).

    Returns:
        Notional per leg (can be negative for short).
    """
    if position == 0:
        return np.zeros(3)
    return position * weights * prices


def compute_traded_notional(
    prev_notionals: NDArray,
    curr_notionals: NDArray,
) -> float:
    """Compute total traded notional (turnover).

    Args:
        prev_notionals: Previous leg notionals.
        curr_notionals: Current leg notionals.

    Returns:
        Sum of absolute notional changes.
    """
    return np.sum(np.abs(curr_notionals - prev_notionals))


def backtest_fly(
    legs_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    config: BacktestConfig | None = None,
    datetime_col: str = "datetime",
    signal_col: str = "signal",
    price_col: str = "price",
    dv01_col: str = "dv01",
    weight_col: str = "weight",
) -> BacktestResult:
    """Run fly backtest with yield-based signals and price-based PnL.

    PnL definition:
    - At each date t, hold a 3-leg portfolio with weights w_front, w_belly, w_back.
    - Position direction comes from signal: pos in {-1, 0, 1}.
    - leg_ret = (price_{t+1} - price_t) / price_t
    - Portfolio return = pos * sum(w_i * leg_ret_i)

    Costs:
    - Transaction costs = cost_bps * traded_notional
    - Traded notional = sum of absolute notional changes across legs

    Args:
        legs_df: DataFrame from build_fly_legs_panel with leg-level data.
            Must have datetime, leg, price, dv01, weight columns.
        signal_df: DataFrame with signal column from build_signal_dataframe.
            Must have datetime and signal columns.
        config: Backtest configuration. Uses defaults if None.
        datetime_col: Datetime column name.
        signal_col: Signal column name in signal_df.
        price_col: Price column name in legs_df.
        dv01_col: DV01 column name in legs_df.
        weight_col: Weight column name in legs_df.

    Returns:
        BacktestResult with pnl_df containing:
        - datetime: Date
        - signal: Raw signal
        - position: Lagged position
        - front_ret, belly_ret, back_ret: Leg returns
        - gross_return: Return before costs
        - traded_notional: Turnover
        - transaction_cost: Total trading costs
        - net_return: Return after costs
        - cumulative_pnl: Cumulative P&L
        - net_dv01, gross_dv01: DV01 exposures
        - gross_notional: Gross notional exposure
        - leverage: Gross/net notional ratio

    Example:
        >>> result = backtest_fly(fly_result.legs_df, signal_df, config)
        >>> print(result.pnl_df[["datetime", "position", "net_return"]].head())
    """
    if config is None:
        config = BacktestConfig()

    # Pivot legs_df to get per-date leg data
    legs_pivot = _pivot_legs_data(
        legs_df, datetime_col, price_col, dv01_col, weight_col
    )

    # Merge signal with legs data
    merged = signal_df[[datetime_col, signal_col]].merge(
        legs_pivot, on=datetime_col, how="inner"
    )
    merged = merged.sort_values(datetime_col).reset_index(drop=True)

    # Compute leg returns (forward-looking for next-day P&L)
    for leg in ["front", "belly", "back"]:
        price_col_name = f"{leg}_price"
        if price_col_name in merged.columns:
            merged[f"{leg}_ret"] = merged[price_col_name].pct_change()

    # Lag signal (trade on next day)
    merged["position"] = merged[signal_col].shift(config.signal_lag).fillna(0).astype(int)

    # Initialize tracking
    results = []
    prev_notionals = np.zeros(3)

    for i in range(len(merged)):
        row = merged.iloc[i]
        dt = row[datetime_col]
        position = int(row["position"])

        # Get base weights from fly construction
        base_weights = np.array([
            row.get("front_weight", 1.0),
            row.get("belly_weight", -2.0),
            row.get("back_weight", 1.0),
        ])

        # Get DV01s and prices
        dv01s = np.array([
            row.get("front_dv01", 0.0),
            row.get("belly_dv01", 0.0),
            row.get("back_dv01", 0.0),
        ])
        prices = np.array([
            row.get("front_price", 100.0),
            row.get("belly_price", 100.0),
            row.get("back_price", 100.0),
        ])

        # Apply sizing
        if config.sizing_mode == SizingMode.DV01_TARGET and position != 0:
            weights = scale_weights_to_dv01_target(
                base_weights, dv01s, config.dv01_target
            )
        elif config.sizing_mode == SizingMode.FIXED_NOTIONAL:
            avg_price = np.mean(prices[prices > 0]) if np.any(prices > 0) else 100.0
            weights = base_weights * (config.fixed_notional / avg_price)
        else:
            weights = base_weights

        # Compute current notionals
        curr_notionals = compute_leg_notionals(weights, prices, position)

        # Compute traded notional (turnover)
        traded_notional = compute_traded_notional(prev_notionals, curr_notionals)

        # Transaction cost
        transaction_cost = traded_notional * config.total_cost_bps() / 10000

        # Get leg returns
        leg_returns = np.array([
            row.get("front_ret", 0.0),
            row.get("belly_ret", 0.0),
            row.get("back_ret", 0.0),
        ])
        leg_returns = np.nan_to_num(leg_returns, nan=0.0)

        # Compute gross portfolio return
        gross_return = position * np.sum(weights * leg_returns) if position != 0 else 0.0

        # Net return
        net_return = gross_return - transaction_cost

        # Exposure metrics
        if position != 0:
            effective_weights = position * weights
            leg_dv01s = effective_weights * dv01s
            net_dv01 = np.sum(leg_dv01s)
            gross_dv01 = np.sum(np.abs(leg_dv01s))
            gross_notional = np.sum(np.abs(curr_notionals))
            net_notional = np.abs(np.sum(curr_notionals))
            leverage = gross_notional / net_notional if net_notional > 1e-10 else 0.0
        else:
            net_dv01 = gross_dv01 = gross_notional = leverage = 0.0

        results.append({
            datetime_col: dt,
            "signal": int(row[signal_col]),
            "position": position,
            "front_ret": leg_returns[0],
            "belly_ret": leg_returns[1],
            "back_ret": leg_returns[2],
            "gross_return": gross_return,
            "traded_notional": traded_notional,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "net_dv01": net_dv01,
            "gross_dv01": gross_dv01,
            "gross_notional": gross_notional,
            "leverage": leverage,
        })

        # Update for next iteration
        prev_notionals = curr_notionals

    # Build results DataFrame
    pnl_df = pd.DataFrame(results)
    pnl_df["cumulative_pnl"] = pnl_df["net_return"].cumsum()

    return BacktestResult(pnl_df=pnl_df, config=config)


def _pivot_legs_data(
    legs_df: pd.DataFrame,
    datetime_col: str,
    price_col: str,
    dv01_col: str,
    weight_col: str,
) -> pd.DataFrame:
    """Pivot legs DataFrame to wide format with per-leg columns."""
    dates = legs_df[datetime_col].unique()

    records = []
    for dt in dates:
        dt_data = legs_df[legs_df[datetime_col] == dt]
        record = {datetime_col: dt}

        for leg in ["front", "belly", "back"]:
            leg_row = dt_data[dt_data["leg"] == leg]
            if len(leg_row) > 0:
                leg_row = leg_row.iloc[0]
                record[f"{leg}_price"] = leg_row.get(price_col, np.nan)
                record[f"{leg}_dv01"] = leg_row.get(dv01_col, np.nan)
                record[f"{leg}_weight"] = leg_row.get(weight_col, np.nan)

        records.append(record)

    return pd.DataFrame(records)


def backtest_fly_from_panel(
    panel_df: pd.DataFrame,
    legs_table: pd.DataFrame,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float | None = 4.0,
    config: BacktestConfig | None = None,
    use_dv01_weights: bool = True,
    robust_zscore: bool = False,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    yield_col: str = "yield",
    price_col: str = "price",
    dv01_col: str = "dv01",
) -> BacktestResult:
    """Convenience function to run complete fly backtest from panel data.

    Args:
        panel_df: Panel DataFrame with datetime, bond_id, yield, price, dv01.
        legs_table: DataFrame from select_fly_legs.
        window: Z-score lookback window.
        entry_z: Entry threshold (absolute).
        exit_z: Exit threshold (absolute).
        stop_z: Stop-loss threshold, or None to disable.
        config: Backtest configuration.
        use_dv01_weights: If True, use DV01-neutral weights.
        robust_zscore: If True, use median/MAD for z-score.
        datetime_col: Datetime column name.
        bond_id_col: Bond ID column name.
        yield_col: Yield column name.
        price_col: Price column name.
        dv01_col: DV01 column name.

    Returns:
        BacktestResult with daily P&L.
    """
    from mlstudy.trading.strategy.structures.specs.fly import build_fly
    from mlstudy.trading.strategy.alpha.mean_reversion.signals import build_signal_dataframe

    # Build fly
    fly_result = build_fly(
        df=panel_df,
        legs_table=legs_table,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        value_cols=[yield_col, price_col, dv01_col],
        use_dv01_weights=use_dv01_weights,
        dv01_col=dv01_col,
        yield_col=yield_col,
    )

    # Get fly yield for signal generation
    fly_yield = fly_result.get_fly_yield()

    # Build signals
    signal_df = build_signal_dataframe(
        fly_yield,
        window=window,
        robust=robust_zscore,
        clip=4.0,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
    )

    # Run backtest
    return backtest_fly(
        legs_df=fly_result.legs_df,
        signal_df=signal_df,
        config=config,
        datetime_col=datetime_col,
        price_col=price_col,
        dv01_col=dv01_col,
    )
