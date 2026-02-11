"""Intraday fly backtest engine.

Handles session-aware execution, stable daily leg selection,
and proper signal-to-position lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing_extensions import Literal  # noqa: UP035

from mlstudy.trading.backtest.engine import BacktestConfig, SizingMode
from mlstudy.trading.backtest.metrics import BacktestMetrics, compute_metrics
from mlstudy.core.data.session import add_session_flags, is_rebalance_bar
from mlstudy.trading.strategy.structures.selection.leg_selection import (
    attach_daily_legs,
    build_daily_legs_table,
    get_leg_values,
)


@dataclass
class IntradayBacktestConfig(BacktestConfig):
    """Configuration for intraday fly backtest.

    Extends BacktestConfig with session and rebalancing parameters.

    Attributes:
        session_start: Session start time (e.g., "07:30").
        session_end: Session end time (e.g., "17:00").
        tz: Timezone for session times.
        selection_time: Time to select fly legs each day.
        rebalance_mode: When to allow position changes:
            - "open_only": Only at session open
            - "every_bar": Every session bar
            - "every_n_bars": Every N session bars
            - "close_only": Only at session close
        rebalance_n_bars: Number of bars between rebalances (for every_n_bars).
        allow_overnight: If True, positions persist outside session.
    """

    session_start: str = "07:30"
    session_end: str = "17:00"
    tz: str = "Europe/Berlin"
    selection_time: str = "07:30"
    rebalance_mode: Literal["open_only", "every_bar", "every_n_bars", "close_only"] = "every_bar"
    rebalance_n_bars: int = 1
    allow_overnight: bool = True


@dataclass
class IntradayBacktestResult:
    """Result of intraday fly backtest.

    Attributes:
        pnl_df: DataFrame with bar-level P&L, positions, exposures.
        daily_df: DataFrame with daily aggregated results.
        legs_table: Daily leg selection table.
        metrics: Computed backtest metrics (from daily data).
        config: Configuration used.
    """

    pnl_df: pd.DataFrame
    daily_df: pd.DataFrame
    legs_table: pd.DataFrame
    metrics: BacktestMetrics
    config: IntradayBacktestConfig


def compute_fly_signal(
    df: pd.DataFrame,
    window: int,
    robust: bool = False,
    clip: float = 4.0,
) -> pd.Series:
    """Compute z-score signal from fly yield.

    Args:
        df: DataFrame with fly_yield column.
        window: Rolling window for z-score.
        robust: If True, use median/MAD instead of mean/std.
        clip: Clip z-score to [-clip, clip].

    Returns:
        Series of z-scores.
    """
    fly_yield = df["fly_yield"]

    if robust:
        center = fly_yield.rolling(window, min_periods=window // 2).median()
        # MAD with 1.4826 scaling for normal distribution
        mad = fly_yield.rolling(window, min_periods=window // 2).apply(
            lambda x: np.median(np.abs(x - np.median(x))) * 1.4826,
            raw=True,
        )
        zscore = (fly_yield - center) / mad.replace(0, np.nan)
    else:
        center = fly_yield.rolling(window, min_periods=window // 2).mean()
        std = fly_yield.rolling(window, min_periods=window // 2).std()
        zscore = (fly_yield - center) / std.replace(0, np.nan)

    return zscore.clip(-clip, clip)


def generate_positions(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float | None,
    can_trade: pd.Series,
) -> pd.Series:
    """Generate position series from z-score signal.

    Mean-reversion logic:
    - Long (1) when z < -entry_z (fly yield is cheap)
    - Short (-1) when z > entry_z (fly yield is rich)
    - Exit to flat when |z| < exit_z
    - Stop out if |z| > stop_z (optional)

    Position changes only occur at bars where can_trade is True.

    Args:
        zscore: Z-score series.
        entry_z: Entry threshold (absolute).
        exit_z: Exit threshold (absolute).
        stop_z: Stop-loss threshold, or None to disable.
        can_trade: Boolean series indicating tradeable bars.

    Returns:
        Position series (-1, 0, or 1).
    """
    positions = pd.Series(0, index=zscore.index, dtype=int)
    current_pos = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        tradeable = can_trade.iloc[i]

        if pd.isna(z):
            positions.iloc[i] = current_pos
            continue

        if tradeable:
            # Can change position
            if current_pos == 0:
                # Not in position - check for entry
                if z < -entry_z:
                    current_pos = 1  # Long (yield cheap, expect reversion up)
                elif z > entry_z:
                    current_pos = -1  # Short (yield rich, expect reversion down)
            else:
                # In position - check for exit or stop
                if stop_z is not None and abs(z) > stop_z:
                    # Stop loss
                    current_pos = 0
                elif abs(z) < exit_z:
                    # Exit on mean reversion
                    current_pos = 0
                elif current_pos == 1 and z > entry_z:
                    # Was long, now short signal
                    current_pos = -1
                elif current_pos == -1 and z < -entry_z:
                    # Was short, now long signal
                    current_pos = 1

        positions.iloc[i] = current_pos

    return positions


def compute_leg_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute returns for each leg from prices.

    Args:
        df: DataFrame with front_price, belly_price, back_price columns.

    Returns:
        DataFrame with leg return columns added.
    """
    df = df.copy()

    for leg in ["front", "belly", "back"]:
        price_col = f"{leg}_price"
        if price_col in df.columns:
            # Simple return: (P_t / P_{t-1}) - 1
            df[f"{leg}_ret"] = df[price_col].pct_change()

    return df


def scale_weights_for_dv01_target(
    weights: np.ndarray,
    dv01s: np.ndarray,
    target_dv01: float,
) -> np.ndarray:
    """Scale weights to achieve target gross DV01.

    Args:
        weights: Base weights [front, belly, back].
        dv01s: DV01 per unit [front, belly, back].
        target_dv01: Target gross DV01.

    Returns:
        Scaled weights.
    """
    gross_dv01 = np.sum(np.abs(weights * dv01s))
    if gross_dv01 < 1e-10:
        return weights
    scale = target_dv01 / gross_dv01
    return weights * scale


def backtest_fly_intraday(
    panel_df: pd.DataFrame,
    tenors: tuple[float, float, float] = (2, 5, 10),
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float | None = 4.0,
    config: IntradayBacktestConfig | None = None,
    use_dv01_weights: bool = True,
    robust_zscore: bool = False,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    yield_col: str = "yield",
    price_col: str = "price",
    dv01_col: str = "dv01",
    verbose: bool = True,
) -> IntradayBacktestResult:
    """Run intraday fly backtest with session-aware execution.

    Signal flow:
    1. Legs selected once per day at selection_time
    2. Fly yield computed from leg yields
    3. Z-score signal computed from fly yield
    4. Signal at bar t determines position at bar t+1 (no lookahead)
    5. PnL computed from price returns

    Args:
        panel_df: Panel DataFrame with bond data.
        tenors: Target tenors (front, belly, back).
        window: Z-score lookback window (in bars).
        entry_z: Entry z-score threshold.
        exit_z: Exit z-score threshold.
        stop_z: Stop-loss threshold, or None to disable.
        config: Backtest configuration.
        use_dv01_weights: If True, use DV01-neutral weights.
        robust_zscore: If True, use median/MAD for z-score.
        datetime_col: Datetime column name.
        bond_id_col: Bond ID column name.
        ttm_col: TTM column name.
        yield_col: Yield column name.
        price_col: Price column name.
        dv01_col: DV01 column name.
        verbose: If True, print progress.

    Returns:
        IntradayBacktestResult with bar-level and daily P&L.

    Example:
        >>> result = backtest_fly_intraday(
        ...     panel_df,
        ...     tenors=(2, 5, 10),
        ...     window=20,
        ...     entry_z=2.0,
        ...     config=IntradayBacktestConfig(
        ...         session_start="07:30",
        ...         session_end="17:00",
        ...         rebalance_mode="open_only",
        ...     ),
        ... )
        >>> print(result.metrics.sharpe_ratio)
    """
    if config is None:
        config = IntradayBacktestConfig()

    df = panel_df.copy()

    # Step 1: Build daily leg selection table
    if verbose:
        print("Building daily leg selections...")

    legs_table = build_daily_legs_table(
        df,
        tenors=tenors,
        selection_time=config.selection_time,
        tz=config.tz,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        ttm_col=ttm_col,
    )

    if len(legs_table) == 0:
        raise ValueError("No valid leg selections found. Check data and tenors.")

    if verbose:
        print(f"  Selected legs for {len(legs_table)} trading days")

    # Step 2: Add session flags
    if verbose:
        print("Adding session flags...")

    df = add_session_flags(
        df,
        datetime_col=datetime_col,
        start=config.session_start,
        end=config.session_end,
        tz=config.tz,
    )

    # Step 3: Attach daily legs to all bars
    df = attach_daily_legs(df, legs_table, datetime_col=datetime_col, tz=config.tz)

    # Step 4: Get leg values at each bar
    value_cols = [yield_col, price_col, dv01_col]
    df = get_leg_values(
        panel_df,
        df,
        value_cols=value_cols,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
    )

    # Step 5: Compute fly yield
    if use_dv01_weights:
        # DV01-neutral weights
        front_dv01 = df[f"front_{dv01_col}"].fillna(1.0)
        belly_dv01 = df[f"belly_{dv01_col}"].fillna(1.0)
        back_dv01 = df[f"back_{dv01_col}"].fillna(1.0)

        # Wings positive, belly negative, net DV01 = 0
        # Belly weight scaled to match front + back DV01
        w_front = 1.0
        w_back = 1.0
        w_belly = -(front_dv01 * w_front + back_dv01 * w_back) / belly_dv01

        df["w_front"] = w_front
        df["w_belly"] = w_belly
        df["w_back"] = w_back
    else:
        df["w_front"] = 1.0
        df["w_belly"] = -2.0
        df["w_back"] = 1.0

    front_yield = df[f"front_{yield_col}"]
    belly_yield = df[f"belly_{yield_col}"]
    back_yield = df[f"back_{yield_col}"]

    df["fly_yield"] = (
        df["w_front"] * front_yield
        + df["w_belly"] * belly_yield
        + df["w_back"] * back_yield
    )

    # Step 6: Compute z-score signal
    if verbose:
        print("Computing signals...")

    df["zscore"] = compute_fly_signal(df, window, robust=robust_zscore)

    # Step 7: Determine tradeable bars
    can_trade = is_rebalance_bar(
        df,
        rebalance_mode=config.rebalance_mode,
        n_bars=config.rebalance_n_bars,
        datetime_col=datetime_col,
    )
    df["can_trade"] = can_trade

    # Step 8: Generate positions (signal at t -> position at t+1)
    raw_signal = generate_positions(
        df["zscore"],
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        can_trade=can_trade,
    )
    df["signal"] = raw_signal

    # Lag signal by 1 bar to get position (no lookahead)
    df["position"] = df["signal"].shift(config.signal_lag).fillna(0).astype(int)

    # Step 9: Compute leg returns from prices
    df = compute_leg_returns(df)

    # Step 10: Compute PnL
    if verbose:
        print("Computing P&L...")

    pnl_records = []
    prev_position = 0

    for i in range(len(df)):
        row = df.iloc[i]
        dt = row[datetime_col]
        position = int(row["position"])

        # Get weights and DV01s
        w = np.array([row["w_front"], row["w_belly"], row["w_back"]])
        dv01s = np.array([
            row.get(f"front_{dv01_col}", 0.0),
            row.get(f"belly_{dv01_col}", 0.0),
            row.get(f"back_{dv01_col}", 0.0),
        ])
        dv01s = np.nan_to_num(dv01s, nan=0.0)

        prices = np.array([
            row.get(f"front_{price_col}", 100.0),
            row.get(f"belly_{price_col}", 100.0),
            row.get(f"back_{price_col}", 100.0),
        ])
        prices = np.nan_to_num(prices, nan=100.0)

        # Apply sizing
        if config.sizing_mode == SizingMode.DV01_TARGET and position != 0:
            scaled_w = scale_weights_for_dv01_target(w, dv01s, config.dv01_target)
        elif config.sizing_mode == SizingMode.FIXED_NOTIONAL:
            avg_price = np.mean(prices[prices > 0]) if np.any(prices > 0) else 100.0
            scaled_w = w * (config.fixed_notional / avg_price)
        else:
            scaled_w = w

        # Current exposure
        effective_w = position * scaled_w
        gross_dv01 = np.sum(np.abs(effective_w * dv01s))
        net_dv01 = np.sum(effective_w * dv01s)
        gross_notional = np.sum(np.abs(effective_w * prices))

        # Leg returns
        leg_rets = np.array([
            row.get("front_ret", 0.0),
            row.get("belly_ret", 0.0),
            row.get("back_ret", 0.0),
        ])
        leg_rets = np.nan_to_num(leg_rets, nan=0.0)

        # Gross return (uses position at bar start, return during bar)
        # PnL = position * sum(w_i * leg_ret_i)
        gross_return = position * np.sum(scaled_w * leg_rets) if position != 0 else 0.0

        # Turnover (change in position)
        position_change = abs(position - prev_position)
        turnover = position_change * gross_notional if position_change > 0 else 0.0

        # Transaction cost
        transaction_cost = turnover * config.total_cost_bps() / 10000

        # Net return
        net_return = gross_return - transaction_cost

        pnl_records.append({
            datetime_col: dt,
            "trading_date": row.get("trading_date"),
            "is_session": row.get("is_session", True),
            "is_open_bar": row.get("is_open_bar", False),
            "is_close_bar": row.get("is_close_bar", False),
            "can_trade": row.get("can_trade", True),
            "fly_yield": row.get("fly_yield"),
            "zscore": row.get("zscore"),
            "signal": int(row.get("signal", 0)),
            "position": position,
            "front_ret": leg_rets[0],
            "belly_ret": leg_rets[1],
            "back_ret": leg_rets[2],
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "gross_dv01": gross_dv01,
            "net_dv01": net_dv01,
            "gross_notional": gross_notional,
            "front_id": row.get("front_id"),
            "belly_id": row.get("belly_id"),
            "back_id": row.get("back_id"),
        })

        prev_position = position

    pnl_df = pd.DataFrame(pnl_records)
    pnl_df["cumulative_pnl"] = pnl_df["net_return"].cumsum()

    # Step 11: Daily aggregation
    if verbose:
        print("Aggregating to daily...")

    daily_df = aggregate_to_daily(pnl_df, datetime_col=datetime_col)

    # Step 12: Compute metrics from daily data
    metrics = compute_metrics(daily_df)

    if verbose:
        print(f"Backtest complete. Sharpe: {metrics.sharpe_ratio:.2f}")

    return IntradayBacktestResult(
        pnl_df=pnl_df,
        daily_df=daily_df,
        legs_table=legs_table,
        metrics=metrics,
        config=config,
    )


def aggregate_to_daily(
    pnl_df: pd.DataFrame,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Aggregate intraday P&L to daily.

    Args:
        pnl_df: Bar-level P&L DataFrame.
        datetime_col: Datetime column name.

    Returns:
        Daily aggregated DataFrame.
    """
    if "trading_date" not in pnl_df.columns:
        pnl_df = pnl_df.copy()
        pnl_df["trading_date"] = pd.to_datetime(pnl_df[datetime_col]).dt.date

    daily = (
        pnl_df.groupby("trading_date")
        .agg({
            "gross_return": "sum",
            "net_return": "sum",
            "turnover": "sum",
            "transaction_cost": "sum",
            "gross_dv01": "last",  # End-of-day exposure
            "net_dv01": "last",
            "gross_notional": "last",
            "position": "last",  # End-of-day position
        })
        .reset_index()
    )

    # Rename for metrics compatibility
    daily = daily.rename(columns={"trading_date": "datetime"})
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    daily["cumulative_pnl"] = daily["net_return"].cumsum()
    daily["traded_notional"] = daily["turnover"]

    return daily
