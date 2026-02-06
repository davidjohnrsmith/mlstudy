"""Mean-reversion signal generation from fly yields.

Generates trading signals based on z-score deviations from
historical mean, with configurable entry/exit bands.

Units note: Returns and PnL are computed as yield differences unless
external returns series is provided. For true P&L, pass price-based returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing_extensions import Literal  # noqa: UP035

import numpy as np
import pandas as pd


class Position(IntEnum):
    """Trading position enum."""

    SHORT = -1
    FLAT = 0
    LONG = 1


class ExitReason(IntEnum):
    """Reason for exiting a position."""

    NONE = 0
    NORMAL_EXIT = 1  # Z-score crossed exit threshold
    STOP_LOSS = 2  # Z-score exceeded stop threshold
    MAX_HOLD = 3  # Position held too long
    SIGNAL_REVERSAL = 4  # Signal reversed


@dataclass
class SignalConfig:
    """Configuration for mean-reversion signal generation.

    Attributes:
        entry_z: Z-score threshold to enter position (absolute value).
        exit_z: Z-score threshold to exit position (absolute value).
        stop_z: Z-score threshold for stop-loss (absolute value).
            Exit if z moves further against us (momentum taking over).
        max_hold_bars: Maximum bars to hold position. Exit if exceeded.
        cooldown_bars: Number of FULL bars after an exit during which
            re-entry is blocked. E.g., cooldown_bars=3 means if you exit
            at bar t, you cannot re-enter until bar t+4.
        cooldown_z_threshold: Z-score threshold to end cooldown early.
            If |z| < this value, cooldown ends. None to disable.
        entry_long_z: Override entry threshold for long (default: -entry_z).
        entry_short_z: Override entry threshold for short (default: +entry_z).

    Example:
        >>> config = SignalConfig(
        ...     entry_z=2.0,
        ...     exit_z=0.5,
        ...     stop_z=4.0,
        ...     max_hold_bars=20,
        ...     cooldown_bars=5,
        ...     cooldown_z_threshold=0.5,
        ... )
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = 4.0
    max_hold_bars: int | None = None
    cooldown_bars: int = 0
    cooldown_z_threshold: float | None = None

    entry_long_z: float | None = None
    entry_short_z: float | None = None

    def get_entry_long(self) -> float:
        """Z-score threshold to enter long (buy)."""
        return self.entry_long_z if self.entry_long_z is not None else -self.entry_z

    def get_entry_short(self) -> float:
        """Z-score threshold to enter short (sell)."""
        return self.entry_short_z if self.entry_short_z is not None else self.entry_z


def rolling_zscore(
    series: pd.Series,
    window: int,
    robust: bool = False,
    clip: float | None = None,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> pd.Series:
    """Compute rolling z-score with numerical safety.

    Args:
        series: Input time series (e.g., fly yield).
        window: Lookback window for mean and std.
        robust: If True, use median and MAD instead of mean/std.
        clip: If set, clip z-scores to [-clip, +clip].
        min_periods: Minimum observations required. Default: window.
        eps: Small value to prevent division by zero.

    Returns:
        Series of z-scores. Finite wherever input is finite.

    Example:
        >>> z = rolling_zscore(fly_yield, window=20, robust=True, clip=4)
    """
    if min_periods is None:
        min_periods = window

    if robust:
        # Robust z-score using median and MAD
        median = series.rolling(window, min_periods=min_periods).median()
        # MAD = median absolute deviation
        mad = series.rolling(window, min_periods=min_periods).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        # Scale MAD to be consistent with std for normal distribution
        # For normal distribution, MAD ≈ 0.6745 * std
        scaled_mad = mad * 1.4826  # 1/0.6745

        # Numerical safety: clip denominator
        scaled_mad_safe = scaled_mad.clip(lower=eps)
        zscore = (series - median) / scaled_mad_safe
    else:
        # Standard z-score
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()

        # Numerical safety: clip denominator
        std_safe = std.clip(lower=eps)
        zscore = (series - mean) / std_safe

    if clip is not None:
        zscore = zscore.clip(-clip, clip)

    zscore.name = "zscore"
    return zscore


def ewma_zscore(
    series: pd.Series,
    span: int,
    clip: float | None = None,
    min_periods: int = 1,
    eps: float = 1e-12,
) -> pd.Series:
    """Compute EWMA-based z-score with numerical safety.

    Uses exponentially weighted mean and std for faster adaptation
    to regime changes.

    Args:
        series: Input time series.
        span: EWMA span parameter (higher = slower adaptation).
        clip: If set, clip z-scores to [-clip, +clip].
        min_periods: Minimum observations required.
        eps: Small value to prevent division by zero.

    Returns:
        Series of z-scores. Finite wherever input is finite.

    Example:
        >>> z = ewma_zscore(fly_yield, span=20, clip=4)
    """
    ewm = series.ewm(span=span, min_periods=min_periods)
    mean = ewm.mean()
    std = ewm.std()

    # Numerical safety: clip denominator
    std_safe = std.clip(lower=eps)
    zscore = (series - mean) / std_safe

    if clip is not None:
        zscore = zscore.clip(-clip, clip)

    zscore.name = "zscore"
    return zscore


def generate_mean_reversion_signal(
    zscore: pd.Series,
    config: SignalConfig | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float | None = 4.0,
    max_hold_bars: int | None = None,
    cooldown_bars: int = 0,
    cooldown_z_threshold: float | None = None,
    return_details: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Generate mean-reversion trading signal from z-score.

    Signal logic:
    - Enter LONG when z < -entry_z (fly is cheap)
    - Enter SHORT when z > +entry_z (fly is rich)
    - Exit to FLAT when z crosses exit_z toward zero
    - Stop to FLAT when z exceeds stop_z (momentum taking over)
    - Exit if position held longer than max_hold_bars
    - Block re-entry for cooldown_bars FULL bars after exit/stop

    Cooldown semantics: cooldown_bars=3 means if you exit at bar t,
    bars t+1, t+2, t+3 are blocked, and you can re-enter at t+4.

    Args:
        zscore: Z-score series.
        config: SignalConfig object (overrides other params if provided).
        entry_z: Entry threshold (absolute value).
        exit_z: Exit threshold (absolute value).
        stop_z: Stop-loss threshold (absolute value).
        max_hold_bars: Maximum bars to hold position.
        cooldown_bars: Number of FULL bars after exit during which re-entry is blocked.
        cooldown_z_threshold: Z threshold to end cooldown early.
        return_details: If True, also return DataFrame with exit reasons.

    Returns:
        Series of signals: -1 (short), 0 (flat), 1 (long).
        If return_details=True, also returns DataFrame with columns:
        - signal, exit_reason, bars_held, cooldown_remaining
        Note: bars_held at exit bar reflects the holding duration (not 0).

    Example:
        >>> signal = generate_mean_reversion_signal(zscore, entry_z=2, exit_z=0.5)
        >>> signal, details = generate_mean_reversion_signal(zscore, config=config, return_details=True)
    """
    if config is not None:
        entry_long = config.get_entry_long()
        entry_short = config.get_entry_short()
        exit_z = config.exit_z
        stop_z = config.stop_z
        max_hold_bars = config.max_hold_bars
        cooldown_bars = config.cooldown_bars
        cooldown_z_threshold = config.cooldown_z_threshold
    else:
        entry_long = -entry_z
        entry_short = entry_z

    # Convert to numpy for speed
    z_arr = zscore.to_numpy()
    n = len(z_arr)

    signal = np.zeros(n, dtype=np.int32)
    exit_reasons = np.zeros(n, dtype=np.int32)
    bars_held = np.zeros(n, dtype=np.int32)
    cooldown_remaining = np.zeros(n, dtype=np.int32)

    position = Position.FLAT
    hold_count = 0
    cooldown_count = 0

    for i in range(n):
        z = z_arr[i]

        if np.isnan(z):
            signal[i] = position
            bars_held[i] = hold_count
            cooldown_remaining[i] = cooldown_count
            continue

        exit_reason = ExitReason.NONE
        exited_this_bar = False

        # Check if cooldown can end early due to z returning near zero
        if (
            cooldown_count > 0
            and cooldown_z_threshold is not None
            and abs(z) < cooldown_z_threshold
        ):
            cooldown_count = 0

        if position == Position.FLAT:
            # Check entry conditions (only if not in cooldown)
            if cooldown_count == 0:
                if z <= entry_long:
                    position = Position.LONG
                    hold_count = 0  # Will be incremented to 1 on next bar
                elif z >= entry_short:
                    position = Position.SHORT
                    hold_count = 0  # Will be incremented to 1 on next bar
            else:
                # Decrement cooldown when flat and in cooldown
                cooldown_count -= 1

        elif position == Position.LONG:
            hold_count += 1

            # Check exit/stop conditions
            if z >= -exit_z:
                # Exit: z crossed back toward zero
                exit_reason = ExitReason.NORMAL_EXIT
                exited_this_bar = True
            elif stop_z is not None and z <= -stop_z:
                # Stop: momentum against us
                exit_reason = ExitReason.STOP_LOSS
                exited_this_bar = True
            elif max_hold_bars is not None and hold_count >= max_hold_bars:
                # Max hold exceeded
                exit_reason = ExitReason.MAX_HOLD
                exited_this_bar = True

            if exited_this_bar:
                # Record bars_held at exit with the final count
                bars_held[i] = hold_count
                position = Position.FLAT
                cooldown_count = cooldown_bars
                hold_count = 0

        elif position == Position.SHORT:
            hold_count += 1

            # Check exit/stop conditions
            if z <= exit_z:
                # Exit: z crossed back toward zero
                exit_reason = ExitReason.NORMAL_EXIT
                exited_this_bar = True
            elif stop_z is not None and z >= stop_z:
                # Stop: momentum against us
                exit_reason = ExitReason.STOP_LOSS
                exited_this_bar = True
            elif max_hold_bars is not None and hold_count >= max_hold_bars:
                # Max hold exceeded
                exit_reason = ExitReason.MAX_HOLD
                exited_this_bar = True

            if exited_this_bar:
                # Record bars_held at exit with the final count
                bars_held[i] = hold_count
                position = Position.FLAT
                cooldown_count = cooldown_bars
                hold_count = 0

        signal[i] = position
        exit_reasons[i] = exit_reason
        if not exited_this_bar:
            bars_held[i] = hold_count
        cooldown_remaining[i] = cooldown_count

    result = pd.Series(signal, index=zscore.index, name="signal")

    if return_details:
        details = pd.DataFrame(
            {
                "signal": signal,
                "exit_reason": exit_reasons,
                "bars_held": bars_held,
                "cooldown_remaining": cooldown_remaining,
            },
            index=zscore.index,
        )
        return result, details

    return result


def compute_signal_strength(
    zscore: pd.Series,
    signal: pd.Series,
    max_z: float = 4.0,
) -> pd.Series:
    """Compute signal strength (conviction level).

    Strength is based on how far z-score is from entry threshold.
    Ranges from 0 (at entry) to 1 (at max_z).

    Args:
        zscore: Z-score series.
        signal: Signal series (-1, 0, 1).
        max_z: Z-score at which strength = 1.

    Returns:
        Series of strength values [0, 1].
    """
    # Convert to numpy for speed
    z_arr = zscore.to_numpy()
    sig_arr = signal.to_numpy()

    strength = np.zeros(len(z_arr), dtype=np.float64)

    for i in range(len(z_arr)):
        z = z_arr[i]
        sig = sig_arr[i]

        if np.isnan(z) or sig == 0:
            strength[i] = 0.0
        else:
            # Strength based on absolute z
            abs_z = abs(z)
            strength[i] = min(abs_z / max_z, 1.0)

    return pd.Series(strength, index=zscore.index, name="strength")


def build_signal_dataframe(
    fly_yield: pd.Series,
    zscore_method: Literal["rolling", "ewma"] = "rolling",
    window: int = 20,
    span: int | None = None,
    robust: bool = False,
    clip: float = 4.0,
    config: SignalConfig | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float | None = 4.0,
    include_strength: bool = True,
) -> pd.DataFrame:
    """Build complete signal DataFrame from fly yield.

    Args:
        fly_yield: Fly yield series with datetime index.
        zscore_method: Method for z-score: "rolling" or "ewma".
        window: Rolling window for z-score (used when zscore_method="rolling").
        span: EWMA span (used when zscore_method="ewma"). If None with ewma, uses window.
        robust: Use robust z-score (median/MAD). Only for rolling.
        clip: Clip z-scores to [-clip, +clip].
        config: SignalConfig object. If provided, overrides entry_z/exit_z/stop_z.
        entry_z: Entry threshold (ignored if config provided).
        exit_z: Exit threshold (ignored if config provided).
        stop_z: Stop-loss threshold (ignored if config provided).
        include_strength: Include signal strength column.

    Returns:
        DataFrame with columns:
        - datetime: Index as column
        - fly_yield: Input fly yield
        - zscore: Z-score
        - signal: Position signal (-1/0/1)
        - strength: Signal strength (optional)

    Example:
        >>> signal_df = build_signal_dataframe(
        ...     fly_yield, zscore_method="rolling", window=20, entry_z=2, exit_z=0.5
        ... )
    """
    # Compute z-score
    if zscore_method == "ewma":
        effective_span = span if span is not None else window
        zscore = ewma_zscore(fly_yield, span=effective_span, clip=clip)
    else:
        zscore = rolling_zscore(fly_yield, window=window, robust=robust, clip=clip)

    # Generate signal
    if config is not None:
        signal = generate_mean_reversion_signal(zscore, config=config)
    else:
        signal = generate_mean_reversion_signal(
            zscore, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z
        )

    # Build DataFrame
    result = pd.DataFrame(
        {
            "fly_yield": fly_yield.values,
            "zscore": zscore.values,
            "signal": signal.values,
        },
        index=fly_yield.index,
    )

    # Add strength if requested
    if include_strength:
        strength = compute_signal_strength(zscore, signal, max_z=clip)
        result["strength"] = strength.values

    # Reset index to get datetime as column
    result = result.reset_index()
    if result.columns[0] != "datetime":
        result = result.rename(columns={result.columns[0]: "datetime"})

    return result


@dataclass
class SignalStats:
    """Statistics for signal analysis."""

    n_observations: int
    n_long_signals: int
    n_short_signals: int
    n_flat_signals: int
    pct_long: float
    pct_short: float
    pct_flat: float
    avg_holding_periods: float
    avg_zscore_at_entry: float


def compute_signal_stats(signal_df: pd.DataFrame) -> SignalStats:
    """Compute statistics for signal DataFrame.

    Args:
        signal_df: DataFrame from build_signal_dataframe.

    Returns:
        SignalStats with signal statistics.

    Note:
        avg_zscore_at_entry uses ONLY true entry points (FLAT -> LONG/SHORT).
        avg_holding_periods uses completed trades only.
    """
    n = len(signal_df)
    n_long = (signal_df["signal"] == 1).sum()
    n_short = (signal_df["signal"] == -1).sum()
    n_flat = (signal_df["signal"] == 0).sum()

    # Count holding periods (consecutive non-flat signals)
    signal = signal_df["signal"].values
    holding_periods = []
    current_hold = 0

    for i in range(len(signal)):
        if signal[i] != 0:
            current_hold += 1
        else:
            if current_hold > 0:
                holding_periods.append(current_hold)
            current_hold = 0

    if current_hold > 0:
        holding_periods.append(current_hold)

    avg_holding = np.mean(holding_periods) if holding_periods else 0.0

    # Average z-score at entry: ONLY true entries (0 -> ±1)
    signal_arr = signal_df["signal"].values
    zscore_arr = signal_df["zscore"].values

    entry_zscores = []
    for i in range(1, len(signal_arr)):
        prev_sig = signal_arr[i - 1]
        curr_sig = signal_arr[i]
        # Entry is transition from FLAT to LONG or SHORT
        if prev_sig == 0 and curr_sig != 0:
            z = zscore_arr[i]
            if not np.isnan(z):
                entry_zscores.append(abs(z))

    avg_z_entry = np.mean(entry_zscores) if entry_zscores else 0.0

    return SignalStats(
        n_observations=n,
        n_long_signals=int(n_long),
        n_short_signals=int(n_short),
        n_flat_signals=int(n_flat),
        pct_long=n_long / n if n > 0 else 0,
        pct_short=n_short / n if n > 0 else 0,
        pct_flat=n_flat / n if n > 0 else 0,
        avg_holding_periods=avg_holding,
        avg_zscore_at_entry=avg_z_entry,
    )


@dataclass
class Trade:
    """Represents a single completed trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int  # 1 for long, -1 for short
    entry_z: float
    exit_z: float
    holding_bars: int
    exit_reason: int
    gross_pnl: float
    cost: float
    net_pnl: float


@dataclass
class TradeStats:
    """Statistics computed from trade blotter."""

    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    profit_factor: float
    avg_hold: float
    pnl_by_exit_reason: dict = field(default_factory=dict)


def build_trade_blotter(
    signal: pd.Series,
    zscore: pd.Series,
    returns: pd.Series | None = None,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Build trade blotter with one row per completed trade.

    A trade starts on transition FLAT->LONG or FLAT->SHORT.
    A trade ends when position returns to FLAT.

    Direct flips (SHORT->LONG or LONG->SHORT) are treated as exit + new entry
    at the same timestamp (two separate trades).

    Args:
        signal: Signal series (-1, 0, 1).
        zscore: Z-score series (for recording entry/exit z).
        returns: Optional returns series. If None, uses zscore.diff() as proxy.
            PnL is computed as: position_{t-1} * returns_t.
        transaction_cost: Cost per unit of position change.

    Returns:
        DataFrame with columns:
        - entry_time, exit_time, side, entry_z, exit_z,
        - holding_bars, exit_reason, gross_pnl, cost, net_pnl

    Note:
        Without a returns series, PnL is based on z-score differences,
        which is a proxy. For true P&L, provide price-based returns.
    """
    sig_arr = signal.to_numpy()
    z_arr = zscore.to_numpy()
    idx = signal.index

    if returns is not None:
        ret_arr = returns.to_numpy()
    else:
        # Use zscore diff as proxy
        ret_arr = np.diff(z_arr, prepend=np.nan)

    trades: list[dict] = []

    in_trade = False
    trade_side = 0
    entry_idx = 0
    entry_z = 0.0
    trade_pnl = 0.0
    trade_cost = 0.0
    holding_bars = 0

    for i in range(len(sig_arr)):
        curr_sig = sig_arr[i]
        prev_sig = sig_arr[i - 1] if i > 0 else 0

        # Accumulate PnL for open trade (position_{t-1} * returns_t)
        if in_trade and i > 0:
            ret = ret_arr[i]
            if not np.isnan(ret):
                trade_pnl += trade_side * ret

        # Check for position changes
        if curr_sig != prev_sig:
            # Handle direct flip (LONG->SHORT or SHORT->LONG) as two trades
            if in_trade and prev_sig != 0 and curr_sig != 0 and curr_sig != prev_sig:
                # Close existing trade
                exit_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                exit_cost = abs(prev_sig) * transaction_cost
                trade_cost += exit_cost

                trades.append(
                    {
                        "entry_time": idx[entry_idx],
                        "exit_time": idx[i],
                        "side": trade_side,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "holding_bars": holding_bars,
                        "exit_reason": ExitReason.SIGNAL_REVERSAL,
                        "gross_pnl": trade_pnl,
                        "cost": trade_cost,
                        "net_pnl": trade_pnl - trade_cost,
                    }
                )

                # Start new trade immediately
                entry_idx = i
                trade_side = int(curr_sig)
                entry_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                trade_pnl = 0.0
                entry_cost = abs(curr_sig) * transaction_cost
                trade_cost = entry_cost
                holding_bars = 1
                in_trade = True

            # Exit to flat
            elif in_trade and curr_sig == 0:
                exit_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                exit_cost = abs(prev_sig) * transaction_cost
                trade_cost += exit_cost

                # Determine exit reason from signal details if available
                # For now use NORMAL_EXIT as default
                exit_reason = ExitReason.NORMAL_EXIT

                trades.append(
                    {
                        "entry_time": idx[entry_idx],
                        "exit_time": idx[i],
                        "side": trade_side,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "holding_bars": holding_bars,
                        "exit_reason": exit_reason,
                        "gross_pnl": trade_pnl,
                        "cost": trade_cost,
                        "net_pnl": trade_pnl - trade_cost,
                    }
                )
                in_trade = False
                trade_side = 0
                trade_pnl = 0.0
                trade_cost = 0.0
                holding_bars = 0

            # Entry from flat
            elif not in_trade and curr_sig != 0:
                entry_idx = i
                trade_side = int(curr_sig)
                entry_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                trade_pnl = 0.0
                entry_cost = abs(curr_sig) * transaction_cost
                trade_cost = entry_cost
                holding_bars = 1
                in_trade = True

        elif in_trade:
            holding_bars += 1

    # Handle unclosed trade at end
    if in_trade:
        exit_z = z_arr[-1] if not np.isnan(z_arr[-1]) else 0.0
        trades.append(
            {
                "entry_time": idx[entry_idx],
                "exit_time": idx[-1],
                "side": trade_side,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "holding_bars": holding_bars,
                "exit_reason": ExitReason.NONE,  # Still open
                "gross_pnl": trade_pnl,
                "cost": trade_cost,
                "net_pnl": trade_pnl - trade_cost,
            }
        )

    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_z",
                "exit_z",
                "holding_bars",
                "exit_reason",
                "gross_pnl",
                "cost",
                "net_pnl",
            ]
        )

    return pd.DataFrame(trades)


def build_trade_blotter_with_details(
    signal: pd.Series,
    details: pd.DataFrame,
    zscore: pd.Series,
    returns: pd.Series | None = None,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Build trade blotter using signal details for accurate exit reasons.

    Same as build_trade_blotter but uses the details DataFrame from
    generate_mean_reversion_signal(..., return_details=True) for
    accurate exit_reason tracking.

    Args:
        signal: Signal series (-1, 0, 1).
        details: Details DataFrame from generate_mean_reversion_signal.
        zscore: Z-score series.
        returns: Optional returns series.
        transaction_cost: Cost per unit of position change.

    Returns:
        DataFrame with trade blotter.
    """
    sig_arr = signal.to_numpy()
    z_arr = zscore.to_numpy()
    exit_reason_arr = details["exit_reason"].to_numpy()
    idx = signal.index

    if returns is not None:
        ret_arr = returns.to_numpy()
    else:
        ret_arr = np.diff(z_arr, prepend=np.nan)

    trades: list[dict] = []

    in_trade = False
    trade_side = 0
    entry_idx = 0
    entry_z = 0.0
    trade_pnl = 0.0
    trade_cost = 0.0
    holding_bars = 0

    for i in range(len(sig_arr)):
        curr_sig = sig_arr[i]
        prev_sig = sig_arr[i - 1] if i > 0 else 0

        if in_trade and i > 0:
            ret = ret_arr[i]
            if not np.isnan(ret):
                trade_pnl += trade_side * ret

        if curr_sig != prev_sig:
            if in_trade and prev_sig != 0 and curr_sig != 0 and curr_sig != prev_sig:
                exit_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                exit_cost = abs(prev_sig) * transaction_cost
                trade_cost += exit_cost

                trades.append(
                    {
                        "entry_time": idx[entry_idx],
                        "exit_time": idx[i],
                        "side": trade_side,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "holding_bars": holding_bars,
                        "exit_reason": ExitReason.SIGNAL_REVERSAL,
                        "gross_pnl": trade_pnl,
                        "cost": trade_cost,
                        "net_pnl": trade_pnl - trade_cost,
                    }
                )

                entry_idx = i
                trade_side = int(curr_sig)
                entry_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                trade_pnl = 0.0
                entry_cost = abs(curr_sig) * transaction_cost
                trade_cost = entry_cost
                holding_bars = 1
                in_trade = True

            elif in_trade and curr_sig == 0:
                exit_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                exit_cost = abs(prev_sig) * transaction_cost
                trade_cost += exit_cost
                exit_reason = int(exit_reason_arr[i])

                trades.append(
                    {
                        "entry_time": idx[entry_idx],
                        "exit_time": idx[i],
                        "side": trade_side,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "holding_bars": holding_bars,
                        "exit_reason": exit_reason,
                        "gross_pnl": trade_pnl,
                        "cost": trade_cost,
                        "net_pnl": trade_pnl - trade_cost,
                    }
                )
                in_trade = False
                trade_side = 0
                trade_pnl = 0.0
                trade_cost = 0.0
                holding_bars = 0

            elif not in_trade and curr_sig != 0:
                entry_idx = i
                trade_side = int(curr_sig)
                entry_z = z_arr[i] if not np.isnan(z_arr[i]) else 0.0
                trade_pnl = 0.0
                entry_cost = abs(curr_sig) * transaction_cost
                trade_cost = entry_cost
                holding_bars = 1
                in_trade = True

        elif in_trade:
            holding_bars += 1

    if in_trade:
        exit_z = z_arr[-1] if not np.isnan(z_arr[-1]) else 0.0
        trades.append(
            {
                "entry_time": idx[entry_idx],
                "exit_time": idx[-1],
                "side": trade_side,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "holding_bars": holding_bars,
                "exit_reason": ExitReason.NONE,
                "gross_pnl": trade_pnl,
                "cost": trade_cost,
                "net_pnl": trade_pnl - trade_cost,
            }
        )

    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "entry_z",
                "exit_z",
                "holding_bars",
                "exit_reason",
                "gross_pnl",
                "cost",
                "net_pnl",
            ]
        )

    return pd.DataFrame(trades)


def compute_trade_stats(trade_blotter: pd.DataFrame) -> TradeStats:
    """Compute statistics from trade blotter.

    Args:
        trade_blotter: DataFrame from build_trade_blotter.

    Returns:
        TradeStats with win_rate, avg_win, avg_loss, expectancy,
        profit_factor, avg_hold, n_trades, pnl_by_exit_reason.
    """
    if len(trade_blotter) == 0:
        return TradeStats(
            n_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            avg_hold=0.0,
            pnl_by_exit_reason={},
        )

    net_pnl = trade_blotter["net_pnl"].values
    winning = net_pnl > 0
    losing = net_pnl < 0

    n_trades = len(trade_blotter)
    n_wins = winning.sum()
    n_losses = losing.sum()

    win_rate = n_wins / n_trades if n_trades > 0 else 0.0

    wins = net_pnl[winning]
    losses = net_pnl[losing]

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0  # Will be negative

    expectancy = net_pnl.mean() if n_trades > 0 else 0.0

    total_wins = wins.sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    avg_hold = trade_blotter["holding_bars"].mean()

    # PnL by exit reason
    pnl_by_exit_reason = {}
    for reason in ExitReason:
        mask = trade_blotter["exit_reason"] == reason
        if mask.any():
            pnl_by_exit_reason[reason.name] = trade_blotter.loc[mask, "net_pnl"].sum()

    return TradeStats(
        n_trades=n_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        profit_factor=profit_factor,
        avg_hold=avg_hold,
        pnl_by_exit_reason=pnl_by_exit_reason,
    )


def backtest_signal(
    signal_df: pd.DataFrame,
    returns: pd.Series | None = None,
    fly_yield_col: str = "fly_yield",
    signal_col: str = "signal",
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Simple backtest of signal on fly yield changes.

    Args:
        signal_df: DataFrame from build_signal_dataframe.
        returns: Optional pre-computed returns. If None, uses fly_yield changes.
        fly_yield_col: Fly yield column name.
        signal_col: Signal column name.
        transaction_cost: Cost per trade (in return units).

    Returns:
        DataFrame with columns:
        - datetime
        - signal: Position
        - returns: Fly yield change (or provided returns)
        - strategy_returns: Signal * returns - costs
        - cumulative_returns: Cumulative strategy returns

    Note:
        Returns computed from fly yield differences are a proxy.
        For true P&L, provide price-based returns.
    """
    df = signal_df.copy()

    # Compute returns (fly yield change = P&L for long fly position)
    if returns is not None:
        df["returns"] = returns.values
    else:
        df["returns"] = df[fly_yield_col].diff()

    # Shift signal by 1 (trade on next day's return)
    df["position"] = df[signal_col].shift(1).fillna(0)

    # Strategy returns
    df["strategy_returns"] = df["position"] * df["returns"]

    # Transaction costs (on position changes)
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_returns"] -= df["trade"] * transaction_cost

    # Cumulative returns
    df["cumulative_returns"] = df["strategy_returns"].cumsum()

    return df[["datetime", "signal", "returns", "strategy_returns", "cumulative_returns"]]


def compute_backtest_stats(backtest_df: pd.DataFrame) -> dict:
    """Compute backtest statistics.

    Args:
        backtest_df: DataFrame from backtest_signal.

    Returns:
        Dict with performance statistics.
    """
    returns = backtest_df["strategy_returns"].dropna()

    total_return = returns.sum()
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

    # Drawdown
    cumulative = backtest_df["cumulative_returns"].dropna()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()

    # Win rate
    winning = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning / total_trades if total_trades > 0 else 0

    return {
        "total_return": total_return,
        "mean_daily_return": mean_return,
        "std_daily_return": std_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_observations": len(returns),
    }
