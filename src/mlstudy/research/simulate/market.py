"""Simulate synthetic panel market data for ML pipeline testing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_market_data(
    n_assets: int = 5,
    n_periods: int = 1000,
    freq: str = "1h",
    start_date: str = "2023-01-01",
    seed: int = 42,
    base_prices: list[float] | None = None,
    volatility: float = 0.02,
    regime_shift_prob: float = 0.005,
    flow_signal_strength: float = 0.1,
) -> pd.DataFrame:
    """Simulate synthetic panel market data.

    Generates a tidy long-format DataFrame with multiple assets over time.
    Includes mild regime shifts and a weak predictive signal from flow_imbalance
    to future returns (for testing ML pipelines).

    Args:
        n_assets: Number of assets to simulate.
        n_periods: Number of time periods per asset.
        freq: Frequency string (e.g., "1h", "1D").
        start_date: Start date for the time series.
        seed: Random seed for reproducibility.
        base_prices: Initial prices per asset. If None, uses [100, 50, 200, 75, 150].
        volatility: Base volatility (std of returns).
        regime_shift_prob: Probability of regime shift at each step.
        flow_signal_strength: Strength of flow_imbalance -> future return signal.

    Returns:
        DataFrame with columns:
        - datetime: Timestamp
        - asset: Asset identifier (e.g., "ASSET_0", "ASSET_1", ...)
        - close: Simulated price
        - volume: Simulated volume
        - flow_imbalance: Order flow imbalance (-1 to 1), weakly predictive of future returns

    Example:
        >>> df = simulate_market_data(n_assets=3, n_periods=100, seed=42)
        >>> df.columns.tolist()
        ['datetime', 'asset', 'close', 'volume', 'flow_imbalance']
    """
    rng = np.random.default_rng(seed)

    if base_prices is None:
        base_prices = [100.0, 50.0, 200.0, 75.0, 150.0]

    # Extend base_prices if needed
    while len(base_prices) < n_assets:
        base_prices.append(rng.uniform(50, 200))

    # Generate datetime index
    dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)

    all_data = []

    for asset_idx in range(n_assets):
        asset_name = f"ASSET_{asset_idx}"
        price = base_prices[asset_idx]

        # Initialize regime (affects drift)
        regime = 0  # -1: bearish, 0: neutral, 1: bullish
        regime_drift = 0.0

        prices = []
        volumes = []
        flow_imbalances = []

        for t in range(n_periods):
            # Regime shifts
            if rng.random() < regime_shift_prob:
                regime = rng.choice([-1, 0, 1])
                regime_drift = regime * 0.0005  # Small drift per regime

            # Generate flow imbalance (mean-reverting, bounded)
            # This will be a weak predictor of future returns
            if t == 0:
                flow = rng.uniform(-0.3, 0.3)
            else:
                # Mean-reverting with noise
                flow = flow_imbalances[-1] * 0.9 + rng.normal(0, 0.2)
                flow = np.clip(flow, -1, 1)

            flow_imbalances.append(flow)

            # Generate return with:
            # 1. Base volatility
            # 2. Regime drift
            # 3. Weak signal from PREVIOUS flow imbalance (no lookahead)
            base_return = rng.normal(0, volatility)
            drift = regime_drift

            # Signal: previous flow imbalance weakly predicts this return
            signal_return = (
                flow_imbalances[-2] * flow_signal_strength * volatility if t > 0 else 0
            )

            total_return = base_return + drift + signal_return
            price = price * (1 + total_return)
            prices.append(price)

            # Volume: base level with some noise and correlation to abs(flow)
            base_volume = 10000 * (1 + asset_idx * 0.5)  # Different base per asset
            volume = base_volume * (1 + rng.exponential(0.5) + abs(flow) * 2)
            volumes.append(volume)

        asset_df = pd.DataFrame(
            {
                "datetime": dates,
                "asset": asset_name,
                "close": prices,
                "volume": volumes,
                "flow_imbalance": flow_imbalances,
            }
        )
        all_data.append(asset_df)

    df = pd.concat(all_data, ignore_index=True)

    # Sort by asset then datetime for consistent ordering
    df = df.sort_values(["asset", "datetime"]).reset_index(drop=True)

    return df


def simulate_with_known_signal(
    n_assets: int = 3,
    n_periods: int = 500,
    freq: str = "1h",
    start_date: str = "2023-01-01",
    seed: int = 42,
    signal_lag: int = 1,
    signal_strength: float = 0.3,
) -> pd.DataFrame:
    """Simulate data with a known, stronger signal for testing.

    The flow_imbalance at time t predicts returns at time t+signal_lag
    with a controlled strength. Useful for verifying ML pipelines can
    detect signal.

    Args:
        n_assets: Number of assets.
        n_periods: Number of time periods.
        freq: Frequency string.
        start_date: Start date.
        seed: Random seed.
        signal_lag: Lag between signal and return.
        signal_strength: R-squared-like strength (0 to 1).

    Returns:
        DataFrame with datetime, asset, close, volume, flow_imbalance.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)
    all_data = []

    for asset_idx in range(n_assets):
        asset_name = f"ASSET_{asset_idx}"

        # Generate flow imbalance first (the signal)
        flow = rng.uniform(-1, 1, n_periods)

        # Smooth it a bit
        flow = pd.Series(flow).ewm(span=5).mean().values

        # Generate returns that are partially predicted by lagged flow
        noise_std = np.sqrt(1 - signal_strength)
        signal_std = np.sqrt(signal_strength)

        returns = np.zeros(n_periods)
        for t in range(n_periods):
            noise = rng.normal(0, noise_std) * 0.02
            signal = flow[t - signal_lag] * signal_std * 0.02 if t >= signal_lag else 0
            returns[t] = noise + signal

        # Convert returns to prices
        prices = 100 * np.cumprod(1 + returns)

        # Volume
        volume = 10000 * (1 + rng.exponential(0.5, n_periods))

        asset_df = pd.DataFrame(
            {
                "datetime": dates,
                "asset": asset_name,
                "close": prices,
                "volume": volume,
                "flow_imbalance": flow,
            }
        )
        all_data.append(asset_df)

    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(["asset", "datetime"]).reset_index(drop=True)

    return df
