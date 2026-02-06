"""Fly universe generation and parameter sweep.

Generate all valid butterfly combinations from a tenor list and
run backtests across multiple flies and parameter sets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from mlstudy.trading.backtest import (
    BacktestConfig,
    BacktestResult,
    SizingMode,
    backtest_fly_from_panel,
    compute_metrics,
)
from mlstudy.trading.strategy.structures.specs.fly import select_fly_legs


def generate_flies_from_tenors(
    tenors: list[float] | None = None,
    min_wing_spread: float = 0.5,
    require_symmetric: bool = False,
) -> list[tuple[float, float, float]]:
    """Generate all valid fly combinations from tenor list.

    A valid fly has:
    - front < belly < back
    - Minimum spread between legs (wing spread)
    - Optionally symmetric wings (belly equidistant from front/back)

    Args:
        tenors: List of available tenors. Default: [1, 2, 3, 5, 7, 10, 15, 30].
        min_wing_spread: Minimum tenor difference between adjacent legs.
        require_symmetric: If True, only return flies with equal wing spreads.

    Returns:
        List of (front, belly, back) tuples representing valid flies.

    Example:
        >>> flies = generate_flies_from_tenors([2, 5, 10, 30])
        >>> # Returns: [(2, 5, 10), (2, 5, 30), (2, 10, 30), (5, 10, 30)]
        >>>
        >>> # Symmetric only
        >>> flies = generate_flies_from_tenors([2, 5, 7, 10], require_symmetric=True)
        >>> # Returns flies where belly-front == back-belly
    """
    if tenors is None:
        tenors = [1, 2, 3, 5, 7, 10, 15, 30]

    tenors = sorted(tenors)
    flies = []

    for i, front in enumerate(tenors):
        for j, belly in enumerate(tenors[i + 1 :], i + 1):
            for back in tenors[j + 1 :]:
                # Check minimum wing spread
                front_wing = belly - front
                back_wing = back - belly

                if front_wing < min_wing_spread or back_wing < min_wing_spread:
                    continue

                # Check symmetry if required (allow small tolerance for floating point)
                if require_symmetric and abs(front_wing - back_wing) > 0.01:
                    continue

                flies.append((front, belly, back))

    return flies


def fly_name(front: float, belly: float, back: float) -> str:
    """Generate human-readable fly name.

    Args:
        front: Front leg tenor.
        belly: Belly leg tenor.
        back: Back leg tenor.

    Returns:
        String like "2s5s10s" or "2y5y10y".
    """

    def tenor_str(t: float) -> str:
        if t < 1:
            return f"{int(t * 12)}m"
        elif t == int(t):
            return f"{int(t)}y"
        else:
            return f"{t}y"

    return f"{tenor_str(front)}{tenor_str(belly)}{tenor_str(back)}"


@dataclass
class ParamGrid:
    """Parameter grid for sweep.

    Attributes:
        windows: Z-score lookback windows to test.
        entry_zs: Entry z-score thresholds to test.
        exit_zs: Exit z-score thresholds to test.
        stop_zs: Stop-loss thresholds to test (None = disabled).
        sizing_modes: Sizing modes to test.
        robust_zscores: Whether to test robust z-score.
    """

    windows: list[int] = field(default_factory=lambda: [20])
    entry_zs: list[float] = field(default_factory=lambda: [2.0])
    exit_zs: list[float] = field(default_factory=lambda: [0.5])
    stop_zs: list[float | None] = field(default_factory=lambda: [4.0])
    sizing_modes: list[SizingMode] = field(
        default_factory=lambda: [SizingMode.FIXED_NOTIONAL]
    )
    robust_zscores: list[bool] = field(default_factory=lambda: [False])

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all parameter combinations."""
        for combo in product(
            self.windows,
            self.entry_zs,
            self.exit_zs,
            self.stop_zs,
            self.sizing_modes,
            self.robust_zscores,
        ):
            yield {
                "window": combo[0],
                "entry_z": combo[1],
                "exit_z": combo[2],
                "stop_z": combo[3],
                "sizing_mode": combo[4],
                "robust_zscore": combo[5],
            }

    def __len__(self) -> int:
        """Total number of parameter combinations."""
        return (
            len(self.windows)
            * len(self.entry_zs)
            * len(self.exit_zs)
            * len(self.stop_zs)
            * len(self.sizing_modes)
            * len(self.robust_zscores)
        )


@dataclass
class FlyBacktestSummary:
    """Summary of a single fly backtest run.

    Attributes:
        fly_name: Human-readable fly name (e.g., "2y5y10y").
        front: Front leg tenor.
        belly: Belly leg tenor.
        back: Back leg tenor.
        params: Parameter dict used for this run.
        metrics: Dict of computed metrics.
        result: Full BacktestResult (optional, for detailed analysis).
    """

    fly_name: str
    front: float
    belly: float
    back: float
    params: dict
    metrics: dict
    result: BacktestResult | None = None


def build_and_backtest_many_flies(
    panel_df: pd.DataFrame,
    flies: list[tuple[float, float, float]] | None = None,
    param_grid: ParamGrid | None = None,
    config: BacktestConfig | None = None,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    ttm_col: str = "ttm_years",
    yield_col: str = "yield",
    price_col: str = "price",
    dv01_col: str = "dv01",
    use_dv01_weights: bool = True,
    keep_results: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build and backtest multiple flies with parameter sweep.

    Args:
        panel_df: Panel DataFrame with bond data.
        flies: List of (front, belly, back) tenor tuples to test.
            If None, generates from standard tenors.
        param_grid: ParamGrid for parameter sweep.
            If None, uses defaults.
        config: Base BacktestConfig (sizing params override param_grid).
        datetime_col: Datetime column name.
        bond_id_col: Bond ID column name.
        ttm_col: TTM column name.
        yield_col: Yield column name.
        price_col: Price column name.
        dv01_col: DV01 column name.
        use_dv01_weights: Whether to use DV01-neutral weights.
        keep_results: If True, include full BacktestResult in output.
        verbose: If True, print progress.

    Returns:
        DataFrame with columns:
        - fly_name: Human-readable fly name
        - front, belly, back: Leg tenors
        - window, entry_z, exit_z, stop_z, sizing_mode, robust_zscore: Params
        - sharpe_ratio, total_pnl, max_drawdown, etc.: Metrics
        - result (optional): Full BacktestResult object

    Example:
        >>> flies = generate_flies_from_tenors([2, 5, 10, 30])
        >>> param_grid = ParamGrid(
        ...     windows=[10, 20, 30],
        ...     entry_zs=[1.5, 2.0, 2.5],
        ... )
        >>> summary_df = build_and_backtest_many_flies(
        ...     panel_df, flies, param_grid
        ... )
        >>> # Find best fly/param combo by Sharpe
        >>> best = summary_df.loc[summary_df["sharpe_ratio"].idxmax()]
    """
    if flies is None:
        flies = generate_flies_from_tenors()

    if param_grid is None:
        param_grid = ParamGrid()

    if config is None:
        config = BacktestConfig()

    # Get available tenors in data
    available_tenors = set(panel_df[ttm_col].round(1).unique())

    results = []
    total_runs = len(flies) * len(param_grid)
    run_count = 0

    for front, belly, back in flies:
        # Check if tenors are available (approximately)
        tenor_available = all(
            any(abs(t - avail) < 0.5 for avail in available_tenors)
            for t in [front, belly, back]
        )

        if not tenor_available:
            if verbose:
                print(f"  Skipping {fly_name(front, belly, back)} - tenors not in data")
            continue

        # Select legs for this fly
        try:
            legs_table = select_fly_legs(
                panel_df,
                tenors=(front, belly, back),
                datetime_col=datetime_col,
                bond_id_col=bond_id_col,
                ttm_col=ttm_col,
            )
        except Exception as e:
            if verbose:
                print(f"  Error selecting legs for {fly_name(front, belly, back)}: {e}")
            continue

        if len(legs_table) == 0:
            if verbose:
                print(f"  Skipping {fly_name(front, belly, back)} - no valid leg selections")
            continue

        for params in param_grid:
            run_count += 1

            if verbose and run_count % 10 == 0:
                print(f"  Progress: {run_count}/{total_runs} runs")

            # Create config for this run
            run_config = BacktestConfig(
                sizing_mode=params["sizing_mode"],
                fixed_notional=config.fixed_notional,
                dv01_target=config.dv01_target,
                transaction_cost_bps=config.transaction_cost_bps,
                slippage_bps=config.slippage_bps,
            )

            try:
                # Run backtest
                bt_result = backtest_fly_from_panel(
                    panel_df=panel_df,
                    legs_table=legs_table,
                    window=params["window"],
                    entry_z=params["entry_z"],
                    exit_z=params["exit_z"],
                    stop_z=params["stop_z"],
                    config=run_config,
                    use_dv01_weights=use_dv01_weights,
                    robust_zscore=params["robust_zscore"],
                    datetime_col=datetime_col,
                    bond_id_col=bond_id_col,
                    yield_col=yield_col,
                    price_col=price_col,
                    dv01_col=dv01_col,
                )

                # Compute metrics
                metrics = compute_metrics(bt_result.pnl_df)
                metrics_dict = metrics.to_dict()

            except Exception as e:
                if verbose:
                    print(f"  Error in backtest {fly_name(front, belly, back)}: {e}")
                continue

            # Build summary row
            row = {
                "fly_name": fly_name(front, belly, back),
                "front": front,
                "belly": belly,
                "back": back,
                "window": params["window"],
                "entry_z": params["entry_z"],
                "exit_z": params["exit_z"],
                "stop_z": params["stop_z"],
                "sizing_mode": params["sizing_mode"].value,
                "robust_zscore": params["robust_zscore"],
                **metrics_dict,
            }

            if keep_results:
                row["result"] = bt_result

            results.append(row)

    if verbose:
        print(f"Completed {len(results)} successful backtests")

    return pd.DataFrame(results)


def summarize_by_fly(
    sweep_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    agg: str = "mean",
) -> pd.DataFrame:
    """Summarize sweep results by fly, aggregating across parameters.

    Args:
        sweep_df: DataFrame from build_and_backtest_many_flies.
        metric: Metric column to summarize.
        agg: Aggregation method ("mean", "median", "max", "min").

    Returns:
        DataFrame with one row per fly, sorted by metric.
    """
    agg_funcs = {
        metric: agg,
        "total_pnl": agg,
        "max_drawdown": "min",  # Worst drawdown
        "n_trades": "mean",
        "hit_rate": "mean",
    }

    # Only use columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in sweep_df.columns}

    summary = (
        sweep_df.groupby(["fly_name", "front", "belly", "back"])
        .agg(agg_funcs)
        .reset_index()
        .sort_values(metric, ascending=False)
    )

    return summary


def summarize_by_params(
    sweep_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    agg: str = "mean",
) -> pd.DataFrame:
    """Summarize sweep results by parameters, aggregating across flies.

    Args:
        sweep_df: DataFrame from build_and_backtest_many_flies.
        metric: Metric column to summarize.
        agg: Aggregation method ("mean", "median", "max", "min").

    Returns:
        DataFrame with one row per parameter combo, sorted by metric.
    """
    param_cols = ["window", "entry_z", "exit_z", "stop_z", "sizing_mode", "robust_zscore"]
    param_cols = [c for c in param_cols if c in sweep_df.columns]

    agg_funcs = {
        metric: agg,
        "total_pnl": agg,
        "max_drawdown": "min",
        "n_trades": "mean",
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in sweep_df.columns}

    summary = (
        sweep_df.groupby(param_cols)
        .agg(agg_funcs)
        .reset_index()
        .sort_values(metric, ascending=False)
    )

    return summary


def get_best_fly_params(
    sweep_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    ascending: bool = False,
) -> pd.Series:
    """Get the best fly/parameter combination by metric.

    Args:
        sweep_df: DataFrame from build_and_backtest_many_flies.
        metric: Metric to optimize.
        ascending: If True, lower is better (e.g., max_drawdown).

    Returns:
        Series with best row's data.
    """
    idx = sweep_df[metric].idxmin() if ascending else sweep_df[metric].idxmax()
    return sweep_df.loc[idx]


def filter_valid_flies(
    sweep_df: pd.DataFrame,
    min_sharpe: float | None = None,
    max_drawdown: float | None = None,
    min_trades: int | None = None,
    min_hit_rate: float | None = None,
) -> pd.DataFrame:
    """Filter sweep results to valid/tradeable flies.

    Args:
        sweep_df: DataFrame from build_and_backtest_many_flies.
        min_sharpe: Minimum Sharpe ratio.
        max_drawdown: Maximum drawdown (negative, e.g., -0.1).
        min_trades: Minimum number of trades.
        min_hit_rate: Minimum hit rate.

    Returns:
        Filtered DataFrame.
    """
    mask = pd.Series(True, index=sweep_df.index)

    if min_sharpe is not None and "sharpe_ratio" in sweep_df.columns:
        mask &= sweep_df["sharpe_ratio"] >= min_sharpe

    if max_drawdown is not None and "max_drawdown" in sweep_df.columns:
        mask &= sweep_df["max_drawdown"] >= max_drawdown  # DD is negative

    if min_trades is not None and "n_trades" in sweep_df.columns:
        mask &= sweep_df["n_trades"] >= min_trades

    if min_hit_rate is not None and "hit_rate" in sweep_df.columns:
        mask &= sweep_df["hit_rate"] >= min_hit_rate

    return sweep_df[mask]
