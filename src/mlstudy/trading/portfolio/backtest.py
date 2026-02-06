"""Portfolio-level backtest from bond-level holdings.

Simulates trading a portfolio of bonds with:
- Target DV01 positions at rebalance times
- P&L computed from price changes
- Transaction costs from turnover
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class RebalanceMode(Enum):
    """When to rebalance the portfolio."""

    TARGET_TIMES = "target_times"  # Rebalance at times where targets are provided
    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold
    PERIODIC = "periodic"  # Rebalance at fixed intervals


@dataclass
class RebalanceRule:
    """Rules for when and how to rebalance.

    Attributes:
        mode: When to trigger rebalances.
        threshold_pct: For THRESHOLD mode, rebalance if any position drifts
            by more than this percentage from target.
        min_trade_dv01: Minimum DV01 change to execute a trade (avoid tiny trades).
    """

    mode: RebalanceMode = RebalanceMode.TARGET_TIMES
    threshold_pct: float | None = None
    min_trade_dv01: float = 0.0


@dataclass
class PortfolioBacktestConfig:
    """Configuration for portfolio backtest.

    Attributes:
        cost_bps: Transaction cost in basis points of notional.
        cost_per_dv01: Alternative cost model: fixed cost per DV01 traded.
            If set, overrides cost_bps.
        slippage_bps: Additional slippage in bps (applied to notional).
        use_dv01_cost_proxy: If True, estimate notional from DV01 for cost calc.
            Assumes notional ≈ DV01 * 10000 (typical for 10y duration).
        dv01_to_notional_factor: Factor to convert DV01 to notional estimate.
            Default 10000 assumes ~10bp DV01 per unit notional.
    """

    cost_bps: float = 0.0
    cost_per_dv01: float | None = None
    slippage_bps: float = 0.0
    use_dv01_cost_proxy: bool = True
    dv01_to_notional_factor: float = 10000.0


@dataclass
class Trade:
    """A single trade execution.

    Attributes:
        timestamp: When the trade occurred.
        bond_id: Bond identifier.
        dv01_change: Change in DV01 (positive = buy, negative = sell).
        notional_change: Estimated notional change (if available).
        price: Execution price (if available).
        cost: Transaction cost incurred.
    """

    timestamp: pd.Timestamp
    bond_id: str
    dv01_change: float
    notional_change: float | None = None
    price: float | None = None
    cost: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "bond_id": self.bond_id,
            "dv01_change": self.dv01_change,
            "notional_change": self.notional_change,
            "price": self.price,
            "cost": self.cost,
        }


@dataclass
class PortfolioBacktestResult:
    """Result of portfolio backtest.

    Attributes:
        pnl_df: DataFrame with columns: timestamp, gross_pnl, cost, net_pnl,
            cumulative_pnl, gross_dv01, net_dv01, plus per-bond P&L columns.
        trades: List of all trades executed.
        holdings_df: DataFrame of holdings over time (timestamp x bond_id).
        summary: Summary statistics dict.
    """

    pnl_df: pd.DataFrame
    trades: list[Trade] = field(default_factory=list)
    holdings_df: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        """Total net P&L."""
        return self.pnl_df["net_pnl"].sum() if "net_pnl" in self.pnl_df else 0.0

    @property
    def total_cost(self) -> float:
        """Total transaction costs."""
        return self.pnl_df["cost"].sum() if "cost" in self.pnl_df else 0.0

    @property
    def total_turnover(self) -> float:
        """Total turnover (sum of absolute DV01 changes)."""
        return sum(abs(t.dv01_change) for t in self.trades)

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame(columns=["timestamp", "bond_id", "dv01_change", "cost"])
        return pd.DataFrame([t.to_dict() for t in self.trades])


def simulate_rebalance(
    current_holdings: pd.Series,
    target_holdings: pd.Series,
    timestamp: pd.Timestamp,
    config: PortfolioBacktestConfig | None = None,
    rebalance_rule: RebalanceRule | None = None,
    dv01_panel: pd.DataFrame | None = None,
) -> tuple[list[Trade], pd.Series]:
    """Simulate rebalancing from current to target holdings.

    Args:
        current_holdings: Current DV01 by bond_id.
        target_holdings: Target DV01 by bond_id.
        timestamp: Time of rebalance.
        config: Backtest configuration for costs.
        rebalance_rule: Rules for minimum trade size, etc.
        dv01_panel: Optional panel with DV01 data for notional estimation.

    Returns:
        Tuple of (list of trades, new holdings after rebalance).
    """
    if config is None:
        config = PortfolioBacktestConfig()
    if rebalance_rule is None:
        rebalance_rule = RebalanceRule()

    # Align indices
    all_bonds = current_holdings.index.union(target_holdings.index)
    current = current_holdings.reindex(all_bonds, fill_value=0.0)
    target = target_holdings.reindex(all_bonds, fill_value=0.0)

    # Compute changes needed
    changes = target - current

    trades = []
    for bond_id, dv01_change in changes.items():
        # Skip zero or tiny trades
        if abs(dv01_change) <= rebalance_rule.min_trade_dv01:
            continue

        # Estimate notional change
        notional_change = None
        if config.use_dv01_cost_proxy:
            notional_change = abs(dv01_change) * config.dv01_to_notional_factor

        # Compute cost
        cost = _compute_trade_cost(dv01_change, notional_change, config)

        trade = Trade(
            timestamp=timestamp,
            bond_id=bond_id,
            dv01_change=dv01_change,
            notional_change=notional_change,
            cost=cost,
        )
        trades.append(trade)

    # New holdings after trades
    new_holdings = current.copy()
    for trade in trades:
        new_holdings[trade.bond_id] = current[trade.bond_id] + trade.dv01_change

    return trades, new_holdings


def _compute_trade_cost(
    dv01_change: float,
    notional_change: float | None,
    config: PortfolioBacktestConfig,
) -> float:
    """Compute transaction cost for a trade.

    Args:
        dv01_change: DV01 change (signed).
        notional_change: Estimated notional (absolute).
        config: Cost configuration.

    Returns:
        Transaction cost (always positive).
    """
    if config.cost_per_dv01 is not None:
        # Cost per DV01 model
        return abs(dv01_change) * config.cost_per_dv01

    # Cost in bps of notional
    if notional_change is not None:
        total_bps = config.cost_bps + config.slippage_bps
        return notional_change * total_bps / 10000.0

    return 0.0


def compute_pnl_from_prices(
    holdings: pd.Series,
    prices_t0: pd.Series,
    prices_t1: pd.Series,
    dv01_per_bond: pd.Series | None = None,
) -> dict:
    """Compute P&L from price changes for a single period.

    Uses DV01-based P&L: P&L ≈ -DV01 * Δyield ≈ DV01 * Δprice / price * 100

    For simplicity, if DV01 is provided, we use:
        P&L = holdings_dv01 * (price_return in bps)

    If DV01 not provided, we assume holdings are in DV01 terms and
    price changes represent yield changes (inverted).

    Args:
        holdings: DV01 holdings by bond_id.
        prices_t0: Prices at start of period.
        prices_t1: Prices at end of period.
        dv01_per_bond: Optional DV01 per bond for more accurate calc.

    Returns:
        Dict with per-bond P&L and total.
    """
    # Align all series
    all_bonds = holdings.index
    h = holdings.reindex(all_bonds, fill_value=0.0)
    p0 = prices_t0.reindex(all_bonds)
    p1 = prices_t1.reindex(all_bonds)

    # Price return in bps: (p1 - p0) / p0 * 10000
    # For bonds, price up = yield down, so long DV01 position profits
    price_return_bps = (p1 - p0) / p0 * 10000

    # P&L = DV01 * price_return_bps / 100 (convert bps to %)
    # Or more simply: P&L = DV01 * (p1 - p0) / p0 * 100
    pnl_by_bond = h * price_return_bps / 100

    # Handle NaN (missing prices)
    pnl_by_bond = pnl_by_bond.fillna(0.0)

    return {
        "pnl_by_bond": pnl_by_bond,
        "total_pnl": pnl_by_bond.sum(),
        "price_return_bps": price_return_bps,
    }


def run_portfolio_backtest(
    price_panel: pd.DataFrame,
    target_dv01: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
    rebalance_rule: RebalanceRule | None = None,
    dv01_panel: pd.DataFrame | None = None,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    price_col: str = "price",
    dv01_col: str = "dv01",
) -> PortfolioBacktestResult:
    """Run portfolio backtest with rebalancing.

    Args:
        price_panel: DataFrame with columns [datetime, bond_id, price].
            Should be in long format with one row per (datetime, bond_id).
        target_dv01: DataFrame with columns [datetime, bond_id, target_dv01].
            Specifies target DV01 at rebalance times. Times not in this
            DataFrame are non-rebalance periods.
        config: Backtest configuration.
        rebalance_rule: Rebalancing rules.
        dv01_panel: Optional DataFrame with [datetime, bond_id, dv01] for
            more accurate cost/P&L calculation.
        datetime_col: Name of datetime column.
        bond_id_col: Name of bond ID column.
        price_col: Name of price column.
        dv01_col: Name of DV01 column in dv01_panel.

    Returns:
        PortfolioBacktestResult with P&L, trades, and holdings history.

    Example:
        >>> result = run_portfolio_backtest(
        ...     price_panel=prices_df,
        ...     target_dv01=targets_df,
        ...     config=PortfolioBacktestConfig(cost_bps=1.0),
        ... )
        >>> print(f"Total P&L: {result.total_pnl:.2f}")
    """
    if config is None:
        config = PortfolioBacktestConfig()
    if rebalance_rule is None:
        rebalance_rule = RebalanceRule()

    # Pivot price panel to wide format: datetime x bond_id
    price_wide = price_panel.pivot(
        index=datetime_col, columns=bond_id_col, values=price_col
    )
    price_wide = price_wide.sort_index()

    # Get rebalance times from target_dv01
    rebalance_times = target_dv01[datetime_col].unique()
    rebalance_times = pd.DatetimeIndex(rebalance_times).sort_values()

    # Pivot target_dv01 to wide format
    target_wide = target_dv01.pivot(
        index=datetime_col, columns=bond_id_col, values="target_dv01"
    )

    # Initialize
    all_bonds = price_wide.columns.tolist()
    current_holdings = pd.Series(0.0, index=all_bonds)
    all_trades = []
    holdings_history = []
    pnl_records = []

    timestamps = price_wide.index.tolist()

    for i, ts in enumerate(timestamps):
        # Check if this is a rebalance time
        is_rebalance = ts in rebalance_times

        if is_rebalance and ts in target_wide.index:
            # Get target for this time
            target = target_wide.loc[ts].fillna(0.0)

            # Simulate rebalance
            trades, new_holdings = simulate_rebalance(
                current_holdings=current_holdings,
                target_holdings=target,
                timestamp=ts,
                config=config,
                rebalance_rule=rebalance_rule,
            )

            all_trades.extend(trades)
            current_holdings = new_holdings.reindex(all_bonds, fill_value=0.0)

            # Record rebalance cost
            rebalance_cost = sum(t.cost for t in trades)
        else:
            rebalance_cost = 0.0

        # Compute P&L from price change (if not first period)
        if i > 0:
            prev_ts = timestamps[i - 1]
            prices_t0 = price_wide.loc[prev_ts]
            prices_t1 = price_wide.loc[ts]

            pnl_result = compute_pnl_from_prices(
                holdings=current_holdings,
                prices_t0=prices_t0,
                prices_t1=prices_t1,
            )
            gross_pnl = pnl_result["total_pnl"]
        else:
            gross_pnl = 0.0

        # Record
        holdings_history.append({
            "timestamp": ts,
            **{f"dv01_{bond}": current_holdings.get(bond, 0.0) for bond in all_bonds},
        })

        pnl_records.append({
            "timestamp": ts,
            "gross_pnl": gross_pnl,
            "cost": rebalance_cost,
            "net_pnl": gross_pnl - rebalance_cost,
            "gross_dv01": current_holdings.abs().sum(),
            "net_dv01": current_holdings.sum(),
            "is_rebalance": is_rebalance,
        })

    # Build result DataFrames
    pnl_df = pd.DataFrame(pnl_records)
    pnl_df["cumulative_pnl"] = pnl_df["net_pnl"].cumsum()

    holdings_df = pd.DataFrame(holdings_history)

    # Compute summary statistics
    summary = _compute_summary(pnl_df, all_trades)

    return PortfolioBacktestResult(
        pnl_df=pnl_df,
        trades=all_trades,
        holdings_df=holdings_df,
        summary=summary,
    )


def _compute_summary(pnl_df: pd.DataFrame, trades: list[Trade]) -> dict:
    """Compute summary statistics for backtest."""
    total_pnl = pnl_df["net_pnl"].sum()
    total_cost = pnl_df["cost"].sum()
    total_gross_pnl = pnl_df["gross_pnl"].sum()

    # Turnover
    total_turnover = sum(abs(t.dv01_change) for t in trades)
    n_trades = len(trades)
    n_rebalances = pnl_df["is_rebalance"].sum()

    # Returns
    returns = pnl_df["net_pnl"]
    if len(returns) > 1 and returns.std() > 0:
        # Annualize assuming daily data (adjust as needed)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = pnl_df["cumulative_pnl"]
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    return {
        "total_pnl": total_pnl,
        "total_gross_pnl": total_gross_pnl,
        "total_cost": total_cost,
        "cost_pct_of_gross": total_cost / abs(total_gross_pnl) * 100 if total_gross_pnl != 0 else 0,
        "total_turnover_dv01": total_turnover,
        "n_trades": n_trades,
        "n_rebalances": n_rebalances,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_cumulative_pnl": cumulative.iloc[-1] if len(cumulative) > 0 else 0,
    }


def run_portfolio_backtest_from_targets(
    price_panel: pd.DataFrame,
    portfolio_targets: list,  # list[PortfolioTarget]
    config: PortfolioBacktestConfig | None = None,
    rebalance_rule: RebalanceRule | None = None,
    datetime_col: str = "datetime",
    bond_id_col: str = "bond_id",
    price_col: str = "price",
) -> PortfolioBacktestResult:
    """Run portfolio backtest from list of PortfolioTarget objects.

    Convenience wrapper that converts PortfolioTarget list to the
    target_dv01 DataFrame format.

    Args:
        price_panel: DataFrame with [datetime, bond_id, price].
        portfolio_targets: List of PortfolioTarget from aggregation.
        config: Backtest configuration.
        rebalance_rule: Rebalancing rules.
        datetime_col: Name of datetime column.
        bond_id_col: Name of bond ID column.
        price_col: Name of price column.

    Returns:
        PortfolioBacktestResult.
    """
    # Convert PortfolioTarget list to DataFrame
    records = []
    for target in portfolio_targets:
        for bond_id, dv01 in target.positions.items():
            records.append({
                datetime_col: target.timestamp,
                bond_id_col: bond_id,
                "target_dv01": dv01,
            })

    target_dv01 = pd.DataFrame(records)

    return run_portfolio_backtest(
        price_panel=price_panel,
        target_dv01=target_dv01,
        config=config,
        rebalance_rule=rebalance_rule,
        datetime_col=datetime_col,
        bond_id_col=bond_id_col,
        price_col=price_col,
    )


def compute_costs_summary(
    trades: list[Trade],
    by: str = "bond",
) -> pd.DataFrame:
    """Summarize transaction costs.

    Args:
        trades: List of trades.
        by: Group by "bond", "timestamp", or "total".

    Returns:
        DataFrame with cost summary.
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame([t.to_dict() for t in trades])

    if by == "bond":
        return df.groupby("bond_id").agg({
            "dv01_change": lambda x: x.abs().sum(),
            "cost": "sum",
        }).rename(columns={"dv01_change": "turnover_dv01"})

    elif by == "timestamp":
        return df.groupby("timestamp").agg({
            "dv01_change": lambda x: x.abs().sum(),
            "cost": "sum",
        }).rename(columns={"dv01_change": "turnover_dv01"})

    else:  # total
        return pd.DataFrame([{
            "total_turnover_dv01": df["dv01_change"].abs().sum(),
            "total_cost": df["cost"].sum(),
            "n_trades": len(df),
        }])
