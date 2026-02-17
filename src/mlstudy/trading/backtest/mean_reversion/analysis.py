"""Post-analysis functions for MR backtest results.

Pure-compute module (no I/O, no matplotlib).
All functions take :class:`MRBacktestResults` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .single_backtest.results import MRBacktestResults
from .types import (
    CODE_NAMES,
    STATE_FLAT,
    STATE_LONG,
    STATE_SHORT,
    TRADE_ENTRY,
    TRADE_EXIT_SL,
    TRADE_EXIT_TIME,
    TRADE_EXIT_TP,
)
from ..metrics import (
    BacktestMetrics,
    compute_avg_holding_period,
    compute_hit_rate,
    compute_max_drawdown,
    compute_n_trades,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_tail_stats,
    compute_turnover,
    compute_win_loss_stats,
)

# Maps tr_type int to human-readable exit type string
_EXIT_TYPE_NAMES: dict[int, str] = {
    TRADE_EXIT_TP: "tp",
    TRADE_EXIT_SL: "sl",
    TRADE_EXIT_TIME: "time",
}


# ---------------------------------------------------------------------------
# Bridge: per-bar DataFrame
# ---------------------------------------------------------------------------


def to_dataframe(
    res: MRBacktestResults,
    datetimes: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a per-bar DataFrame from backtest results.

    Columns: equity, pnl, cumulative_pnl, position, state, code, holding.
    If *datetimes* is provided, a ``datetime`` column is added.
    """
    T = len(res.equity)
    # position: scalar state indicator (-1/0/+1)
    position = res.state.copy().astype(np.float64)

    df = pd.DataFrame(
        {
            "equity": res.equity,
            "pnl": res.pnl,
            "cumulative_pnl": np.cumsum(res.pnl),
            "position": position,
            "state": res.state,
            "code": res.codes,
            "holding": res.holding,
        }
    )
    if datetimes is not None:
        df["datetime"] = datetimes[:T]
    return df


# ---------------------------------------------------------------------------
# Performance metrics (wraps metrics.py functions)
# ---------------------------------------------------------------------------


def compute_performance_metrics(res: MRBacktestResults) -> BacktestMetrics:
    """Compute all standard performance metrics from *res*."""
    returns = pd.Series(res.pnl, dtype=np.float64)
    cumulative_pnl = pd.Series(np.cumsum(res.pnl), dtype=np.float64)
    position = pd.Series(res.state, dtype=np.float64)

    total_pnl = float(returns.sum())
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())

    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    max_dd, max_dd_dur = compute_max_drawdown(cumulative_pnl)

    annual_return = mean_ret * 252
    calmar = abs(annual_return / max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Turnover: traded_notional from position deltas × mid prices
    # gross_notional from abs(positions) × mid prices
    pos_delta = np.abs(np.diff(res.positions, axis=0, prepend=0.0))
    mid_for_turnover = np.ones_like(res.positions)  # normalized
    if res.positions.ndim == 2:
        traded_notional = pd.Series(
            np.sum(pos_delta * mid_for_turnover, axis=1), dtype=np.float64
        )
        gross_notional = pd.Series(
            np.sum(np.abs(res.positions) * mid_for_turnover, axis=1),
            dtype=np.float64,
        )
    else:
        traded_notional = pd.Series(pos_delta.ravel(), dtype=np.float64)
        gross_notional = pd.Series(
            np.abs(res.positions).ravel(), dtype=np.float64
        )
    turnover = compute_turnover(traded_notional, gross_notional)

    avg_hold = compute_avg_holding_period(position)
    hit_rate = compute_hit_rate(returns, position)
    profit_factor = compute_profit_factor(returns)
    avg_win, avg_loss, win_loss = compute_win_loss_stats(returns)
    skew, kurt, var_95, cvar_95 = compute_tail_stats(returns)
    n_trades = compute_n_trades(position)
    pct_in_market = float((position != 0).mean())

    return BacktestMetrics(
        total_pnl=total_pnl,
        mean_daily_return=mean_ret,
        std_daily_return=std_ret,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        calmar_ratio=calmar,
        turnover_annual=turnover,
        avg_holding_period=avg_hold,
        hit_rate=hit_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss,
        skewness=skew,
        kurtosis=kurt,
        var_95=var_95,
        cvar_95=cvar_95,
        n_trades=n_trades,
        pct_time_in_market=pct_in_market,
    )


# ---------------------------------------------------------------------------
# Round-trip pairing
# ---------------------------------------------------------------------------


def compute_round_trips(res: MRBacktestResults) -> pd.DataFrame:
    """Pair each entry trade with its corresponding exit.

    Returns a DataFrame with one row per round-trip:
        entry_bar, exit_bar, side, holding_bars, exit_type,
        entry_cost, exit_cost, total_cost,
        entry_pkg_yield, exit_pkg_yield, yield_delta,
        pnl, slippage_entry, slippage_exit
    """
    rows: list[dict] = []
    pending_entry: dict | None = None

    for i in range(res.n_trades):
        ttype = int(res.tr_type[i])
        if ttype == TRADE_ENTRY:
            pending_entry = {
                "idx": i,
                "bar": int(res.tr_bar[i]),
                "side": int(res.tr_side[i]),
                "cost": float(res.tr_cost[i]),
                "pkg_yield": float(res.tr_pkg_yield[i]),
                "vwaps": res.tr_vwaps[i].copy(),
                "mids": res.tr_mids[i].copy(),
            }
        elif pending_entry is not None:
            # This is an exit trade closing the pending entry
            entry = pending_entry
            exit_bar = int(res.tr_bar[i])
            exit_cost = float(res.tr_cost[i])
            exit_pkg_yield = float(res.tr_pkg_yield[i])

            # slippage: mean |vwap - mid| across legs
            slip_entry = float(np.mean(np.abs(entry["vwaps"] - entry["mids"])))
            slip_exit = float(np.mean(np.abs(res.tr_vwaps[i] - res.tr_mids[i])))

            # PnL: sum of pnl array between entry bar and exit bar (inclusive)
            entry_bar = entry["bar"]
            pnl_slice = res.pnl[entry_bar : exit_bar + 1]
            trade_pnl = float(np.sum(pnl_slice))

            rows.append(
                {
                    "entry_bar": entry_bar,
                    "exit_bar": exit_bar,
                    "side": entry["side"],
                    "holding_bars": exit_bar - entry_bar,
                    "exit_type": _EXIT_TYPE_NAMES.get(ttype, "unknown"),
                    "entry_cost": entry["cost"],
                    "exit_cost": exit_cost,
                    "total_cost": entry["cost"] + exit_cost,
                    "entry_pkg_yield": entry["pkg_yield"],
                    "exit_pkg_yield": exit_pkg_yield,
                    "yield_delta": exit_pkg_yield - entry["pkg_yield"],
                    "pnl": trade_pnl,
                    "slippage_entry": slip_entry,
                    "slippage_exit": slip_exit,
                }
            )
            pending_entry = None

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def compute_code_distribution(res: MRBacktestResults) -> dict[str, int]:
    """Count of each attempt code across all T bars, keyed by CODE_NAMES."""
    unique, counts = np.unique(res.codes, return_counts=True)
    return {
        CODE_NAMES.get(int(code), f"UNKNOWN_{code}"): int(cnt)
        for code, cnt in zip(unique, counts)
    }


def compute_state_distribution(res: MRBacktestResults) -> dict[str, float]:
    """Fraction of bars in each state: flat, long, short."""
    T = len(res.state)
    if T == 0:
        return {"flat": 0.0, "long": 0.0, "short": 0.0}
    return {
        "flat": float(np.sum(res.state == STATE_FLAT)) / T,
        "long": float(np.sum(res.state == STATE_LONG)) / T,
        "short": float(np.sum(res.state == STATE_SHORT)) / T,
    }


# ---------------------------------------------------------------------------
# Exit-type stats
# ---------------------------------------------------------------------------


def compute_exit_type_stats(round_trips: pd.DataFrame) -> pd.DataFrame:
    """Per exit-type breakdown: count, win_rate, mean_pnl, mean_holding_bars, mean_cost."""
    if round_trips.empty:
        return pd.DataFrame(
            columns=["count", "win_rate", "mean_pnl", "mean_holding_bars", "mean_cost"]
        )

    rows = []
    for exit_type, group in round_trips.groupby("exit_type"):
        rows.append(
            {
                "exit_type": exit_type,
                "count": len(group),
                "win_rate": float((group["pnl"] > 0).mean()) if len(group) > 0 else 0.0,
                "mean_pnl": float(group["pnl"].mean()),
                "mean_holding_bars": float(group["holding_bars"].mean()),
                "mean_cost": float(group["total_cost"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("exit_type")


# ---------------------------------------------------------------------------
# Execution quality
# ---------------------------------------------------------------------------


def compute_execution_quality(res: MRBacktestResults) -> pd.DataFrame:
    """Per-trade slippage: |vwap - mid| per leg, aggregated as mean/median/max."""
    if res.n_trades == 0:
        return pd.DataFrame(columns=["trade_idx", "leg", "slippage"])

    rows: list[dict] = []
    for i in range(res.n_trades):
        slippage_per_leg = np.abs(res.tr_vwaps[i] - res.tr_mids[i])
        for leg_idx, slip in enumerate(slippage_per_leg):
            rows.append(
                {
                    "trade_idx": i,
                    "leg": leg_idx,
                    "slippage": float(slip),
                }
            )

    df = pd.DataFrame(rows)

    # Aggregate summary
    summary = df.groupby("leg")["slippage"].agg(["mean", "median", "max"])
    return summary


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(res: MRBacktestResults) -> None:
    """Formatted text report to stdout."""
    metrics = compute_performance_metrics(res)
    rt = compute_round_trips(res)
    code_dist = compute_code_distribution(res)
    state_dist = compute_state_distribution(res)

    print("=" * 60)
    print("  MR Backtest Summary")
    print("=" * 60)

    # Performance metrics
    print("\n--- Performance Metrics ---")
    print(f"  Total PnL:           {metrics.total_pnl:>12.4f}")
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>12.4f}")
    print(f"  Sortino Ratio:       {metrics.sortino_ratio:>12.4f}")
    print(f"  Max Drawdown:        {metrics.max_drawdown:>12.4f}")
    print(f"  Max DD Duration:     {metrics.max_drawdown_duration:>12d} bars")
    print(f"  Calmar Ratio:        {metrics.calmar_ratio:>12.4f}")
    print(f"  Hit Rate:            {metrics.hit_rate:>12.4f}")
    print(f"  Profit Factor:       {metrics.profit_factor:>12.4f}")
    print(f"  Avg Win:             {metrics.avg_win:>12.4f}")
    print(f"  Avg Loss:            {metrics.avg_loss:>12.4f}")
    print(f"  Win/Loss Ratio:      {metrics.win_loss_ratio:>12.4f}")
    print(f"  Skewness:            {metrics.skewness:>12.4f}")
    print(f"  Kurtosis:            {metrics.kurtosis:>12.4f}")
    print(f"  VaR 95:              {metrics.var_95:>12.4f}")
    print(f"  CVaR 95:             {metrics.cvar_95:>12.4f}")
    print(f"  # Trades:            {metrics.n_trades:>12d}")
    print(f"  % Time in Market:    {metrics.pct_time_in_market:>12.4f}")
    print(f"  Turnover (annual):   {metrics.turnover_annual:>12.4f}")
    print(f"  Avg Holding Period:  {metrics.avg_holding_period:>12.4f} bars")

    # Round-trip stats
    print(f"\n--- Round Trips ({len(rt)} total) ---")
    if not rt.empty:
        exit_stats = compute_exit_type_stats(rt)
        print(exit_stats.to_string(float_format="{:.4f}".format))

    # Code distribution
    print("\n--- Code Distribution ---")
    for name, cnt in sorted(code_dist.items(), key=lambda x: -x[1]):
        print(f"  {name:<30s} {cnt:>6d}")

    # State distribution
    print("\n--- State Distribution ---")
    for name, frac in state_dist.items():
        print(f"  {name:<10s} {frac:>8.4f}")

    print("=" * 60)
