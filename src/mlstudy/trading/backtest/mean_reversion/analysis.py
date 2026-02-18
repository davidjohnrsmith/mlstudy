# """Post-analysis functions for MR backtest results.
#
# Pure-compute module (no I/O, no matplotlib).
# All functions take :class:`MRBacktestResults` directly.
# """
#
# from __future__ import annotations
#
# import numpy as np
# import pandas as pd
#
# from .single_backtest.results import MRBacktestResults
# from .single_backtest.state import CODE_NAMES, ActionCode, State, TradeType
# from ..metrics.metrics import BacktestMetrics
# from ..metrics.metrics_calculator import MetricsCalculator
#
# # Maps tr_type int to human-readable exit type string
# _EXIT_TYPE_NAMES: dict[int, str] = {
#     TradeType.TRADE_EXIT_TP: "tp",
#     TradeType.TRADE_EXIT_SL: "sl",
#     TradeType.TRADE_EXIT_TIME: "time",
# }
#
#
# # ---------------------------------------------------------------------------
# # Bridge: per-bar DataFrame
# # ---------------------------------------------------------------------------
#
#
# def to_dataframe(
#     res: MRBacktestResults,
#     datetimes: pd.DatetimeIndex | None = None,
# ) -> pd.DataFrame:
#     """Build a per-bar DataFrame from backtest results.
#
#     Columns: equity, pnl, cumulative_pnl, position, state, code, holding.
#     If *datetimes* is provided, a ``datetime`` column is added.
#
#     Delegates to :meth:`MRBacktestResults.to_bar_df`.
#     """
#     df = res.to_bar_df()
#     # If datetimes were passed explicitly but aren't on the results object,
#     # add them to maintain backward compatibility.
#     if datetimes is not None and "datetime" not in df.columns:
#         df.insert(0, "datetime", datetimes[: len(df)])
#     return df
#
#
# # ---------------------------------------------------------------------------
# # Performance metrics (wraps metrics.py functions)
# # ---------------------------------------------------------------------------
#
#
# def compute_performance_metrics(res: MRBacktestResults) -> BacktestMetrics:
#     """Compute all standard performance metrics from *res*."""
#     return MetricsCalculator(res.bar_df, res.trade_df).compute_all()
#
#
# # ---------------------------------------------------------------------------
# # Round-trip pairing
# # ---------------------------------------------------------------------------
#
#
# def compute_round_trips(res: MRBacktestResults) -> pd.DataFrame:
#     """Pair each entry trade with its corresponding exit.
#
#     Returns a DataFrame with one row per round-trip:
#         entry_bar, exit_bar, side, holding_bars, exit_type,
#         entry_cost, exit_cost, total_cost,
#         entry_pkg_yield, exit_pkg_yield, yield_delta,
#         pnl, slippage_entry, slippage_exit
#     """
#     rows: list[dict] = []
#     pending_entry: dict | None = None
#
#     for i in range(res.n_trades):
#         ttype = int(res.tr_type[i])
#         if ttype == TradeType.TRADE_ENTRY:
#             pending_entry = {
#                 "idx": i,
#                 "bar": int(res.tr_bar[i]),
#                 "side": int(res.tr_side[i]),
#                 "cost": float(res.tr_cost[i]),
#                 "pkg_yield": float(res.tr_pkg_yield[i]),
#                 "vwaps": res.tr_vwaps[i].copy(),
#                 "mids": res.tr_mids[i].copy(),
#             }
#         elif pending_entry is not None:
#             # This is an exit trade closing the pending entry
#             entry = pending_entry
#             exit_bar = int(res.tr_bar[i])
#             exit_cost = float(res.tr_cost[i])
#             exit_pkg_yield = float(res.tr_pkg_yield[i])
#
#             # slippage: mean |vwap - mid| across legs
#             slip_entry = float(np.mean(np.abs(entry["vwaps"] - entry["mids"])))
#             slip_exit = float(np.mean(np.abs(res.tr_vwaps[i] - res.tr_mids[i])))
#
#             # PnL: sum of pnl array between entry bar and exit bar (inclusive)
#             entry_bar = entry["bar"]
#             pnl_slice = res.pnl[entry_bar : exit_bar + 1]
#             trade_pnl = float(np.sum(pnl_slice))
#
#             rows.append(
#                 {
#                     "entry_bar": entry_bar,
#                     "exit_bar": exit_bar,
#                     "side": entry["side"],
#                     "holding_bars": exit_bar - entry_bar,
#                     "exit_type": _EXIT_TYPE_NAMES.get(ttype, "unknown"),
#                     "entry_cost": entry["cost"],
#                     "exit_cost": exit_cost,
#                     "total_cost": entry["cost"] + exit_cost,
#                     "entry_pkg_yield": entry["pkg_yield"],
#                     "exit_pkg_yield": exit_pkg_yield,
#                     "yield_delta": exit_pkg_yield - entry["pkg_yield"],
#                     "pnl": trade_pnl,
#                     "slippage_entry": slip_entry,
#                     "slippage_exit": slip_exit,
#                 }
#             )
#             pending_entry = None
#
#     return pd.DataFrame(rows)
#
#
# # ---------------------------------------------------------------------------
# # Distributions
# # ---------------------------------------------------------------------------
#
#
# def compute_code_distribution(res: MRBacktestResults) -> dict[str, int]:
#     """Count of each attempt code across all T bars, keyed by CODE_NAMES."""
#     unique, counts = np.unique(res.codes, return_counts=True)
#     return {
#         CODE_NAMES.get(int(code), f"UNKNOWN_{code}"): int(cnt)
#         for code, cnt in zip(unique, counts)
#     }
#
#
# def compute_state_distribution(res: MRBacktestResults) -> dict[str, float]:
#     """Fraction of bars in each state: flat, long, short."""
#     T = len(res.state)
#     if T == 0:
#         return {"flat": 0.0, "long": 0.0, "short": 0.0}
#     return {
#         "flat": float(np.sum(res.state == State.STATE_FLAT)) / T,
#         "long": float(np.sum(res.state == State.STATE_LONG)) / T,
#         "short": float(np.sum(res.state == State.STATE_SHORT)) / T,
#     }
#
#
# # ---------------------------------------------------------------------------
# # Exit-type stats
# # ---------------------------------------------------------------------------
#
#
# def compute_exit_type_stats(round_trips: pd.DataFrame) -> pd.DataFrame:
#     """Per exit-type breakdown: count, win_rate, mean_pnl, mean_holding_bars, mean_cost."""
#     if round_trips.empty:
#         return pd.DataFrame(
#             columns=["count", "win_rate", "mean_pnl", "mean_holding_bars", "mean_cost"]
#         )
#
#     rows = []
#     for exit_type, group in round_trips.groupby("exit_type"):
#         rows.append(
#             {
#                 "exit_type": exit_type,
#                 "count": len(group),
#                 "win_rate": float((group["pnl"] > 0).mean()) if len(group) > 0 else 0.0,
#                 "mean_pnl": float(group["pnl"].mean()),
#                 "mean_holding_bars": float(group["holding_bars"].mean()),
#                 "mean_cost": float(group["total_cost"].mean()),
#             }
#         )
#     return pd.DataFrame(rows).set_index("exit_type")
#
#
# # ---------------------------------------------------------------------------
# # Execution quality
# # ---------------------------------------------------------------------------
#
#
# def compute_execution_quality(res: MRBacktestResults) -> pd.DataFrame:
#     """Per-trade slippage: |vwap - mid| per leg, aggregated as mean/median/max."""
#     if res.n_trades == 0:
#         return pd.DataFrame(columns=["trade_idx", "leg", "slippage"])
#
#     rows: list[dict] = []
#     for i in range(res.n_trades):
#         slippage_per_leg = np.abs(res.tr_vwaps[i] - res.tr_mids[i])
#         for leg_idx, slip in enumerate(slippage_per_leg):
#             rows.append(
#                 {
#                     "trade_idx": i,
#                     "leg": leg_idx,
#                     "slippage": float(slip),
#                 }
#             )
#
#     df = pd.DataFrame(rows)
#
#     # Aggregate summary
#     summary = df.groupby("leg")["slippage"].agg(["mean", "median", "max"])
#     return summary
#
#
# # ---------------------------------------------------------------------------
# # Text summary
# # ---------------------------------------------------------------------------
#
#
# def print_summary(res: MRBacktestResults) -> None:
#     """Formatted text report to stdout."""
#     metrics = compute_performance_metrics(res)
#     rt = compute_round_trips(res)
#     code_dist = compute_code_distribution(res)
#     state_dist = compute_state_distribution(res)
#
#     print("=" * 60)
#     print("  MR Backtest Summary")
#     print("=" * 60)
#
#     # Performance metrics
#     print("\n--- Performance Metrics ---")
#     print(f"  Total PnL:           {metrics.total_pnl:>12.4f}")
#     print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>12.4f}")
#     print(f"  Sortino Ratio:       {metrics.sortino_ratio:>12.4f}")
#     print(f"  Max Drawdown:        {metrics.max_drawdown:>12.4f}")
#     print(f"  Max DD Duration:     {metrics.max_drawdown_duration:>12d} bars")
#     print(f"  Calmar Ratio:        {metrics.calmar_ratio:>12.4f}")
#     print(f"  Hit Rate:            {metrics.hit_rate:>12.4f}")
#     print(f"  Profit Factor:       {metrics.profit_factor:>12.4f}")
#     print(f"  Avg Win:             {metrics.avg_win:>12.4f}")
#     print(f"  Avg Loss:            {metrics.avg_loss:>12.4f}")
#     print(f"  Win/Loss Ratio:      {metrics.win_loss_ratio:>12.4f}")
#     print(f"  Skewness:            {metrics.skewness:>12.4f}")
#     print(f"  Kurtosis:            {metrics.kurtosis:>12.4f}")
#     print(f"  VaR 95:              {metrics.var_95:>12.4f}")
#     print(f"  CVaR 95:             {metrics.cvar_95:>12.4f}")
#     print(f"  # Trades:            {metrics.n_trades:>12d}")
#     print(f"  % Time in Market:    {metrics.pct_time_in_market:>12.4f}")
#     print(f"  Turnover (annual):   {metrics.turnover_annual:>12.4f}")
#     print(f"  Avg Holding Period:  {metrics.avg_holding_period:>12.4f} bars")
#
#     # Round-trip stats
#     print(f"\n--- Round Trips ({len(rt)} total) ---")
#     if not rt.empty:
#         exit_stats = compute_exit_type_stats(rt)
#         print(exit_stats.to_string(float_format="{:.4f}".format))
#
#     # Code distribution
#     print("\n--- Code Distribution ---")
#     for name, cnt in sorted(code_dist.items(), key=lambda x: -x[1]):
#         print(f"  {name:<30s} {cnt:>6d}")
#
#     # State distribution
#     print("\n--- State Distribution ---")
#     for name, frac in state_dist.items():
#         print(f"  {name:<10s} {frac:>8.4f}")
#
#     print("=" * 60)
