"""Portfolio sweep executor.

Uses ``common.sweep.sweep_dispatch`` for parallel execution and
``common.sweep.sweep_persist`` (via ``PortfolioSweepPersister``) for saving.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from mlstudy.trading.backtest.metrics.portfolio_metrics_calculator import PortfolioMetricsCalculator
from mlstudy.trading.backtest.portfolio.single_backtest.engine import run_backtest
from mlstudy.trading.backtest.common.sweep.sweep_dispatch import dispatch
from mlstudy.trading.backtest.common.sweep.sweep_persist import summary_table
from .sweep_persist import PortfolioSweepPersister
from mlstudy.trading.backtest.common.sweep.sweep_rank import RankingPlan, SweepRanker
from mlstudy.trading.backtest.common.sweep.sweep_types import (
    SweepResultLight,
    SweepError,
    SweepResult,
    SweepScenario,
    SweepSummary,
)


# ---------------------------------------------------------------------------
# run-one callback for dispatch
# ---------------------------------------------------------------------------

def _run_one_portfolio(
    scenario_idx: int,
    scenario: SweepScenario,
    market_data: dict,
    mode: str,
) -> SweepResultLight | SweepResult:
    try:
        res = run_backtest(cfg=scenario.cfg, **market_data)
        bar_df = res.close_bar_df if res.close_bar_df is not None else res.bar_df
        metrics = PortfolioMetricsCalculator(bar_df, res.trade_df).compute_all()

        if mode == "metrics_only":
            return SweepResultLight(
                scenario_idx=scenario_idx,
                scenario=scenario,
                metrics=metrics,
            )

        return SweepResult(
            scenario_idx=scenario_idx,
            scenario=scenario,
            results=res,
            metrics=metrics,
        )
    except Exception as exc:
        raise SweepError(scenario_idx, scenario.cfg) from exc


# ---------------------------------------------------------------------------
# PortfolioSweepExecutor
# ---------------------------------------------------------------------------


class PortfolioSweepExecutor:
    @staticmethod
    def run_sweep(
        scenarios: list[SweepScenario],
        *,
        # Instrument market L2
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        mid_px,
        # Risk
        dv01,
        # Signals
        fair_price,
        zscore,
        adf_p_value,
        # Static meta
        tradable,
        pos_limits_long,
        pos_limits_short,
        # Optional meta
        maturity=None,
        issuer_bucket=None,
        maturity_bucket=None,
        # Optional bucket caps
        issuer_dv01_caps=None,
        mat_bucket_dv01_caps=None,
        # Hedge arrays (all-or-nothing)
        hedge_bid_px=None,
        hedge_bid_sz=None,
        hedge_ask_px=None,
        hedge_ask_sz=None,
        hedge_mid_px=None,
        hedge_dv01=None,
        hedge_ratios=None,
        # Optional context
        datetimes=None,
        # Sweep control
        backend: str = "serial",
        n_workers: int | None = None,
        chunk_size: int | None = None,
        mode: str = "full",
        keep_top_k_full: int = 0,
        save_top_full_dir: str | Path | None = None,
        ranking_plan: RankingPlan | None = None,
    ) -> list[SweepResult] | list[SweepResultLight] | SweepSummary:
        """Run a parameter sweep over portfolio backtest scenarios.

        Parameters
        ----------
        scenarios : list[SweepScenario]
            Scenarios built via ``ScenarioBuilder.make_scenarios``.
        bid_px, bid_sz, ask_px, ask_sz : (T, B, L)
            Instrument L2 order book.
        mid_px : (T, B)
        dv01 : (T, B)
        fair_price : (T, B)
        zscore : (T, B)
        adf_p_value : (T, B)
        tradable : (B,)
        pos_limits_long, pos_limits_short : (B,)
        hedge_* : optional hedge arrays
        backend : "serial" | "thread" | "process"
        mode : "full" | "metrics_only"
        keep_top_k_full : int
            If > 0 and mode="metrics_only", re-run top-k in full mode.
        ranking_plan : RankingPlan, optional

        Returns
        -------
        list[SweepResult] | list[SweepResultLight] | SweepSummary
        """
        market_data: dict[str, Any] = dict(
            bid_px=bid_px,
            bid_sz=bid_sz,
            ask_px=ask_px,
            ask_sz=ask_sz,
            mid_px=mid_px,
            dv01=dv01,
            fair_price=fair_price,
            zscore=zscore,
            adf_p_value=adf_p_value,
            tradable=tradable,
            pos_limits_long=pos_limits_long,
            pos_limits_short=pos_limits_short,
        )
        # Optional arrays
        for name, val in [
            ("maturity", maturity),
            ("issuer_bucket", issuer_bucket),
            ("maturity_bucket", maturity_bucket),
            ("issuer_dv01_caps", issuer_dv01_caps),
            ("mat_bucket_dv01_caps", mat_bucket_dv01_caps),
            ("hedge_bid_px", hedge_bid_px),
            ("hedge_bid_sz", hedge_bid_sz),
            ("hedge_ask_px", hedge_ask_px),
            ("hedge_ask_sz", hedge_ask_sz),
            ("hedge_mid_px", hedge_mid_px),
            ("hedge_dv01", hedge_dv01),
            ("hedge_ratios", hedge_ratios),
            ("datetimes", datetimes),
        ]:
            if val is not None:
                market_data[name] = val

        workers = n_workers or os.cpu_count() or 1

        n = len(scenarios)
        csize = chunk_size if chunk_size is not None else max(10, n // (10 * workers))
        csize = max(1, csize)

        indexed = list(enumerate(scenarios))

        if mode == "full":
            return dispatch(indexed, market_data, _run_one_portfolio, backend, workers, csize, "full")

        if mode != "metrics_only":
            raise ValueError(f"Unknown mode {mode!r}; choose from 'full', 'metrics_only'")

        metrics_results = dispatch(indexed, market_data, _run_one_portfolio, backend, workers, csize, "metrics_only")

        if keep_top_k_full <= 0:
            return metrics_results

        if ranking_plan is None:
            ranking_plan = RankingPlan()

        ranked = SweepRanker.rank_scenarios(metrics_results, ranking_plan)
        top_k = ranked[:keep_top_k_full]

        top_indexed = [(r.scenario_idx, r.scenario) for r in top_k]
        top_full_results = dispatch(top_indexed, market_data, _run_one_portfolio, backend, workers, csize, "full")

        rank_order = {r.scenario_idx: i for i, r in enumerate(top_k)}
        top_full_results.sort(key=lambda r: rank_order[r.scenario_idx])

        if save_top_full_dir is not None:
            PortfolioSweepPersister.save_top_full(top_full_results, save_top_full_dir)

        ranked_all = SweepRanker.rank_scenarios(metrics_results, ranking_plan)
        return SweepSummary(all_metrics=ranked_all, top_full=top_full_results)

    @staticmethod
    def summary_table(
        results: list[SweepResult] | list[SweepResultLight],
    ) -> pd.DataFrame:
        return summary_table(results)
