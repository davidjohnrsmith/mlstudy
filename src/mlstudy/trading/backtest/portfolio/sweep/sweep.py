"""Portfolio sweep executor.

Uses ``common.sweep.sweep_dispatch`` for parallel execution and
``common.sweep.sweep_persist`` (via ``PortfolioSweepPersister``) for saving.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

from mlstudy.trading.backtest.metrics.portfolio_metrics_calculator import PortfolioMetricsCalculator
from mlstudy.trading.backtest.portfolio.single_backtest.engine import run_backtest, run_backtest_chunked
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
        chunk_params = market_data.get("_chunk_params")
        if chunk_params is not None:
            # Chunked mode: re-create iterator from stored params
            loader = chunk_params["loader"]
            chunks_iter = loader.load_chunked(**chunk_params["load_kwargs"])
            data_chunks = (md.to_dict() for md in chunks_iter)
            res = run_backtest_chunked(data_chunks=data_chunks, cfg=scenario.cfg)
        else:
            res = run_backtest(cfg=scenario.cfg, **market_data)
        use_close = res.close_bar_df is not None
        bar_df = res.close_bar_df if use_close else res.bar_df
        init_eq = res.initial_capital if use_close else None
        metrics = PortfolioMetricsCalculator(
            bar_df, res.trade_df,
            initial_equity=init_eq,
            hedge_ratios=res.hedge_ratios,
            dv01=res.dv01,
            hedge_dv01=res.hedge_dv01,
            hedge_mid_px=res.hedge_mid_px,
            hedge_bid_px=res.hedge_bid_px,
            hedge_ask_px=res.hedge_ask_px,
            instrument_ids=res.instrument_ids,
        ).compute_all()

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
        # -- Market data (shared across all scenarios) -----------------
        # Instrument L2 book
        bid_px,             # (T, B, L)
        bid_sz,             # (T, B, L)
        ask_px,             # (T, B, L)
        ask_sz,             # (T, B, L)
        mid_px,             # (T, B)
        # Risk
        dv01,               # (T, B)
        # Signals
        fair_price,         # (T, B)
        zscore,             # (T, B)
        adf_p_value,        # (T, B)
        # Static instrument meta
        tradable,           # (B,)
        pos_limits_long,    # (B,)
        pos_limits_short,   # (B,)
        max_trade_notional_inc,  # (B,)
        max_trade_notional_dec,  # (B,)
        # Time-varying / static meta
        maturity,           # (T, B) or (B,) or None
        issuer_bucket,      # (B,) or None
        maturity_bucket,    # (T, B) or (B,) or None
        # Bucket caps
        issuer_dv01_caps,   # (n_issuers,) or None
        mat_bucket_dv01_caps,  # (n_buckets,) or None
        # Instrument identifiers
        instrument_ids,
        # Per-instrument execution
        qty_step,           # (B,) or scalar
        # Hedge L2 book
        hedge_bid_px,       # (T, H, L) or None
        hedge_bid_sz,       # (T, H, L) or None
        hedge_ask_px,       # (T, H, L) or None
        hedge_ask_sz,       # (T, H, L) or None
        hedge_mid_px,       # (T, H) or None
        # Hedge risk / ratios
        hedge_dv01,         # (T, H) or None
        hedge_ratios,       # (T, B, H) or None
        # Per-hedge execution
        hedge_qty_step,     # (H,) or None
        # Timestamps
        datetimes,
        # -- Sweep control (parallel dispatch of scenarios) -------------
        backend: str = "serial",
        n_workers: int | None = None,
        chunk_size: int | None = None,   # scenarios per dispatch batch
        mode: str = "full",
        keep_top_k_full: int = 0,
        save_top_full_dir: str | Path | None = None,
        ranking_plan: RankingPlan | None = None,
        # -- Data chunking (load market data in chunks) ----------------
        chunk_params: dict | None = None,  # passed to loader.load_chunked()
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
            max_trade_notional_inc=max_trade_notional_inc,
            max_trade_notional_dec=max_trade_notional_dec,
            maturity=maturity,
            issuer_bucket=issuer_bucket,
            maturity_bucket=maturity_bucket,
            issuer_dv01_caps=issuer_dv01_caps,
            mat_bucket_dv01_caps=mat_bucket_dv01_caps,
            instrument_ids=instrument_ids,
            qty_step=qty_step,
            hedge_bid_px=hedge_bid_px,
            hedge_bid_sz=hedge_bid_sz,
            hedge_ask_px=hedge_ask_px,
            hedge_ask_sz=hedge_ask_sz,
            hedge_mid_px=hedge_mid_px,
            hedge_dv01=hedge_dv01,
            hedge_ratios=hedge_ratios,
            hedge_qty_step=hedge_qty_step,
            datetimes=datetimes,
        )
        if chunk_params is not None:
            market_data["_chunk_params"] = chunk_params

        workers = n_workers or os.cpu_count() or 1

        n = len(scenarios)
        csize = chunk_size if chunk_size is not None else max(10, n // (10 * workers))
        csize = max(1, csize)

        chunked = chunk_params is not None
        logger.info(
            "run_sweep: %d scenarios, backend=%s, workers=%d, chunk_size=%d, "
            "mode=%s, chunked=%s",
            n, backend, workers, csize, mode, chunked,
        )

        indexed = list(enumerate(scenarios))

        if mode == "full":
            return dispatch(indexed, market_data, _run_one_portfolio, backend, workers, csize, "full")

        if mode != "metrics_only":
            raise ValueError(f"Unknown mode {mode!r}; choose from 'full', 'metrics_only'")

        metrics_results = dispatch(indexed, market_data, _run_one_portfolio, backend, workers, csize, "metrics_only")
        logger.info("Metrics pass done: %d results", len(metrics_results))

        if keep_top_k_full <= 0:
            return metrics_results

        if ranking_plan is None:
            ranking_plan = RankingPlan()

        ranked = SweepRanker.rank_scenarios(metrics_results, ranking_plan)
        top_k = ranked[:keep_top_k_full]

        top_indexed = [(r.scenario_idx, r.scenario) for r in top_k]
        logger.info("Re-running top %d scenarios in full mode", len(top_indexed))
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
