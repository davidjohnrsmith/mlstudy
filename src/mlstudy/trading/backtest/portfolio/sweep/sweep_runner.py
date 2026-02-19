"""End-to-end sweep interface: load config, build scenarios, run sweep, persist results.

Usage from a script or notebook::

    from mlstudy.trading.backtest.portfolio.sweep.sweep_runner import PortfolioSweepRunner

    # Option A: by YAML path
    summary = PortfolioSweepRunner.run_sweep_from_config("configs/portfolio_grid_v1.yaml")

    # Option B: by config-map name
    summary = PortfolioSweepRunner.run_sweep_from_config("portfolio_grid_v1")

    # Option C: with market data already loaded
    summary = PortfolioSweepRunner.run_sweep_from_config(
        "portfolio_grid_v1", market_data=my_market_data
    )

The function returns a ``SweepRunResult`` that bundles:
- the parsed config
- all raw results (list or SweepSummary)
- the summary table as a DataFrame
- the output directory where artifacts are saved
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..configs.sweep_config import PortfolioSweepConfig
from ..configs.utils import _resolve_config
from .sweep import PortfolioSweepExecutor
from .sweep_persist import PortfolioSweepPersister
from ...common.sweep.sweep_build import ScenarioBuilder
from mlstudy.trading.backtest.common.sweep.sweep_types import (
    SweepResult,
    SweepResultLight,
    SweepSummary,
)

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "bid_px", "bid_sz", "ask_px", "ask_sz", "mid_px",
    "dv01", "fair_price", "zscore", "adf_p_value",
    "tradable", "pos_limits_long", "pos_limits_short",
    "hedge_bid_px", "hedge_bid_sz", "hedge_ask_px", "hedge_ask_sz",
    "hedge_mid_px", "hedge_dv01", "hedge_ratios",
}


@dataclass
class SweepRunResult:
    """Everything produced by a single portfolio sweep run."""

    config: PortfolioSweepConfig
    raw: list[SweepResult] | list[SweepResultLight] | SweepSummary
    table: pd.DataFrame
    output_dir: Path | None

    @property
    def all_metrics(self) -> list[SweepResultLight] | None:
        if isinstance(self.raw, SweepSummary):
            return self.raw.all_metrics
        if self.raw and isinstance(self.raw[0], SweepResultLight):
            return self.raw
        return None

    @property
    def top_full(self) -> list[SweepResult] | None:
        if isinstance(self.raw, SweepSummary):
            return self.raw.top_full
        if self.raw and isinstance(self.raw[0], SweepResult):
            return self.raw
        return None


class PortfolioSweepRunner:
    @staticmethod
    def run_sweep_from_config(
        config: str | Path | PortfolioSweepConfig,
        *,
        market_data: dict[str, Any] | None = None,
        data_path: str | Path | None = None,
        instrument_ids: list[str] | None = None,
        hedge_ids: list[str] | None = None,
        output_dir: str | Path | None = None,
        save: bool = True,
        config_map_path: str | Path | None = None,
        **market_data_kwargs: Any,
    ) -> SweepRunResult:
        """End-to-end: load config, build scenarios, run sweep, save results.

        Parameters
        ----------
        config : str, Path, or PortfolioSweepConfig
            One of:
            - A ``PortfolioSweepConfig`` object (already loaded).
            - A path to a YAML config file (contains ``/`` or ends with ``.yaml``).
            - A config-map name (looked up via ``sweep_config_map.yaml``).
        market_data : dict, optional
            Pre-loaded market data arrays.  Keys must match ``run_sweep``
            signature.  If *None*, auto-loaded from the config ``data``
            section.
        data_path : str or Path, optional
            Directory containing parquet files.  Passed to
            ``PortfolioDataLoader.load()`` when the config has a ``data``
            section.
        instrument_ids : list[str], optional
            Ordered list of trading instrument IDs.  Passed to
            ``PortfolioDataLoader.load()`` when auto-loading market data.
        hedge_ids : list[str], optional
            Ordered list of hedge instrument IDs.  Passed to
            ``PortfolioDataLoader.load()`` when auto-loading market data.
        output_dir : str or Path, optional
            Directory to save results.  Defaults to ``runs/<grid_name>/``.
        save : bool
            Whether to persist results to disk (default True).
        config_map_path : str or Path, optional
            Path to the config map YAML (only used when *config* is a name).
        **market_data_kwargs
            Market data arrays passed as keyword arguments.  Merged with
            *market_data* dict (kwargs take precedence).

        Returns
        -------
        SweepRunResult
        """
        # --- 1. Load config -------------------------------------------------------
        cfg = _resolve_config(config, config_map_path)

        # --- 2. Merge market data -------------------------------------------------
        md = dict(market_data or {})
        md.update(market_data_kwargs)
        if not md and cfg.data_loader is not None:
            logger.info("Loading market data from config data section")
            loaded = cfg.data_loader.load(
                instrument_ids=instrument_ids,
                hedge_ids=hedge_ids,
                data_path=data_path,
            )
            md = loaded.to_dict()
        if not md:
            raise ValueError(
                "No market data provided.  Pass market_data=dict(...), "
                "individual arrays as keyword arguments, or add a 'data' "
                "section to the YAML config."
            )

        missing = _REQUIRED_KEYS - set(md)
        if missing:
            raise ValueError(f"Missing market data keys: {sorted(missing)}")

        # --- 3. Build scenarios ---------------------------------------------------
        scenarios = ScenarioBuilder.make_scenarios(
            cfg.base_config,
            cfg.grid,
            name_prefix=cfg.grid_name,
        )

        # --- 4. Run sweep ---------------------------------------------------------
        t0 = time.perf_counter()
        raw = PortfolioSweepExecutor.run_sweep(scenarios, **md, **cfg.sweep_kwargs)
        elapsed = time.perf_counter() - t0

        # --- 5. Build summary table -----------------------------------------------
        if isinstance(raw, SweepSummary):
            table = PortfolioSweepExecutor.summary_table(raw.all_metrics)
        else:
            table = PortfolioSweepExecutor.summary_table(raw)

        # --- 6. Persist results ---------------------------------------------------
        resolved_output_dir = None
        if save:
            resolved_output_dir = PortfolioSweepRunner.resolve_output_dir(
                output_dir, cfg,
            )
            PortfolioSweepPersister.persist(
                resolved_output_dir, cfg, raw, table, len(scenarios), elapsed,
            )

        return SweepRunResult(
            config=cfg,
            raw=raw,
            table=table,
            output_dir=resolved_output_dir,
        )

    @staticmethod
    def resolve_output_dir(
        output_dir: str | Path | None,
        cfg: PortfolioSweepConfig,
    ) -> Path:
        if output_dir is not None:
            d = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            d = Path("runs") / cfg.grid_name / timestamp
        d.mkdir(parents=True, exist_ok=True)
        return d
