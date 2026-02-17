"""End-to-end sweep interface: load config, build scenarios, run sweep, persist results.

Usage from a script or notebook::

    from mlstudy.trading.backtest.mean_reversion.sweep_runner import run_sweep_from_config

    # Option A: by YAML path
    summary = run_sweep_from_config("configs/mr_grid_v1.yaml")

    # Option B: by config-map name
    summary = run_sweep_from_config("mr_grid_v1")

    # Option C: with market data already loaded
    summary = run_sweep_from_config("mr_grid_v1", market_data=my_market_data)

The function returns a ``SweepRunResult`` that bundles:
- the parsed config
- all raw results (list or SweepSummary)
- the summary table as a DataFrame
- the output directory where artifacts are saved
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mlstudy.trading.backtest.mean_reversion.configs.utils import _resolve_config
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_results_saver import _persist

logger = logging.getLogger(__name__)

import pandas as pd

from mlstudy.trading.backtest.mean_reversion.sweep import run_sweep, summary_table
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_build import make_scenarios
from mlstudy.trading.backtest.mean_reversion.configs.sweep_config import SweepConfig
from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import MetricsOnlyResult, SweepResult, SweepSummary


@dataclass
class SweepRunResult:
    """Everything produced by a single sweep run."""

    config: SweepConfig
    raw: list[SweepResult] | list[MetricsOnlyResult] | SweepSummary
    table: pd.DataFrame
    output_dir: Path | None

    @property
    def all_metrics(self) -> list[MetricsOnlyResult] | None:
        if isinstance(self.raw, SweepSummary):
            return self.raw.all_metrics
        if self.raw and isinstance(self.raw[0], MetricsOnlyResult):
            return self.raw
        return None

    @property
    def top_full(self) -> list[SweepResult] | None:
        if isinstance(self.raw, SweepSummary):
            return self.raw.top_full
        if self.raw and isinstance(self.raw[0], SweepResult):
            return self.raw
        return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sweep_from_config(
    config: str | Path | SweepConfig,
    *,
    market_data: dict[str, Any] | None = None,
    data_path: str | Path | None = None,
    instrument_ids: list[str] | None = None,
    ref_instrument_id: str | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
    config_map_path: str | Path | None = None,
    **market_data_kwargs: Any,
) -> SweepRunResult:
    """End-to-end: load config, build scenarios, run sweep, save results.

    Parameters
    ----------
    config : str, Path, or SweepConfig
        One of:
        - A ``SweepConfig`` object (already loaded).
        - A path to a YAML config file (contains ``/`` or ends with ``.yaml``).
        - A config-map name (looked up via ``sweep_config_map.yaml``).
    market_data : dict, optional
        Pre-loaded market data arrays.  Keys must match ``run_sweep``
        signature: ``bid_px``, ``bid_sz``, ``ask_px``, ``ask_sz``,
        ``mid_px``, ``dv01``, ``zscore``, ``expected_yield_pnl_bps``,
        ``package_yield_bps``, ``hedge_ratios``.
        If *None*, you must pass them as ``**market_data_kwargs``.
    data_path : str or Path, optional
        Directory containing parquet files.  Passed to
        ``BacktestDataLoader.load()`` when the config has a ``data``
        section.  Use this to keep the YAML config platform-independent
        and supply the data directory at runtime.
    instrument_ids : list[str], optional
        Ordered list of instrument IDs.  Passed to
        ``BacktestDataLoader.load()`` when auto-loading market data.
    ref_instrument_id : str, optional
        Reference instrument ID for signal filtering.  Passed to
        ``BacktestDataLoader.load()`` when auto-loading market data.
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
    import time

    # --- 1. Load config -------------------------------------------------------
    cfg = _resolve_config(config, config_map_path)

    # --- 2. Merge market data -------------------------------------------------
    md = dict(market_data or {})
    md.update(market_data_kwargs)
    if not md and cfg.data_loader is not None:
        if instrument_ids is None or ref_instrument_id is None:
            raise ValueError(
                "instrument_ids and ref_instrument_id must be provided "
                "when auto-loading market data from the config data section."
            )
        logger.info("Loading market data from config data section")
        loaded = cfg.data_loader.load(
            instrument_ids=instrument_ids,
            ref_instrument_id=ref_instrument_id,
            data_path=data_path,
        )
        md = loaded.to_dict()
    if not md:
        raise ValueError(
            "No market data provided.  Pass market_data=dict(...), "
            "individual arrays as keyword arguments, or add a 'data' "
            "section to the YAML config."
        )

    _REQUIRED_KEYS = {
        "bid_px", "bid_sz", "ask_px", "ask_sz", "mid_px",
        "dv01", "zscore", "expected_yield_pnl_bps",
        "package_yield_bps", "hedge_ratios",
    }
    missing = _REQUIRED_KEYS - set(md)
    if missing:
        raise ValueError(f"Missing market data keys: {sorted(missing)}")

    # --- 3. Build scenarios ---------------------------------------------------
    scenarios = make_scenarios(
        cfg.base_config,
        cfg.grid,
        name_prefix=cfg.grid_name,
    )

    # --- 4. Run sweep ---------------------------------------------------------
    t0 = time.perf_counter()
    raw = run_sweep(scenarios, **md, **cfg.sweep_kwargs)
    elapsed = time.perf_counter() - t0

    # --- 5. Build summary table -----------------------------------------------
    if isinstance(raw, SweepSummary):
        table = summary_table(raw.all_metrics)
    else:
        table = summary_table(raw)

    # --- 6. Persist results ---------------------------------------------------
    resolved_output_dir = None
    if save:
        resolved_output_dir = _resolve_output_dir(output_dir, cfg)
        _persist(
            resolved_output_dir, cfg, raw, table, len(scenarios), elapsed,
            zscore=md.get("zscore"),
        )

    return SweepRunResult(
        config=cfg,
        raw=raw,
        table=table,
        output_dir=resolved_output_dir,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _resolve_output_dir(
    output_dir: str | Path | None,
    cfg: SweepConfig,
) -> Path:
    if output_dir is not None:
        d = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("runs") / cfg.grid_name / timestamp
    d.mkdir(parents=True, exist_ok=True)
    return d


