from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mlstudy.trading.backtest.mean_reversion.sweep.sweep_types import SweepResult

ARRAY_FIELDS = [
    "positions",
    "cash",
    "equity",
    "pnl",
    "codes",
    "state",
    "holding",
    "tr_bar",
    "tr_type",
    "tr_side",
    "tr_sizes",
    "tr_risks",
    "tr_vwaps",
    "tr_mids",
    "tr_cost",
    "tr_code",
    "tr_pkg_yield",
]


def _save_top_full(results: list[SweepResult], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, sr in enumerate(results):
        scenario_dir = output_dir / f"scenario_{rank:03d}"
        scenario_dir.mkdir(exist_ok=True)

        spec = {
            "name": sr.scenario.name,
            "tags": sr.scenario.tags,
            "config": asdict(sr.scenario.cfg),
            "metrics": asdict(sr.metrics),
            "scenario_idx": sr.scenario_idx,
        }
        with open(scenario_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2, default=str)

        for field_name in ARRAY_FIELDS:
            arr = getattr(sr.results, field_name)
            np.save(scenario_dir / f"{field_name}.npy", arr)
