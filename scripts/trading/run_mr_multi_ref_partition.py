#!/usr/bin/env python
"""Launch script for a single partition of a distributed multi-ref sweep.

Run one partition on a remote machine::

    python scripts/trading/run_mr_multi_ref_partition.py \
        --config configs/mr_grid_v1.yaml \
        --data-path /mnt/data/20240101 \
        --ref-instruments UST_2Y,UST_5Y,UST_10Y,UST_20Y,UST_30Y \
        --num-partitions 3 \
        --partition-index 0 \
        --outdir /shared/runs/20240101

Then, after all partitions finish, collect results on any machine::

    python scripts/trading/run_mr_multi_ref_partition.py \
        --config configs/mr_grid_v1.yaml \
        --ref-instruments UST_2Y,UST_5Y,UST_10Y,UST_20Y,UST_30Y \
        --num-partitions 3 \
        --collect \
        --outdir /shared/runs/20240101

The full ref list (``--ref-instruments``) must be identical and in the same
order on every machine so that ``partition_instruments`` produces
deterministic, non-overlapping subsets.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from mlstudy.trading.backtest.mean_reversion.sweep.multi_ref_runner import (
    MultiRefSweepRunner,
    partition_instruments,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single partition of a distributed multi-ref sweep, "
        "or collect all partitions into combined analytics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to sweep YAML config (or config-map name).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Directory with parquet files (not needed for --collect).",
    )
    parser.add_argument(
        "--ref-instruments",
        type=str,
        required=False,
        help="Comma-separated reference instrument IDs, e.g. UST_2Y,UST_5Y,UST_10Y",
        default="UST_2Y,UST_5Y",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated instrument IDs (auto-detected per ref if omitted).",
    )

    part = parser.add_argument_group("partitioning")
    part.add_argument(
        "--num-partitions",
        type=int,
        required=True,
        help="Total number of partitions (machines).",
    )
    part.add_argument(
        "--partition-index",
        type=int,
        default=None,
        help="Zero-based index of this partition. Required unless --collect.",
    )

    part.add_argument(
        "--collect",
        action="store_true",
        help="Collect all partitions and produce combined analytics "
        "(no sweep is run).",
    )

    ranking = parser.add_argument_group("ranking")
    ranking.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of rows in per-ref-best and param leaderboard.",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Partitions land in "
        "<outdir>/<grid_name>/partition_NNN/.",
    )
    out.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist results to disk (ignored for --collect).",
    )
    out.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return parser.parse_args(argv)


def _print(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def _run_partition(args: argparse.Namespace) -> int:
    quiet = args.quiet
    ref_instruments = [s.strip() for s in args.ref_instruments.split(",") if s.strip()]

    if args.partition_index is None:
        print("ERROR: --partition-index is required when not using --collect.")
        return 1

    if not ref_instruments:
        print("ERROR: --ref-instruments must contain at least one instrument ID.")
        return 1

    instruments = None
    if args.instruments:
        instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    subset = partition_instruments(ref_instruments, args.num_partitions, args.partition_index)
    _print(f"Partition {args.partition_index}/{args.num_partitions}  "
           f"({len(subset)} of {len(ref_instruments)} refs): {subset}", quiet)

    if not subset:
        _print("This partition has no instruments — nothing to do.", quiet)
        return 0

    t0 = time.perf_counter()
    result = MultiRefSweepRunner.run_partition(
        config=args.config,
        ref_instrument_ids=ref_instruments,
        num_partitions=args.num_partitions,
        partition_index=args.partition_index,
        data_path=args.data_path,
        instrument_ids=instruments,
        output_dir=args.outdir,
        save=not args.no_save,
        top_n=args.top_n,
    )
    elapsed = time.perf_counter() - t0

    _print(f"\nPartition {args.partition_index} complete", quiet)
    _print(f"  Refs      : {len(subset)}", quiet)
    _print(f"  Scenarios : {len(result.cross_ref_summary)}", quiet)
    _print(f"  Elapsed   : {elapsed:.1f}s", quiet)
    if result.output_dir:
        _print(f"  Output    : {result.output_dir}", quiet)

    return 0


def _collect(args: argparse.Namespace) -> int:
    quiet = args.quiet
    ref_instruments = [s.strip() for s in args.ref_instruments.split(",") if s.strip()]

    if args.outdir is None:
        print("ERROR: --outdir is required for --collect.")
        return 1

    from mlstudy.trading.backtest.mean_reversion.configs.utils import _resolve_config

    cfg = _resolve_config(args.config, None)
    base_dir = __import__("pathlib").Path(args.outdir) / cfg.grid_name

    _print(f"Collecting {args.num_partitions} partitions from {base_dir} ...", quiet)

    t0 = time.perf_counter()
    result = MultiRefSweepRunner.collect_partitions(
        base_output_dir=base_dir,
        config=args.config,
        num_partitions=args.num_partitions,
        ref_instrument_ids=ref_instruments or None,
        top_n=args.top_n,
    )
    elapsed = time.perf_counter() - t0

    _print(f"\nCollect complete", quiet)
    _print(f"  Total refs : {len(result.cross_ref_summary['ref_instrument_id'].unique()) if not result.cross_ref_summary.empty else 0}", quiet)
    _print(f"  Scenarios  : {len(result.cross_ref_summary)}", quiet)
    _print(f"  Elapsed    : {elapsed:.1f}s", quiet)
    _print(f"  Output     : {result.output_dir}", quiet)

    if not result.per_ref_best.empty:
        print(f"\n{'=' * 70}")
        print("BEST SCENARIO PER REF (ranked by RankingPlan)")
        print(f"{'=' * 70}")
        display_cols = [c for c in [
            "rank", "ref_instrument_id", "name",
            "total_pnl", "sharpe_ratio", "max_drawdown", "n_trades",
        ] if c in result.per_ref_best.columns]
        display_cols = list(dict.fromkeys(display_cols))
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(result.per_ref_best[display_cols].to_string(index=False))

    if not result.param_leaderboard.empty:
        print(f"\n{'=' * 70}")
        print(f"PARAM LEADERBOARD (top {args.top_n})")
        print(f"{'=' * 70}")
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(result.param_leaderboard.to_string(index=False))

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.collect:
        return _collect(args)
    return _run_partition(args)


if __name__ == "__main__":
    sys.exit(main())
