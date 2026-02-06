#!/usr/bin/env python
"""Build supervised dataset from simulated market data.

Usage:
    python scripts/make_supervised_from_simulated.py
    python scripts/make_supervised_from_simulated.py --input data/simulated/latest.csv
    python scripts/make_supervised_from_simulated.py --horizon 5 --windows 5,10,20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlstudy.core.features import FeatureSpec
from mlstudy.ml.pipeline.make_dataset import TargetSpec, make_supervised_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build supervised dataset from simulated data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/simulated/latest.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/simulated/prepared",
        help="Output directory for prepared dataset.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forward return horizon (steps).",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="5,10,20",
        help="Comma-separated rolling window sizes.",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="close",
        help="Price column name.",
    )
    parser.add_argument(
        "--volume-col",
        type=str,
        default="volume",
        help="Volume column name.",
    )
    parser.add_argument(
        "--datetime-col",
        type=str,
        default="datetime",
        help="Datetime column name.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="asset",
        help="Group column name.",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        choices=["forward_return", "forward_direction"],
        default="forward_return",
        help="Target type.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        print("Run 'python scripts/simulate_data.py' first to generate data.")
        return 1

    # Parse windows
    windows = [int(w.strip()) for w in args.windows.split(",")]

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=[args.datetime_col])
    print(f"Loaded {len(df)} rows")

    # Build feature specs
    feature_specs = []

    # Price features
    feature_specs.append(
        FeatureSpec(name="returns", params={"price_col": args.price_col, "periods": 1})
    )

    for w in windows:
        feature_specs.append(
            FeatureSpec(name="momentum", params={"price_col": args.price_col, "window": w})
        )
        feature_specs.append(
            FeatureSpec(
                name="rolling_volatility", params={"price_col": args.price_col, "window": w}
            )
        )

    # Volume features
    if args.volume_col in df.columns:
        for w in windows:
            feature_specs.append(
                FeatureSpec(
                    name="volume_shock", params={"volume_col": args.volume_col, "window": w}
                )
            )

    # Flow features - add EWMA of flow_imbalance
    # Since we may not have this in registry, we'll add lagged flow values manually later
    # For now, let's add what we have in the registry

    print(f"\nBuilding {len(feature_specs)} features...")

    # Target spec
    target_spec = TargetSpec(
        target_type=args.target_type,
        price_col=args.price_col,
        horizon_steps=args.horizon,
        log_return=True,
    )

    # Build dataset
    X, y, meta_df, dataset_meta = make_supervised_dataset(
        df,
        feature_specs=feature_specs,
        target_spec=target_spec,
        datetime_col=args.datetime_col,
        group_col=args.group_col,
        dropna=True,
    )

    print("\nDataset summary:")
    print(f"  Samples: {dataset_meta.n_samples}")
    print(f"  Features: {dataset_meta.n_features}")
    print(f"  Target: {dataset_meta.target_col}")
    print(f"  Rows dropped (NaN): {dataset_meta.null_dropped}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet if available, otherwise CSV
    try:
        X.to_parquet(output_dir / "X.parquet", index=False)
        y.to_parquet(output_dir / "y.parquet", index=False)
        meta_df.to_parquet(output_dir / "meta.parquet", index=False)
        format_used = "parquet"
    except ImportError:
        X.to_csv(output_dir / "X.csv", index=False)
        y.to_csv(output_dir / "y.csv", index=False)
        meta_df.to_csv(output_dir / "meta.csv", index=False)
        format_used = "csv"

    # Save feature info
    info_df = pd.DataFrame(
        {
            "feature": dataset_meta.feature_cols,
            "index": range(len(dataset_meta.feature_cols)),
        }
    )
    info_df.to_csv(output_dir / "feature_info.csv", index=False)

    print(f"\nSaved to {output_dir}/ ({format_used} format):")
    print(f"  X.{format_used}: {X.shape}")
    print(f"  y.{format_used}: {y.shape}")
    print(f"  meta.{format_used}: {meta_df.shape}")
    print("  feature_info.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
