#!/usr/bin/env python
"""Train multi-horizon prediction models with uncertainty quantification.

Usage:
    python scripts/train_multi_horizon.py \
        --data data/bonds.parquet \
        --value-col hedged_yield \
        --datetime-col datetime \
        --group-col bond_id \
        --features "feat_*" \
        --horizons "1,5,10,20" \
        --strategy per_horizon \
        --uncertainty quantile+conformal \
        --output outputs/multi_horizon_001

Strategies:
    - per_horizon: Train one model per horizon (Strategy A)
    - horizon_feature: Train single model with horizon as feature (Strategy B)

Uncertainty modes:
    - none: Point predictions only
    - quantile: Quantile regression (q10/q50/q90)
    - quantile+conformal: Quantile + conformal calibration for coverage guarantee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_horizons(s: str) -> list[int]:
    """Parse comma-separated horizons."""
    return [int(h.strip()) for h in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-horizon prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input data (parquet or csv)",
    )
    parser.add_argument(
        "--value-col",
        required=True,
        help="Column with values to predict (e.g., hedged_yield)",
    )
    parser.add_argument(
        "--datetime-col",
        default="datetime",
        help="Datetime column name (default: datetime)",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Group column for panel data (e.g., bond_id)",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Feature columns (comma-separated or glob pattern like 'feat_*')",
    )

    # Horizon arguments
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=[1, 5, 10, 20],
        help="Prediction horizons (comma-separated, default: 1,5,10,20)",
    )
    parser.add_argument(
        "--strategy",
        choices=["per_horizon", "horizon_feature"],
        default="per_horizon",
        help="Training strategy (default: per_horizon)",
    )

    # Uncertainty arguments
    parser.add_argument(
        "--uncertainty",
        choices=["none", "quantile", "quantile+conformal"],
        default="none",
        help="Uncertainty quantification mode (default: none)",
    )
    parser.add_argument(
        "--quantiles",
        type=lambda s: [float(q) for q in s.split(",")],
        default=[0.1, 0.5, 0.9],
        help="Quantiles for uncertainty (default: 0.1,0.5,0.9)",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.9,
        help="Target coverage for conformal calibration (default: 0.9)",
    )

    # Split arguments
    parser.add_argument(
        "--train-end",
        required=True,
        help="End date for training data (exclusive), e.g., 2023-01-01",
    )
    parser.add_argument(
        "--val-end",
        required=True,
        help="End date for validation data (exclusive)",
    )

    # Model arguments
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )

    # Preprocessing arguments
    parser.add_argument(
        "--scale",
        choices=["none", "standard", "robust"],
        default="standard",
        help="Scaling method (default: standard)",
    )
    parser.add_argument(
        "--impute",
        choices=["none", "mean", "median"],
        default="median",
        help="Imputation method (default: median)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    print(f"Loading data from {args.data}")
    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=[args.datetime_col])

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df[args.datetime_col].min()} to {df[args.datetime_col].max()}")

    if args.group_col:
        print(f"  Groups: {df[args.group_col].nunique()}")

    # Determine feature columns
    if "*" in args.features:
        import fnmatch

        pattern = args.features
        feature_cols = [c for c in df.columns if fnmatch.fnmatch(c, pattern)]
    else:
        feature_cols = [c.strip() for c in args.features.split(",")]

    print(f"  Features: {len(feature_cols)}")

    # Setup config
    from mlstudy.core.preprocess import PreprocessConfig, Preprocessor
    from mlstudy.train.multi_horizon import (
        MultiHorizonConfig,
        save_multi_horizon_result,
        train_multi_horizon,
    )

    config = MultiHorizonConfig(
        horizons=args.horizons,
        strategy=args.strategy,
        uncertainty=args.uncertainty,
        quantiles=args.quantiles,
        target_coverage=args.target_coverage,
        model_type="lgbm",
        model_params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
    )

    prep_config = PreprocessConfig(
        impute=args.impute,
        scale=args.scale,
    )
    preprocessor = Preprocessor(prep_config)

    print("\nTraining configuration:")
    print(f"  Horizons: {config.horizons}")
    print(f"  Strategy: {config.strategy}")
    print(f"  Uncertainty: {config.uncertainty}")
    print(f"  Train end: {args.train_end}")
    print(f"  Val end: {args.val_end}")

    # Train
    print("\nTraining models...")
    result = train_multi_horizon(
        df=df,
        feature_cols=feature_cols,
        value_col=args.value_col,
        datetime_col=args.datetime_col,
        group_col=args.group_col,
        train_end=args.train_end,
        val_end=args.val_end,
        config=config,
        preprocessor=preprocessor,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results per horizon:")
    print("=" * 60)

    for horizon, hr in result.horizon_results.items():
        print(f"\nHorizon {horizon}:")
        print(f"  RMSE: {hr.metrics['rmse']:.6f}")
        print(f"  MAE:  {hr.metrics['mae']:.6f}")
        print(f"  R2:   {hr.metrics['r2']:.4f}")

        if hr.coverage_stats:
            print(f"  Coverage (pre-cal):  {hr.coverage_stats['coverage_pre_calibration']:.3f}")
            if "coverage_post_calibration" in hr.coverage_stats:
                print(f"  Coverage (post-cal): {hr.coverage_stats['coverage_post_calibration']:.3f}")
            print(f"  Mean interval width: {hr.coverage_stats['mean_width']:.6f}")
            if "calibrated_mean_width" in hr.coverage_stats:
                print(f"  Calibrated width:    {hr.coverage_stats['calibrated_mean_width']:.6f}")

    # Save artifacts
    print(f"\nSaving artifacts to {args.output}")
    artifacts = save_multi_horizon_result(result, args.output)

    # Save feature schema
    feature_schema = {
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "value_col": args.value_col,
        "datetime_col": args.datetime_col,
        "group_col": args.group_col,
    }
    schema_path = args.output / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(feature_schema, f, indent=2)

    print("\nArtifacts saved:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
