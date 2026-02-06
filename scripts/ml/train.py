#!/usr/bin/env python
"""CLI for training ML models.

Usage examples:

    # Time split with regression
    python scripts/train.py \\
        --data-csv data/features.csv \\
        --datetime-col date \\
        --target-col forward_return \\
        --feature-cols "momentum,volatility,volume" \\
        --task regression \\
        --model ridge \\
        --split time \\
        --train-end 2023-06-30 \\
        --val-end 2023-09-30 \\
        --test-end 2023-12-31

    # Walk-forward with classification
    python scripts/train.py \\
        --data-csv data/features.csv \\
        --datetime-col date \\
        --target-col signal \\
        --task classification \\
        --model rf \\
        --split walk_forward \\
        --train-days 252 \\
        --val-days 63 \\
        --test-days 63 \\
        --step-days 21
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mlstudy.core.preprocess import PreprocessConfig
from mlstudy.ml.train.run_experiment import ExperimentConfig, run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML models for trading research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--datetime-col",
        type=str,
        default="datetime",
        help="Name of datetime column.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Name of grouping column (e.g., symbol).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Name of target column.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated feature columns. If not provided, infers all numeric columns.",
    )

    # Task and model
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        required=True,
        help="Task type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Model name (linear, ridge, rf, hgb, logistic).",
    )

    # Split strategy
    parser.add_argument(
        "--split",
        type=str,
        choices=["time", "walk_forward"],
        default="time",
        help="Split strategy.",
    )

    # Time split params
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Training end date (for time split).",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="Validation end date (for time split).",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Test end date (for time split).",
    )

    # Walk-forward params
    parser.add_argument(
        "--train-days",
        type=int,
        default=252,
        help="Training window days (for walk-forward).",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=63,
        help="Validation window days (for walk-forward).",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=63,
        help="Test window days (for walk-forward).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=21,
        help="Step days between folds (for walk-forward).",
    )
    parser.add_argument(
        "--expanding",
        action="store_true",
        default=True,
        help="Use expanding window (default). Use --no-expanding for rolling.",
    )
    parser.add_argument(
        "--no-expanding",
        dest="expanding",
        action="store_false",
        help="Use rolling window instead of expanding.",
    )

    # Preprocessing params
    parser.add_argument(
        "--impute",
        type=str,
        choices=["none", "median", "mean"],
        default="median",
        help="Imputation strategy.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["none", "standard", "robust"],
        default="standard",
        help="Scaling strategy.",
    )
    parser.add_argument(
        "--winsorize",
        type=float,
        default=None,
        help="Winsorization percentile (e.g., 0.01 for 1st/99th percentile).",
    )

    # Other params
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="Minimum samples per group.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. If not provided, auto-generated.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Read data
    csv_path = Path(args.data_csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Parse datetime column
    if args.datetime_col in df.columns:
        df[args.datetime_col] = pd.to_datetime(df[args.datetime_col])

    # Infer feature columns if not provided
    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",")]
    else:
        # Use all numeric columns except datetime, group, and target
        exclude = {args.datetime_col, args.target_col}
        if args.group_col:
            exclude.add(args.group_col)
        feature_cols = [
            c for c in df.select_dtypes(include=["number"]).columns if c not in exclude
        ]
        print(f"Inferred feature columns: {feature_cols}")

    # Validate time split params
    if args.split == "time" and (not args.train_end or not args.val_end):
        print(
            "Error: --train-end and --val-end required for time split",
            file=sys.stderr,
        )
        return 1

    # Build config
    preprocess_config = PreprocessConfig(
        impute=args.impute,
        scale=args.scale,
        winsorize=args.winsorize,
    )

    config = ExperimentConfig(
        task=args.task,
        model_name=args.model,
        split_strategy=args.split,
        datetime_col=args.datetime_col,
        target_col=args.target_col,
        feature_cols=feature_cols,
        group_col=args.group_col,
        preprocess=preprocess_config,
        random_state=args.random_state,
        train_end=args.train_end,
        val_end=args.val_end,
        test_end=args.test_end,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        expanding=args.expanding,
        min_count=args.min_count,
    )

    # Run experiment
    print(f"Running {args.task} experiment with {args.model} model...")
    print(f"Split strategy: {args.split}")

    result = run_experiment(df, config, output_dir=args.outdir)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print("\nTrain metrics:")
    for k, v in result.train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nValidation metrics:")
    for k, v in result.val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in result.test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if result.fold_metrics:
        print(f"\nWalk-forward: {len(result.fold_metrics)} folds")

    print(f"\nOutputs saved to: {result.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
