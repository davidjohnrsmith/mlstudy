#!/usr/bin/env python
"""Train models on prepared dataset.

Usage:
    python scripts/train_on_prepared.py --task regression --model ridge
    python scripts/train_on_prepared.py --task classification --split walk_forward
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
        description="Train models on prepared dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/simulated/prepared",
        help="Directory with prepared X, y, meta files.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Task type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Model name (ridge, rf, hgb for regression; logistic, rf for classification).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["time", "walk_forward"],
        default="time",
        help="Split strategy.",
    )
    # Time split params
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
        help="Fraction of data for training (time split).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data for validation (time split).",
    )
    # Walk-forward params
    parser.add_argument(
        "--train-days",
        type=int,
        default=30,
        help="Training window in days (walk-forward).",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=7,
        help="Validation window in days (walk-forward).",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=7,
        help="Test window in days (walk-forward).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=7,
        help="Step between folds (walk-forward).",
    )
    # Preprocess
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
        "--outdir",
        type=str,
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args(argv)


def load_prepared_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prepared X, y, meta from directory."""
    data_path = Path(data_dir)

    # Try parquet first, then CSV
    if (data_path / "X.parquet").exists():
        X = pd.read_parquet(data_path / "X.parquet")
        y = pd.read_parquet(data_path / "y.parquet")
        meta = pd.read_parquet(data_path / "meta.parquet")
    elif (data_path / "X.csv").exists():
        X = pd.read_csv(data_path / "X.csv")
        y = pd.read_csv(data_path / "y.csv")
        meta = pd.read_csv(data_path / "meta.csv")
    else:
        raise FileNotFoundError(f"No prepared data found in {data_dir}")

    return X, y, meta


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Load prepared data
    print(f"Loading data from {args.data_dir}...")
    try:
        X, y, meta = load_prepared_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run 'python scripts/make_supervised_from_simulated.py' first.")
        return 1

    print(f"Loaded X: {X.shape}, y: {y.shape}")

    # Combine into single DataFrame for the experiment runner
    target_col = y.columns[0]
    df = pd.concat([meta, X, y], axis=1)

    # Parse datetime if needed
    datetime_col = "datetime"
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Determine group column
    group_col = "asset" if "asset" in df.columns else None

    # Determine date splits based on fractions
    df_sorted = df.sort_values(datetime_col)
    n = len(df_sorted)
    dates = df_sorted[datetime_col]

    if args.split == "time":
        train_idx = int(n * args.train_frac)
        val_idx = int(n * (args.train_frac + args.val_frac))

        train_end = str(dates.iloc[train_idx - 1].date())
        val_end = str(dates.iloc[val_idx - 1].date())
        test_end = str(dates.iloc[-1].date())

        print("\nTime split:")
        print(f"  Train end: {train_end}")
        print(f"  Val end: {val_end}")
        print(f"  Test end: {test_end}")

    # Build config
    preprocess_config = PreprocessConfig(
        impute=args.impute,
        scale=args.scale,
    )

    feature_cols = list(X.columns)

    config = ExperimentConfig(
        task=args.task,
        model_name=args.model,
        split_strategy=args.split,
        datetime_col=datetime_col,
        target_col=target_col,
        feature_cols=feature_cols,
        group_col=group_col,
        preprocess=preprocess_config,
        random_state=args.seed,
        train_end=train_end if args.split == "time" else None,
        val_end=val_end if args.split == "time" else None,
        test_end=test_end if args.split == "time" else None,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        expanding=True,
    )

    # Run experiment
    print(f"\nTraining {args.model} for {args.task}...")
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
