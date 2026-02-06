#!/usr/bin/env python
"""Export trained model to deployable artifact bundle.

Usage:
    python scripts/export_artifact.py \
        --model outputs/experiment_001/model.pkl \
        --preprocessor outputs/experiment_001/preprocessor.pkl \
        --features outputs/experiment_001/features.json \
        --output artifacts/run_001 \
        --task regression

Or from experiment directory:
    python scripts/export_artifact.py \
        --experiment outputs/experiment_001 \
        --output artifacts/run_001
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Export model to deployable artifact bundle"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to pickled model file",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        help="Path to pickled preprocessor file (optional)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Path to feature names JSON file",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        help="Experiment directory (auto-detects model/preprocessor/features)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output artifact directory",
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Task type (default: regression)",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "xgboost", "lightgbm"],
        help="Override auto-detected model type",
    )

    args = parser.parse_args()

    # Load from experiment directory if specified
    if args.experiment:
        exp_dir = args.experiment
        model_path = args.model or exp_dir / "model.pkl"
        preprocessor_path = args.preprocessor or exp_dir / "preprocessor.pkl"
        features_path = args.features or exp_dir / "features.json"

        # Check for metadata.json to get task
        metadata_path = exp_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            args.task = metadata.get("task", args.task)
    else:
        model_path = args.model
        preprocessor_path = args.preprocessor
        features_path = args.features

    if not model_path or not model_path.exists():
        parser.error("Model file not found. Specify --model or --experiment")

    # Load model
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load preprocessor if available
    preprocessor = None
    if preprocessor_path and preprocessor_path.exists():
        print(f"Loading preprocessor from {preprocessor_path}")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

    # Load feature names
    feature_names = None
    feature_dtypes = None
    if features_path and features_path.exists():
        print(f"Loading features from {features_path}")
        with open(features_path) as f:
            features_data = json.load(f)
        if isinstance(features_data, list):
            feature_names = features_data
        else:
            feature_names = features_data.get("feature_names", [])
            feature_dtypes = features_data.get("feature_dtypes", {})

    # Export artifact
    from mlstudy.deploy.export import export_artifact

    print(f"Exporting artifact to {args.output}")
    artifacts = export_artifact(
        model=model,
        path=args.output,
        preprocessor=preprocessor,
        feature_names=feature_names,
        feature_dtypes=feature_dtypes,
        task=args.task,
        model_type=args.model_type,
    )

    print("\nExported artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    # Verify by loading
    from mlstudy.deploy.serve import ArtifactPredictor

    print("\nVerifying artifact...")
    predictor = ArtifactPredictor.load(args.output)
    print(f"  Model type: {predictor.model_type}")
    print(f"  Task: {predictor.task}")
    print(f"  Features: {predictor.n_features}")

    # Quick test prediction
    if predictor.n_features > 0:
        test_X = np.zeros((1, predictor.n_features))
        pred = predictor.predict(test_X)
        print(f"  Test prediction shape: {pred.shape}")

    print("\nArtifact export complete!")


if __name__ == "__main__":
    main()
