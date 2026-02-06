#!/usr/bin/env python
"""Simulate real-time prediction using exported artifacts.

Loads an artifact bundle and simulates streaming predictions
on synthetic market data.

Usage:
    python scripts/realtime_simulate.py \
        --artifact artifacts/run_001 \
        --n-samples 1000 \
        --interval 0.1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def simulate_market_tick(
    n_features: int,
    prev_features: np.ndarray | None = None,
    volatility: float = 0.02,
) -> np.ndarray:
    """Simulate a single market data tick.

    Args:
        n_features: Number of features.
        prev_features: Previous feature values for autocorrelation.
        volatility: Feature volatility.

    Returns:
        Feature array of shape (n_features,).
    """
    if prev_features is None:
        # Initialize with standard normal
        return np.random.randn(n_features)

    # Random walk with mean reversion
    noise = np.random.randn(n_features) * volatility
    mean_reversion = -0.1 * prev_features
    return prev_features + noise + mean_reversion


def main():
    parser = argparse.ArgumentParser(
        description="Simulate real-time prediction with exported artifacts"
    )
    parser.add_argument(
        "--artifact",
        "-a",
        type=Path,
        required=True,
        help="Path to artifact directory",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples to simulate (default: 100)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=0.0,
        help="Interval between predictions in seconds (default: 0, no delay)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for predictions (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each prediction",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load artifact
    from mlstudy.deploy.serve import ArtifactPredictor

    print(f"Loading artifact from {args.artifact}")
    predictor = ArtifactPredictor.load(args.artifact)

    print(f"Model type: {predictor.model_type}")
    print(f"Task: {predictor.task}")
    print(f"Features: {predictor.n_features}")

    n_features = predictor.n_features
    if n_features == 0:
        # Default if schema not available
        n_features = 10
        print(f"Warning: No feature schema, using {n_features} features")

    # Simulate predictions
    print(f"\nSimulating {args.n_samples} predictions...")
    print("-" * 60)

    predictions = []
    probas = []
    latencies = []
    prev_features = None

    for i in range(0, args.n_samples, args.batch_size):
        batch_size = min(args.batch_size, args.n_samples - i)

        # Generate features
        batch_features = []
        for _ in range(batch_size):
            features = simulate_market_tick(n_features, prev_features)
            batch_features.append(features)
            prev_features = features

        X = np.array(batch_features)

        # Time prediction
        start = time.perf_counter()
        pred = predictor.predict(X)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms

        predictions.extend(pred.tolist())

        if predictor.task == "classification":
            proba = predictor.predict_proba(X)
            if proba is not None:
                probas.extend(proba[:, 1].tolist())

        if args.verbose:
            for j, p in enumerate(pred):
                sample_idx = i + j
                if predictor.task == "classification" and probas:
                    print(f"[{sample_idx:4d}] pred={p}, proba={probas[-batch_size + j]:.4f}, latency={elapsed*1000:.2f}ms")
                else:
                    print(f"[{sample_idx:4d}] pred={p:.6f}, latency={elapsed*1000:.2f}ms")

        if args.interval > 0:
            time.sleep(args.interval)

    # Summary statistics
    print("-" * 60)
    print("\nSummary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Mean latency: {np.mean(latencies):.2f} ms")
    print(f"  P50 latency: {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")

    predictions = np.array(predictions)
    if predictor.task == "regression":
        print(f"\n  Prediction mean: {predictions.mean():.6f}")
        print(f"  Prediction std: {predictions.std():.6f}")
        print(f"  Prediction min: {predictions.min():.6f}")
        print(f"  Prediction max: {predictions.max():.6f}")
    else:
        unique, counts = np.unique(predictions, return_counts=True)
        print("\n  Class distribution:")
        for u, c in zip(unique, counts):  # noqa: B905
            print(f"    {u}: {c} ({c/len(predictions)*100:.1f}%)")

    # Throughput
    total_time = sum(latencies) / 1000  # seconds
    throughput = len(predictions) / total_time if total_time > 0 else 0
    print(f"\n  Throughput: {throughput:.0f} predictions/second")


if __name__ == "__main__":
    main()
