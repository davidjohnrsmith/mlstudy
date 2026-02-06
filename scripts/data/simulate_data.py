#!/usr/bin/env python
"""Generate simulated market data and save to disk.

Usage:
    python scripts/simulate_data.py --n-assets 5 --n-periods 2000 --seed 42
    python scripts/simulate_data.py --output data/simulated/custom.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from mlstudy.research.simulate.market import simulate_market_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate simulated market data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n-assets",
        type=int,
        default=5,
        help="Number of assets to simulate.",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=2000,
        help="Number of time periods per asset.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="1h",
        help="Frequency (e.g., '1h', '1D').",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.02,
        help="Base volatility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. If not specified, auto-generates timestamped filename.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/simulated/market_data_{timestamp}.csv")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Simulating market data...")
    print(f"  Assets: {args.n_assets}")
    print(f"  Periods: {args.n_periods}")
    print(f"  Frequency: {args.freq}")
    print(f"  Seed: {args.seed}")

    df = simulate_market_data(
        n_assets=args.n_assets,
        n_periods=args.n_periods,
        freq=args.freq,
        start_date=args.start_date,
        seed=args.seed,
        volatility=args.volatility,
    )

    print(f"\nGenerated {len(df)} rows ({args.n_assets} assets x {args.n_periods} periods)")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Also save a "latest" symlink/copy for easy access
    latest_path = output_path.parent / "latest.csv"
    df.to_csv(latest_path, index=False)
    print(f"Also saved as: {latest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
