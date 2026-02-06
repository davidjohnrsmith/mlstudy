"""Distribution comparison metrics.

This module provides functions to compute statistical distances and tests
between two samples:
- Kolmogorov-Smirnov test
- Wasserstein (Earth Mover's) distance
- Jensen-Shannon divergence (histogram-based)
- Effect size metrics (median diff, IQR diff)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


class KSResult(NamedTuple):
    """Result of Kolmogorov-Smirnov test."""

    statistic: float
    pvalue: float


class EffectSizeResult(NamedTuple):
    """Effect size metrics between two distributions."""

    median_diff: float
    iqr_diff: float


def ks_test(a: ArrayLike, b: ArrayLike) -> KSResult:
    """Perform two-sample Kolmogorov-Smirnov test.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        KSResult with statistic and p-value.
    """
    result = stats.ks_2samp(a, b)
    return KSResult(statistic=float(result.statistic), pvalue=float(result.pvalue))


def wasserstein(a: ArrayLike, b: ArrayLike) -> float:
    """Compute Wasserstein (Earth Mover's) distance between two samples.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Wasserstein distance (non-negative float).
    """
    return float(stats.wasserstein_distance(a, b))


def jensen_shannon_divergence(
    a: ArrayLike,
    b: ArrayLike,
    bins: int | str = "auto",
) -> float:
    """Compute Jensen-Shannon divergence using histogram estimates.

    The JS divergence is a symmetric measure of similarity between two
    probability distributions, bounded between 0 (identical) and 1 (maximally
    different when using log base 2).

    Binning approach:
    - Uses numpy.histogram_bin_edges with the specified method on the combined
      data to ensure both distributions use identical bin edges.
    - Default "auto" uses numpy's automatic bin selection.
    - Adds a small epsilon to avoid log(0) in KL divergence calculation.

    Args:
        a: First sample.
        b: Second sample.
        bins: Number of bins or binning strategy (default "auto").
              See numpy.histogram_bin_edges for options.

    Returns:
        JS divergence in [0, 1] (using log base 2).
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Compute common bin edges from combined data
    combined = np.concatenate([a, b])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    # Compute histograms (normalized to sum to 1)
    hist_a, _ = np.histogram(a, bins=bin_edges, density=False)
    hist_b, _ = np.histogram(b, bins=bin_edges, density=False)

    # Convert to probability distributions
    p = hist_a / hist_a.sum()
    q = hist_b / hist_b.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps

    # Renormalize after adding epsilon
    p = p / p.sum()
    q = q / q.sum()

    # Compute JS divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    # where M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # KL divergence: KL(P||M) = sum(P * log2(P/M))
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    js = 0.5 * kl_pm + 0.5 * kl_qm

    return float(js)


def effect_size_metrics(a: ArrayLike, b: ArrayLike) -> EffectSizeResult:
    """Compute effect-size style metrics between two distributions.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        EffectSizeResult with:
        - median_diff: median(a) - median(b)
        - iqr_diff: IQR(a) - IQR(b), where IQR = Q75 - Q25
    """
    a = np.asarray(a)
    b = np.asarray(b)

    median_a = float(np.median(a))
    median_b = float(np.median(b))

    q25_a, q75_a = np.percentile(a, [25, 75])
    q25_b, q75_b = np.percentile(b, [25, 75])

    iqr_a = q75_a - q25_a
    iqr_b = q75_b - q25_b

    return EffectSizeResult(
        median_diff=median_a - median_b,
        iqr_diff=float(iqr_a - iqr_b),
    )
