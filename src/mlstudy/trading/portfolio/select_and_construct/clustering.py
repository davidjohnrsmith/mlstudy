"""Clustering algorithms for redundancy reduction.

This module consolidates hierarchical_cluster and threshold_clusters
into a unified API.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def correlation_clusters(
    corr: pd.DataFrame,
    threshold: float = 0.8,
) -> Dict[int, List[str]]:
    """
    Build clusters as connected components where edge exists if corr >= threshold.

    This is a simple graph-based clustering using a correlation threshold.
    (Avoids scipy dependency. If you want dendrogram clustering, add scipy.)

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix [N x N] with strategy names as index/columns.
    threshold : float
        Correlation threshold for linking two strategies.
        Two strategies are connected if corr >= threshold.

    Returns
    -------
    Dict[int, List[str]]
        Mapping from cluster ID to list of strategy names in that cluster.

    Notes
    -----
    - Correlations should typically be computed after vol-scaling strategies.
    - NaN correlations are treated as < threshold (no edge).
    """
    names = list(corr.columns)
    C = corr.to_numpy(dtype=float)
    n = len(names)

    # Build adjacency list
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(C[i, j]) and C[i, j] >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components via DFS
    seen: set = set()
    clusters: Dict[int, List[str]] = {}
    cid = 0
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp: List[str] = []
        while stack:
            u = stack.pop()
            comp.append(names[u])
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        clusters[cid] = comp
        cid += 1

    return clusters


# Aliases for backward compatibility
hierarchical_cluster = correlation_clusters
threshold_clusters = correlation_clusters
