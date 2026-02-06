"""Correlation and covariance estimation utilities.

This module consolidates correlation/covariance functions from
strategy_pre_selector, strategy_portfolio_constructor, and new_issuance/portfolio.
"""
from __future__ import annotations

from typing import Tuple, Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd


CovMethod = Literal["sample", "ewma", "diag_shrink", "ledoit_wolf"]
CorrMethod = Literal["pearson", "spearman"]


# -----------------------------
# Correlation
# -----------------------------
def corr_matrix(df: pd.DataFrame, method: CorrMethod = "pearson") -> pd.DataFrame:
    """
    Compute correlation matrix after dropping rows with any NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Returns matrix [T x N].
    method : CorrMethod
        "pearson" or "spearman".

    Returns
    -------
    pd.DataFrame
        Correlation matrix [N x N].
    """
    d = df.dropna()
    if d.shape[0] < 2:
        raise ValueError("Not enough overlapping data to compute correlation.")
    return d.corr(method=method)


def pairwise_overlap_corr(
    df: pd.DataFrame,
    min_obs: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise correlation computed on overlapping (non-NaN) observations per pair.

    Use this for non-synchronous data where different columns have different
    active periods (e.g., instruments with different issuance dates).

    Parameters
    ----------
    df : pd.DataFrame
        Returns matrix [T x N] with NaN where data is missing.
    min_obs : int
        Minimum overlapping observations to compute correlation.

    Returns
    -------
    corr : pd.DataFrame
        Correlation matrix [N x N] with NaN where overlap < min_obs.
    nobs : pd.DataFrame
        Overlap count matrix [N x N].
    """
    cols = list(df.columns)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    nobs = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    for i, a in enumerate(cols):
        for j in range(i, len(cols)):
            b = cols[j]
            both = df[[a, b]].dropna()
            n = both.shape[0]
            nobs.loc[a, b] = nobs.loc[b, a] = n
            if a == b:
                corr.loc[a, a] = 1.0 if n >= 2 else np.nan
            elif n >= min_obs:
                corr.loc[a, b] = corr.loc[b, a] = float(both[a].corr(both[b]))
    return corr, nobs


def vol_scale(df: pd.DataFrame, target_vol: float = 1.0) -> pd.DataFrame:
    """
    Scale each column to have the same sample stdev (in-sample).

    target_vol=1 gives unit-vol series for correlation/clustering stability.
    """
    d = df.dropna()
    vols = d.std(ddof=1)
    scale = target_vol / vols.replace(0.0, np.nan)
    scale = scale.fillna(0.0)
    return d * scale


# -----------------------------
# Covariance Estimation
# -----------------------------
def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance matrix from returns."""
    r = returns.dropna()
    if r.shape[0] < 2:
        raise ValueError("Not enough overlapping data for sample covariance.")
    return np.cov(r.to_numpy(dtype=float), rowvar=False, ddof=1)


def ewma_cov(returns: pd.DataFrame, half_life: int = 40) -> np.ndarray:
    """
    Exponentially weighted covariance (RiskMetrics-style) using half-life in periods.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix [T x N].
    half_life : int
        Decay half-life in periods.

    Notes
    -----
    - Uses exponentially decaying weights with newest observations weighted most.
    - Drops any rows containing NaNs.
    """
    x = returns.to_numpy(dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)

    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    if x.shape[0] < 2:
        raise ValueError("Not enough data after dropping NaNs for EWMA covariance.")

    lam = 0.5 ** (1.0 / float(half_life))
    T = x.shape[0]
    w = (1.0 - lam) * lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()

    xw = x * w[:, None]
    cov = xw.T @ x
    return cov


def diag_shrink_cov(sample_cov_mat: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Simple diagonal shrinkage:
        Sigma = (1-alpha) * S + alpha * diag(S)

    Parameters
    ----------
    sample_cov_mat : np.ndarray
        Sample covariance matrix.
    alpha : float
        Shrinkage intensity in [0,1].
        alpha=0 -> sample covariance, alpha=1 -> diagonal covariance.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    d = np.diag(np.diag(sample_cov_mat))
    return (1.0 - alpha) * sample_cov_mat + alpha * d


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf covariance shrinkage (requires sklearn).

    Falls back to diagonal shrinkage if sklearn isn't available.
    """
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore
    except ImportError:
        s = sample_cov(returns)
        return diag_shrink_cov(s, alpha=0.15)

    x = returns.dropna().to_numpy(dtype=float)
    if x.shape[0] < 2:
        raise ValueError("Not enough data for Ledoit-Wolf covariance.")
    lw = LedoitWolf().fit(x)
    return lw.covariance_


def estimate_cov(
    returns: pd.DataFrame,
    method: CovMethod = "diag_shrink",
    ewma_half_life: int = 40,
    shrink_alpha: float = 0.1,
) -> np.ndarray:
    """
    Estimate covariance matrix from returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of aligned returns [T x N].
    method : CovMethod
        Covariance estimator:
        - 'sample'       : sample covariance
        - 'ewma'         : exponentially weighted covariance
        - 'diag_shrink'  : sample covariance shrunk toward diagonal
        - 'ledoit_wolf'  : Ledoit-Wolf shrinkage (sklearn)
    ewma_half_life : int
        Half-life for EWMA method.
    shrink_alpha : float
        Shrinkage intensity for diag_shrink method.
    """
    r = returns.dropna()
    if r.shape[0] < 2:
        raise ValueError("Not enough overlapping data to estimate covariance.")

    if method == "sample":
        return sample_cov(r)
    if method == "ewma":
        return ewma_cov(r, half_life=ewma_half_life)
    if method == "diag_shrink":
        s = sample_cov(r)
        return diag_shrink_cov(s, alpha=shrink_alpha)
    if method == "ledoit_wolf":
        return ledoit_wolf_cov(r)

    raise ValueError(f"Unknown covariance method: {method}")


def cov_from_pairwise_corr(df: pd.DataFrame, corr: pd.DataFrame) -> np.ndarray:
    """
    Convert a pairwise correlation matrix into covariance using per-column std.

    Parameters
    ----------
    df : pd.DataFrame
        Returns matrix used to compute per-column std (on available data).
    corr : pd.DataFrame
        Correlation matrix (can have NaN for pairs with insufficient overlap).

    Returns
    -------
    np.ndarray
        Covariance matrix.

    Notes
    -----
    Stds are computed on available data per column (can differ).
    """
    std = df.std(ddof=1).to_numpy(dtype=float)
    C = corr.to_numpy(dtype=float)
    cov = C * std[:, None] * std[None, :]
    return cov


def make_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a symmetric matrix to nearest PSD via eigenvalue clipping.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric matrix (possibly indefinite).
    eps : float
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray
        Positive semi-definite matrix.
    """
    A = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T
