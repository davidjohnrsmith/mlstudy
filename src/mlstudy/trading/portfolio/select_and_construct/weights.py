"""Portfolio weight computation utilities.

This module consolidates weight computation from strategy_portfolio_constructor
and new_issuance/portfolio into a unified API.
"""
from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Helper functions
# -----------------------------
def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility given weights and covariance."""
    w = w.reshape(-1)
    v = float(w @ cov @ w)
    return float(np.sqrt(max(v, 0.0)))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Risk contribution to portfolio volatility.

        RC_i = w_i * (Sigma w)_i / vol

    Parameters
    ----------
    w : np.ndarray
        Portfolio weights.
    cov : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Risk contribution per asset.
    """
    w = w.reshape(-1)
    mw = cov @ w
    vol = portfolio_vol(w, cov)
    if vol <= 0:
        return np.zeros_like(w)
    return (w * mw) / vol


def _project_bounds(w: np.ndarray, bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if bounds is None:
        return w
    lo, hi = bounds
    return np.minimum(np.maximum(w, lo), hi)


def _enforce_gross_net(
    w: np.ndarray,
    gross_limit: Optional[float] = None,
    net_limit: Optional[float] = None,
) -> np.ndarray:
    """
    Basic scaling to meet gross and net exposure constraints.

    gross_limit applies to sum(abs(w)).
    net_limit applies to abs(sum(w)).
    """
    w = w.copy()

    if gross_limit is not None:
        gross = float(np.sum(np.abs(w)))
        if gross > 0 and gross > gross_limit:
            w *= gross_limit / gross

    if net_limit is not None:
        net = float(np.sum(w))
        if abs(net) > net_limit and abs(net) > 0:
            shift = (net - np.clip(net, -net_limit, net_limit)) / w.size
            w = w - shift

    return w


# -----------------------------
# Weight computation methods
# -----------------------------
def equal_weights(n: int) -> np.ndarray:
    """
    Equal weights normalized to sum(abs) = 1.

    Parameters
    ----------
    n : int
        Number of assets.

    Returns
    -------
    np.ndarray
        Weight vector of length n.
    """
    if n <= 0:
        return np.array([], dtype=float)
    w = np.ones(n, dtype=float)
    return w / np.sum(np.abs(w))


def equal_weights_series(names: List[str]) -> pd.Series:
    """Equal weights as a Series with asset names as index."""
    if not names:
        return pd.Series(dtype=float)
    w = equal_weights(len(names))
    return pd.Series(w, index=names, dtype=float)


def inverse_vol_weights(
    vols: np.ndarray,
    gross_limit: float = 1.0,
) -> np.ndarray:
    """
    Inverse volatility weights.

    Parameters
    ----------
    vols : np.ndarray
        Volatilities per asset.
    gross_limit : float
        Gross exposure target (sum of abs weights).

    Returns
    -------
    np.ndarray
        Weight vector.
    """
    vols = np.asarray(vols, dtype=float).reshape(-1)
    if vols.size == 0:
        return np.array([], dtype=float)
    inv = np.where(vols > 0, 1.0 / vols, 0.0)
    total = np.sum(inv)
    if total <= 0:
        return np.zeros_like(inv)
    return inv / total * gross_limit


def risk_parity_weights(
    cov: np.ndarray,
    signs: Optional[np.ndarray] = None,
    gross_limit: Optional[float] = 1.0,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_iter: int = 5000,
    tol: float = 1e-8,
    step: float = 0.05,
    seed_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute equal risk contribution (ERC) weights.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix [N x N].
    signs : Optional[np.ndarray]
        Optional vector (+1/-1) to enforce a sign per strategy.
        If provided, solves for x >= 0 and sets w = signs * x.
    gross_limit : Optional[float]
        Gross exposure target applied as sum(abs(w)) <= gross_limit.
        Default 1.0 -> normalized gross weights.
    bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Optional elementwise bounds (lo, hi).
    max_iter : int
        Maximum iterations for the solver.
    tol : float
        Convergence tolerance.
    step : float
        Step size for gradient descent.
    seed_weights : Optional[np.ndarray]
        Initial weights for warm-starting.

    Returns
    -------
    np.ndarray
        Risk parity weights.
    """
    n = cov.shape[0]
    if cov.shape[1] != n:
        raise ValueError("cov must be square")
    if n == 0:
        return np.array([], dtype=float)

    if signs is None:
        signs = np.ones(n, dtype=float)
    signs = np.asarray(signs, dtype=float).reshape(-1)
    if signs.size != n:
        raise ValueError("signs must have length N")

    if seed_weights is None:
        x = np.ones(n, dtype=float) / n
    else:
        w0 = np.asarray(seed_weights, dtype=float).reshape(-1)
        if w0.size != n:
            raise ValueError("seed_weights must have length N")
        x = np.abs(w0) + 1e-12
        x /= np.sum(x)

    target = 1.0 / n

    for _ in range(max_iter):
        w = signs * x
        rc = risk_contributions(w, cov)
        vol = portfolio_vol(w, cov)
        if vol <= 0:
            break

        rc_share = rc / (rc.sum() if rc.sum() != 0 else 1.0)
        err = rc_share - target
        if float(np.max(np.abs(err))) < tol:
            break

        x = x * (1.0 - step * err)
        x = np.maximum(x, 1e-12)
        x /= np.sum(x)

    w = signs * x
    w = _project_bounds(w, bounds)
    w = _enforce_gross_net(w, gross_limit=gross_limit, net_limit=None)
    w = _project_bounds(w, bounds)
    w = _enforce_gross_net(w, gross_limit=gross_limit, net_limit=None)
    return w


def risk_parity_weights_series(
    cov: np.ndarray,
    names: List[str],
    max_iter: int = 3000,
    step: float = 0.05,
) -> pd.Series:
    """
    Risk parity weights as a Series with asset names as index.

    Simplified interface for time-varying universe use cases.
    """
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    w = risk_parity_weights(cov, max_iter=max_iter, step=step, tol=1e-6)
    return pd.Series(w, index=names, dtype=float)


def mean_variance_weights(
    cov: np.ndarray,
    mu: np.ndarray,
    risk_aversion: float = 1.0,
    gross_limit: Optional[float] = 1.0,
    net_limit: Optional[float] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Simple mean-variance weights:
        w = inv(cov + ridge*I) * mu / risk_aversion

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix [N x N].
    mu : np.ndarray
        Expected returns or alpha scores (only relative magnitudes/direction matter).
    risk_aversion : float
        Risk aversion parameter.
    gross_limit : Optional[float]
        Gross exposure constraint.
    net_limit : Optional[float]
        Net exposure constraint.
    bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Elementwise bounds (lo, hi).
    ridge : float
        Ridge regularization for matrix inversion.

    Returns
    -------
    np.ndarray
        Mean-variance weights.
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    mu = np.asarray(mu, dtype=float).reshape(-1)
    if mu.size != n:
        raise ValueError("mu must have length N")

    cov_reg = cov + ridge * np.eye(n)
    w = np.linalg.solve(cov_reg, mu) / float(risk_aversion)

    if gross_limit is not None:
        g = float(np.sum(np.abs(w)))
        if g > 0:
            w *= gross_limit / g

    w = _project_bounds(w, bounds)
    w = _enforce_gross_net(w, gross_limit=gross_limit, net_limit=net_limit)
    w = _project_bounds(w, bounds)
    return w


def no_trade_band(prev_w: np.ndarray, target_w: np.ndarray, band: float = 0.0) -> np.ndarray:
    """
    No-trade band rule:
    - if abs(target_w[i] - prev_w[i]) < band, keep prev_w[i].

    Parameters
    ----------
    prev_w : np.ndarray
        Previous weights.
    target_w : np.ndarray
        Target weights.
    band : float
        Threshold for no-trade band.

    Returns
    -------
    np.ndarray
        Adjusted weights.
    """
    if band <= 0:
        return target_w
    prev_w = prev_w.reshape(-1)
    target_w = target_w.reshape(-1)
    out = target_w.copy()
    mask = np.abs(target_w - prev_w) < band
    out[mask] = prev_w[mask]
    return out
