from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd


CovMethod = Literal["sample", "ewma", "diag_shrink", "ledoit_wolf"]


# -----------------------------
# Generic strategy container
# -----------------------------
@dataclass(frozen=True)
class StrategyBacktest:
    """
    Container for a single strategy backtest return series.

    Parameters
    ----------
    name:
        Strategy identifier.
    returns:
        pd.Series of periodic returns (e.g., daily), indexed by datetime-like.
        Returns are assumed to be arithmetic returns (not log returns).
    """
    name: str
    returns: pd.Series

    def cleaned_returns(self) -> pd.Series:
        """
        Returns a cleaned return series:
        - drops duplicate timestamps (keeps last)
        - sorts by index
        - drops NaNs
        """
        r = self.returns.copy()
        r = r[~r.index.duplicated(keep="last")]
        r = r.sort_index()
        return r.dropna()

    def realized_vol(self, ann_factor: float = 252.0) -> float:
        """Annualized realized volatility (std * sqrt(ann_factor))."""
        r = self.cleaned_returns()
        if len(r) < 2:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(ann_factor))

    def sharpe(self, ann_factor: float = 252.0) -> float:
        """Annualized Sharpe ratio (mean / std * sqrt(ann_factor))."""
        r = self.cleaned_returns()
        if len(r) < 2:
            return np.nan
        vol = r.std(ddof=1)
        if vol == 0:
            return np.nan
        return float((r.mean() / vol) * np.sqrt(ann_factor))


# -----------------------------
# Data alignment
# -----------------------------
def align_returns(
    strategies: Dict[str, StrategyBacktest],
    how: Literal["inner", "outer"] = "inner",
    fillna: Optional[float] = None,
) -> pd.DataFrame:
    """
    Align strategy returns into a single DataFrame [T x N].

    Parameters
    ----------
    strategies:
        Dict mapping name -> StrategyBacktest
    how:
        'inner' keeps only overlapping dates across all strategies.
        'outer' unions all dates (may introduce NaNs).
    fillna:
        If provided, fill missing values after alignment.
        Warning: filling with 0.0 can be inappropriate for many strategies.

    Returns
    -------
    pd.DataFrame
        DataFrame of aligned returns with columns as strategies.
    """
    series = {k: v.cleaned_returns() for k, v in strategies.items()}
    df = pd.concat(series, axis=1, join=how).sort_index()
    if fillna is not None:
        df = df.fillna(fillna)
    return df


# -----------------------------
# Covariance estimation
# -----------------------------
def ewma_cov(returns: pd.DataFrame, half_life: int = 40) -> np.ndarray:
    """
    Exponentially weighted covariance (RiskMetrics-style) using half-life in periods.

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


def diag_shrink_cov(sample_cov: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Simple diagonal shrinkage:
        Sigma = (1-alpha) * S + alpha * diag(S)

    Parameters
    ----------
    sample_cov:
        Sample covariance matrix.
    alpha:
        Shrinkage intensity in [0,1].
        alpha=0 -> sample covariance, alpha=1 -> diagonal covariance.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    d = np.diag(np.diag(sample_cov))
    return (1.0 - alpha) * sample_cov + alpha * d


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf covariance shrinkage (requires sklearn).
    Falls back to diagonal shrinkage if sklearn isn't available.
    """
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore
    except Exception:
        s = np.cov(returns.dropna().to_numpy(dtype=float), rowvar=False, ddof=1)
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
    returns:
        DataFrame of aligned returns [T x N].
    method:
        Covariance estimator:
        - 'sample'       : sample covariance
        - 'ewma'         : exponentially weighted covariance
        - 'diag_shrink'  : sample covariance shrunk toward diagonal
        - 'ledoit_wolf'  : Ledoit-Wolf shrinkage (sklearn)
    """
    r = returns.dropna()
    if r.shape[0] < 2:
        raise ValueError("Not enough overlapping data to estimate covariance.")

    if method == "sample":
        return np.cov(r.to_numpy(dtype=float), rowvar=False, ddof=1)
    if method == "ewma":
        return ewma_cov(r, half_life=ewma_half_life)
    if method == "diag_shrink":
        s = np.cov(r.to_numpy(dtype=float), rowvar=False, ddof=1)
        return diag_shrink_cov(s, alpha=shrink_alpha)
    if method == "ledoit_wolf":
        return ledoit_wolf_cov(r)

    raise ValueError(f"Unknown covariance method: {method}")


# -----------------------------
# Risk + constraints helpers
# -----------------------------
def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility given weights and covariance."""
    w = w.reshape(-1)
    v = float(w @ cov @ w)
    return float(np.sqrt(max(v, 0.0)))


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Risk contribution to portfolio volatility:
        RC_i = w_i * (Sigma w)_i / vol
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


def no_trade_band(prev_w: np.ndarray, target_w: np.ndarray, band: float = 0.0) -> np.ndarray:
    """
    No-trade band rule:
    - if abs(target_w[i] - prev_w[i]) < band, keep prev_w[i].
    """
    if band <= 0:
        return target_w
    prev_w = prev_w.reshape(-1)
    target_w = target_w.reshape(-1)
    out = target_w.copy()
    mask = np.abs(target_w - prev_w) < band
    out[mask] = prev_w[mask]
    return out


# -----------------------------
# Generic portfolio constructor
# -----------------------------
class StrategyPortfolioConstructor:
    """
    Combine independently backtested strategies into a portfolio.

    Features
    --------
    - Align returns and estimate covariance (sample/EWMA/shrinkage/Ledoit-Wolf).
    - Vol-target scaling of strategies.
    - Risk parity (equal risk contribution) with optional sign constraints.
    - Mean-variance style weights with simple constraints.
    - No-trade bands & simple transaction-cost helper for static allocations.

    Notes
    -----
    - This class expects *return series* as inputs. It does not know anything about
      instruments, DV01, curve buckets, etc. Those can be layered on as constraints
      if you supply exposures separately.
    """

    def __init__(self, strategies: Dict[str, StrategyBacktest]):
        if not strategies:
            raise ValueError("strategies must be non-empty")
        self.strategies = dict(strategies)

    def returns_df(self, how: Literal["inner", "outer"] = "inner") -> pd.DataFrame:
        return align_returns(self.strategies, how=how)

    def scale_to_target_vol(
        self,
        returns: pd.DataFrame,
        target_vol: float,
        ann_factor: float = 252.0,
        vol_window: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Scale each strategy's returns so it has approximately target_vol (annualized).

        If vol_window is provided, use only the last vol_window observations to compute
        the scaling factor (single point-in-time scale based on the most recent window).

        Returns
        -------
        scaled_returns:
            returns scaled by per-strategy multipliers.
        scales:
            array of multipliers applied to each column.
        """
        r = returns.dropna()
        if r.shape[0] < 2:
            raise ValueError("Not enough data to scale.")

        window = r.tail(vol_window) if vol_window is not None else r

        vols = window.std(ddof=1).to_numpy(dtype=float) * np.sqrt(ann_factor)
        vols = np.where(vols <= 0, np.nan, vols)
        scales = target_vol / vols
        scales = np.where(np.isfinite(scales), scales, 0.0)

        scaled = r * scales
        return scaled, scales

    def covariance(
        self,
        returns: pd.DataFrame,
        method: CovMethod = "diag_shrink",
        ewma_half_life: int = 40,
        shrink_alpha: float = 0.1,
    ) -> np.ndarray:
        return estimate_cov(
            returns=returns,
            method=method,
            ewma_half_life=ewma_half_life,
            shrink_alpha=shrink_alpha,
        )

    def risk_parity_weights(
        self,
        cov: np.ndarray,
        signs: Optional[np.ndarray] = None,
        gross_limit: Optional[float] = 1.0,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_iter: int = 5_000,
        tol: float = 1e-8,
        step: float = 0.05,
        seed_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute equal risk contribution (ERC) weights.

        Parameters
        ----------
        cov:
            Covariance matrix [N x N].
        signs:
            Optional vector (+1/-1) to enforce a sign per strategy.
            If provided, solves for x >= 0 and sets w = signs * x.
        gross_limit:
            Gross exposure target applied as sum(abs(w)) <= gross_limit.
            Default 1.0 -> normalized gross weights.
        bounds:
            Optional elementwise bounds (lo, hi).
        """
        n = cov.shape[0]
        if cov.shape[1] != n:
            raise ValueError("cov must be square")
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

    def mean_variance_weights(
        self,
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
            w ∝ inv(cov + ridge*I) * mu / risk_aversion

        mu can be:
        - expected returns, OR
        - alpha scores (only relative magnitudes/direction matter).
        """
        n = cov.shape[0]
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

    def portfolio_backtest(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        tc_per_unit_turnover: float = 0.0,
        prev_weights: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Compute portfolio return series from constituent returns and a *static* weight vector.

        If prev_weights is provided, applies a one-time transaction cost at the start:
            cost = tc_per_unit_turnover * sum(abs(w - prev_w))
        """
        r = returns.dropna()
        w = np.asarray(weights, dtype=float).reshape(-1)
        if r.shape[1] != w.size:
            raise ValueError("weights length must match number of columns in returns")

        port = pd.Series(r.to_numpy(dtype=float) @ w, index=r.index, name="portfolio_return")

        if prev_weights is not None and tc_per_unit_turnover > 0:
            prev_w = np.asarray(prev_weights, dtype=float).reshape(-1)
            if prev_w.size != w.size:
                raise ValueError("prev_weights length mismatch")
            turnover = float(np.sum(np.abs(w - prev_w)))
            port.iloc[0] -= tc_per_unit_turnover * turnover

        return port


# ----------------------------
# Example usage (generic)
# ----------------------------
if __name__ == "__main__":
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    rng = np.random.default_rng(7)
    s1 = pd.Series(0.0002 + 0.010 * rng.standard_normal(len(idx)), index=idx)
    s2 = pd.Series(0.0001 + 0.008 * rng.standard_normal(len(idx)), index=idx)
    s3 = pd.Series(0.00015 + 0.009 * rng.standard_normal(len(idx)), index=idx)

    strategies = {
        "strategy_1": StrategyBacktest("strategy_1", s1),
        "strategy_2": StrategyBacktest("strategy_2", s2),
        "strategy_3": StrategyBacktest("strategy_3", s3),
    }

    pc = StrategyPortfolioConstructor(strategies)
    R = pc.returns_df(how="inner")

    # Scale to equal annualized vol
    R_scaled, scales = pc.scale_to_target_vol(R, target_vol=0.08, ann_factor=252)

    # Covariance estimate
    cov = pc.covariance(R_scaled, method="diag_shrink", shrink_alpha=0.15)

    # Risk parity weights
    w_rp = pc.risk_parity_weights(cov, gross_limit=1.0)

    # Mean-variance weights (if you have alpha scores / forecasts)
    mu = np.array([0.7, 0.2, 0.5])
    w_mv = pc.mean_variance_weights(cov, mu=mu, risk_aversion=1.0, gross_limit=1.0)

    port_rp = pc.portfolio_backtest(R_scaled, w_rp)
    port_mv = pc.portfolio_backtest(R_scaled, w_mv)

    print("Scales:", scales)
    print("RP weights:", dict(zip(R_scaled.columns, w_rp.round(4))))
    print("MV weights:", dict(zip(R_scaled.columns, w_mv.round(4))))
    print("RP ann vol:", port_rp.std(ddof=1) * np.sqrt(252))
    print("MV ann vol:", port_mv.std(ddof=1) * np.sqrt(252))
