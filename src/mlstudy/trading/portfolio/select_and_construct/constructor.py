"""Unified portfolio constructor supporting static and time-varying universes.

This module consolidates StrategyPortfolioConstructor and
UniverseAwarePortfolioBacktester into a unified API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from .correlation import (
    CovMethod,
    estimate_cov,
    pairwise_overlap_corr,
    cov_from_pairwise_corr,
    make_psd,
)
from .weights import (
    equal_weights,
    risk_parity_weights,
    mean_variance_weights,
    no_trade_band,
    portfolio_vol,
    risk_contributions,
)
from .gates import UniverseConfig, StandaloneGates, RedundancyGates
from .selector import StrategySelector, build_returns_panel


WeightMethod = Literal["equal", "risk_parity", "mean_variance"]


def _clean_series(s: pd.Series) -> pd.Series:
    """Clean a return series: dedupe, sort, convert to float."""
    s = s.copy()
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s.astype(float).dropna()


@dataclass(frozen=True)
class StrategyBacktest:
    """
    Container for a single strategy backtest return series.

    Parameters
    ----------
    name : str
        Strategy identifier.
    returns : pd.Series
        Periodic returns (e.g., daily), indexed by datetime-like.
        Returns are assumed to be arithmetic returns (not log returns).
    """
    name: str
    returns: pd.Series

    def cleaned_returns(self) -> pd.Series:
        """Returns a cleaned return series: dedupe, sort, drop NaN."""
        return _clean_series(self.returns)

    def realized_vol(self, ann_factor: float = 252.0) -> float:
        """Annualized realized volatility."""
        r = self.cleaned_returns()
        if len(r) < 2:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(ann_factor))

    def sharpe(self, ann_factor: float = 252.0) -> float:
        """Annualized Sharpe ratio."""
        r = self.cleaned_returns()
        if len(r) < 2:
            return np.nan
        vol = r.std(ddof=1)
        if vol == 0:
            return np.nan
        return float((r.mean() / vol) * np.sqrt(ann_factor))


def align_strategy_returns(
    strategies: Dict[str, StrategyBacktest],
    how: Literal["inner", "outer"] = "inner",
    fillna: Optional[float] = None,
) -> pd.DataFrame:
    """
    Align strategy returns into a single DataFrame [T x N].

    Parameters
    ----------
    strategies : Dict[str, StrategyBacktest]
        Mapping from name to StrategyBacktest.
    how : Literal["inner", "outer"]
        'inner' keeps only overlapping dates.
        'outer' unions all dates (may introduce NaNs).
    fillna : Optional[float]
        If provided, fill missing values after alignment.
    """
    series = {k: v.cleaned_returns() for k, v in strategies.items()}
    df = pd.concat(series, axis=1, join=how).sort_index()
    if fillna is not None:
        df = df.fillna(fillna)
    return df


class PortfolioConstructor:
    """
    Unified portfolio constructor for combining strategies.

    Supports two modes:
    1. Static universe: Use with pre-selected strategies.
       See `construct()` and `backtest()` methods.

    2. Time-varying universe: Use with StrategySelector for dynamic rebalancing.
       See `run_dynamic_backtest()` method.

    Features
    --------
    - Align returns and estimate covariance (sample/EWMA/shrinkage/Ledoit-Wolf).
    - Vol-target scaling of strategies.
    - Weight computation: equal, risk parity (ERC), mean-variance.
    - No-trade bands & transaction-cost modeling.
    - Support for non-synchronous data via pairwise overlap correlation.
    """

    def __init__(
        self,
        strategies: Optional[Dict[str, StrategyBacktest]] = None,
        returns_panel: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize portfolio constructor.

        Parameters
        ----------
        strategies : Optional[Dict[str, StrategyBacktest]]
            For static universe: mapping from name to StrategyBacktest.
        returns_panel : Optional[pd.DataFrame]
            For time-varying universe: returns panel [T x N] with NaN where inactive.
        """
        if strategies is not None:
            self.strategies = dict(strategies)
            self._panel = align_strategy_returns(strategies, how="outer")
        elif returns_panel is not None:
            self.strategies = None
            self._panel = returns_panel.copy().sort_index()
        else:
            raise ValueError("Either strategies or returns_panel must be provided")

    @classmethod
    def from_backtests(cls, strategies: Dict[str, StrategyBacktest]) -> "PortfolioConstructor":
        """Create constructor from StrategyBacktest objects."""
        return cls(strategies=strategies)

    @classmethod
    def from_panel(cls, returns_panel: pd.DataFrame) -> "PortfolioConstructor":
        """Create constructor from returns panel."""
        return cls(returns_panel=returns_panel)

    @classmethod
    def from_series(cls, returns: Dict[str, pd.Series]) -> "PortfolioConstructor":
        """Create constructor from return series dict."""
        panel = build_returns_panel(returns)
        return cls(returns_panel=panel)

    def returns_df(self, how: Literal["inner", "outer"] = "inner") -> pd.DataFrame:
        """Get aligned returns DataFrame."""
        if how == "inner":
            return self._panel.dropna()
        return self._panel

    # -------------------------
    # Covariance estimation
    # -------------------------
    def scale_to_target_vol(
        self,
        returns: pd.DataFrame,
        target_vol: float,
        ann_factor: float = 252.0,
        vol_window: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Scale each strategy's returns to have approximately target_vol (annualized).

        Parameters
        ----------
        returns : pd.DataFrame
            Returns matrix [T x N].
        target_vol : float
            Target annualized volatility.
        ann_factor : float
            Annualization factor.
        vol_window : Optional[int]
            If provided, use only the last vol_window observations for scaling.

        Returns
        -------
        scaled_returns : pd.DataFrame
            Scaled returns.
        scales : np.ndarray
            Multipliers applied to each column.
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

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: CovMethod = "diag_shrink",
        ewma_half_life: int = 40,
        shrink_alpha: float = 0.1,
    ) -> np.ndarray:
        """
        Estimate covariance matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Returns matrix [T x N].
        method : CovMethod
            Estimation method: 'sample', 'ewma', 'diag_shrink', 'ledoit_wolf'.
        ewma_half_life : int
            Half-life for EWMA method.
        shrink_alpha : float
            Shrinkage intensity for diag_shrink method.
        """
        return estimate_cov(
            returns=returns,
            method=method,
            ewma_half_life=ewma_half_life,
            shrink_alpha=shrink_alpha,
        )

    def estimate_covariance_pairwise(
        self,
        returns: pd.DataFrame,
        min_overlap: int = 60,
    ) -> np.ndarray:
        """
        Estimate covariance using pairwise overlap correlation.

        Use this for non-synchronous data where strategies have different active periods.

        Parameters
        ----------
        returns : pd.DataFrame
            Returns matrix [T x N] with NaN where data is missing.
        min_overlap : int
            Minimum overlapping observations for correlation.

        Returns
        -------
        np.ndarray
            Covariance matrix (PSD-projected).
        """
        corr, _ = pairwise_overlap_corr(returns, min_obs=min_overlap)

        # Fill missing correlations with 0 (conservative), keep diagonal 1
        corr_filled = corr.copy()
        arr = np.array(corr_filled.to_numpy(copy=False), copy=True)
        np.fill_diagonal(arr, 1.0)
        corr_filled = corr_filled.copy()
        corr_filled.iloc[:, :] = arr
        corr_filled = corr_filled.fillna(0.0)

        cov = cov_from_pairwise_corr(returns, corr_filled)
        return make_psd(cov, eps=1e-10)

    # -------------------------
    # Weight computation
    # -------------------------
    def compute_weights(
        self,
        cov: np.ndarray,
        method: WeightMethod = "risk_parity",
        mu: Optional[np.ndarray] = None,
        signs: Optional[np.ndarray] = None,
        gross_limit: float = 1.0,
        risk_aversion: float = 1.0,
    ) -> np.ndarray:
        """
        Compute portfolio weights.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix [N x N].
        method : WeightMethod
            'equal', 'risk_parity', or 'mean_variance'.
        mu : Optional[np.ndarray]
            Expected returns (required for mean_variance).
        signs : Optional[np.ndarray]
            Sign constraints for risk_parity (+1 or -1 per asset).
        gross_limit : float
            Gross exposure constraint.
        risk_aversion : float
            Risk aversion for mean_variance.

        Returns
        -------
        np.ndarray
            Portfolio weights.
        """
        n = cov.shape[0]
        if n == 0:
            return np.array([], dtype=float)

        if method == "equal":
            return equal_weights(n) * gross_limit
        elif method == "risk_parity":
            return risk_parity_weights(cov, signs=signs, gross_limit=gross_limit)
        elif method == "mean_variance":
            if mu is None:
                raise ValueError("mu required for mean_variance method")
            return mean_variance_weights(cov, mu, risk_aversion=risk_aversion, gross_limit=gross_limit)
        else:
            raise ValueError(f"Unknown weight method: {method}")

    # -------------------------
    # Static portfolio backtest
    # -------------------------
    def construct(
        self,
        strategy_names: Optional[List[str]] = None,
        cov_method: CovMethod = "diag_shrink",
        weight_method: WeightMethod = "risk_parity",
        target_vol: Optional[float] = None,
        mu: Optional[np.ndarray] = None,
        gross_limit: float = 1.0,
        ann_factor: float = 252.0,
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Construct portfolio weights for a static set of strategies.

        Parameters
        ----------
        strategy_names : Optional[List[str]]
            Strategies to include (default: all).
        cov_method : CovMethod
            Covariance estimation method.
        weight_method : WeightMethod
            Weight computation method.
        target_vol : Optional[float]
            If provided, scale strategies to this target vol before weighting.
        mu : Optional[np.ndarray]
            Expected returns for mean_variance.
        gross_limit : float
            Gross exposure constraint.
        ann_factor : float
            Annualization factor.

        Returns
        -------
        weights : np.ndarray
            Portfolio weights.
        returns : pd.DataFrame
            (Possibly scaled) returns used for construction.
        cov : np.ndarray
            Estimated covariance matrix.
        """
        if strategy_names is None:
            strategy_names = list(self._panel.columns)

        R = self._panel[strategy_names].dropna()
        if R.shape[0] < 2:
            raise ValueError("Not enough overlapping data for portfolio construction.")

        if target_vol is not None:
            R, _ = self.scale_to_target_vol(R, target_vol, ann_factor=ann_factor)

        cov = self.estimate_covariance(R, method=cov_method)
        weights = self.compute_weights(cov, method=weight_method, mu=mu, gross_limit=gross_limit)

        return weights, R, cov

    def backtest(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        tc_per_unit_turnover: float = 0.0,
        prev_weights: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """
        Compute portfolio return series from constituent returns and static weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Returns matrix [T x N].
        weights : np.ndarray
            Portfolio weights.
        tc_per_unit_turnover : float
            Transaction cost per unit of turnover.
        prev_weights : Optional[np.ndarray]
            Previous weights (for turnover-based TC).

        Returns
        -------
        pd.Series
            Portfolio return series.
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

    # -------------------------
    # Time-varying universe backtest
    # -------------------------
    def run_dynamic_backtest(
        self,
        selector: StrategySelector,
        universe_cfg: UniverseConfig = UniverseConfig(),
        standalone: StandaloneGates = StandaloneGates(),
        redundancy: RedundancyGates = RedundancyGates(),
        weight_method: WeightMethod = "risk_parity",
        cov_min_overlap: int = 60,
        tc_per_turnover: float = 0.0,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run a dynamic backtest with time-varying universe.

        Process:
          - On each rebalance date:
              * Pre-select eligible strategies using selector
              * Estimate covariance on trailing window (pairwise overlap)
              * Compute weights
          - Hold weights until next rebalance
          - Portfolio return each day is sum_i w_i * r_i

        Parameters
        ----------
        selector : StrategySelector
            Configured selector for strategy selection.
        universe_cfg : UniverseConfig
            Rebalancing and lookback config.
        standalone : StandaloneGates
            Standalone screening criteria.
        redundancy : RedundancyGates
            Redundancy reduction config.
        weight_method : WeightMethod
            'equal' or 'risk_parity'.
        cov_min_overlap : int
            Minimum overlap for pairwise covariance.
        tc_per_turnover : float
            Transaction cost per unit turnover.

        Returns
        -------
        portfolio_returns : pd.Series
            Portfolio return series.
        weights_history : pd.DataFrame
            Weights at each rebalance date.
        """
        R = self._panel

        # Generate rebalance dates
        cal = pd.date_range(R.index.min(), R.index.max(), freq=universe_cfg.rebalance_freq)
        snapped = []
        for t in cal:
            loc = R.index.searchsorted(t, side="right") - 1
            if loc >= 0:
                snapped.append(R.index[loc])
        rebal_dates = pd.DatetimeIndex(sorted(set(snapped)))

        if len(rebal_dates) == 0:
            raise ValueError("No rebalance dates found within returns_panel index.")

        weights_hist: Dict[pd.Timestamp, pd.Series] = {}
        port = pd.Series(0.0, index=R.index, name="portfolio_return")
        prev_w = pd.Series(dtype=float)

        for k, t in enumerate(rebal_dates):
            selected, metrics, clusters = selector.select_at(
                t, universe_cfg=universe_cfg, standalone=standalone, redundancy=redundancy
            )

            if not selected:
                w = pd.Series(dtype=float)
            else:
                # Get trailing window for covariance
                W = selector._get_window(t, universe_cfg.lookback)[selected]
                cov = self.estimate_covariance_pairwise(W, min_overlap=cov_min_overlap)

                weights_arr = self.compute_weights(
                    cov, method=weight_method, gross_limit=1.0
                )
                w = pd.Series(weights_arr, index=selected, dtype=float)

            # Transaction cost
            if tc_per_turnover > 0 and not prev_w.empty:
                all_names = sorted(set(prev_w.index).union(set(w.index)))
                w_prev = prev_w.reindex(all_names).fillna(0.0)
                w_new = w.reindex(all_names).fillna(0.0)
                turnover = float((w_new - w_prev).abs().sum())
                port.loc[t] -= tc_per_turnover * turnover

            weights_hist[t] = w
            prev_w = w

            # Apply weights from t until next rebalance
            t_end = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else None
            slice_idx = R.loc[t:t_end].index if t_end is None else R.loc[t:t_end].index[:-1]

            if w.empty:
                continue

            # Portfolio return (NaN returns treated as 0)
            subR = R.loc[slice_idx, w.index]
            port.loc[slice_idx] += (subR.fillna(0.0).to_numpy(dtype=float) @ w.to_numpy(dtype=float))

        weights_df = pd.DataFrame(weights_hist).T.sort_index()
        return port, weights_df


# Aliases for backward compatibility
StrategyPortfolioConstructor = PortfolioConstructor
UniverseAwarePortfolioBacktester = PortfolioConstructor
