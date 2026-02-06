"""Screening gates for strategy selection.

This module consolidates StandaloneGates and RedundancyGates from
strategy_pre_selector and new_issuance/portfolio into a unified API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


RankMetric = Literal["sharpe", "sortino", "nw_tstat", "ann_return"]
CorrMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class StandaloneGates:
    """
    Standalone (single-strategy) screening criteria.

    This dataclass defines conservative "quality gates" applied to each strategy
    independently, before any portfolio construction or redundancy reduction.
    The intent is to filter out strategies that are structurally weak, fragile,
    or statistically unsupported (e.g., poor risk-adjusted returns, excessive
    tail risk, insufficient history), rather than to aggressively "pick winners".

    Any threshold can be disabled by setting it to None (except min_obs).

    Attributes
    ----------
    min_obs : int, default 252
        Minimum number of observations required for the strategy to be considered.
        With daily data, 252 observations is approximately one trading year.
        This guards covariance/correlation estimation stability and reduces
        selection bias from short samples.

    min_sharpe : Optional[float], default 0.3
        Minimum required annualized Sharpe ratio for the strategy's return stream.
        Computed as: mean(returns) / std(returns) * sqrt(ann_factor)
        Use this as a broad quality screen, not a fine ranking tool.

    min_nw_tstat : Optional[float], default 1.0
        Minimum required Newey-West t-statistic of the mean return.
        This screens for statistical support of a non-zero mean in the presence
        of autocorrelation and heteroskedasticity in returns.
        Values around 1.0 are a light filter; 2.0 is stricter.

    max_mdd : Optional[float], default -0.25
        Maximum allowed max drawdown (MDD), expressed as a negative number.
        The strategy passes if: max_drawdown >= max_mdd
        Example: max_mdd = -0.25 means no drawdowns worse than -25%.

    max_worst_period : Optional[float], default -0.12
        Maximum allowed worst rolling period return.
        The strategy passes if: worst_rolling_return >= max_worst_period
        Example: max_worst_period = -0.12 means worst rolling return must be >= -12%.

    min_positive_year_frac : Optional[float], default 0.5
        Minimum fraction of calendar years with positive cumulative return.
        0.5 means at least half of years must be positive.
        For shorter histories (< 3 years), this metric becomes noisy.
        Set to None to disable.

    ann_factor : float, default 252.0
        Annualization factor for metrics (252 for daily data).

    nw_lags : int, default 5
        Number of lags for Newey-West t-statistic.

    worst_period_window : int, default 21
        Rolling window size for worst period return (21 ~= 1 month for daily).
    """
    min_obs: int = 252
    min_sharpe: Optional[float] = 0.3
    min_nw_tstat: Optional[float] = 1.0
    max_mdd: Optional[float] = -0.25
    max_worst_period: Optional[float] = -0.12
    min_positive_year_frac: Optional[float] = 0.5
    ann_factor: float = 252.0
    nw_lags: int = 5
    worst_period_window: int = 21


@dataclass(frozen=True)
class RedundancyGates:
    """
    Redundancy reduction controls (cross-strategy filtering).

    After standalone gates are applied, many strategies can still be redundant,
    meaning they deliver largely the same risk/return profile (high correlation,
    similar drivers, overlapping factor exposures). Redundancy reduction tries
    to keep a smaller, more diverse set by clustering strategies based on
    correlation and retaining only the best-ranked strategies in each cluster.

    Attributes
    ----------
    enabled : bool, default True
        Whether to perform redundancy reduction.
        - True: cluster strategies and keep top strategies per cluster.
        - False: skip this step and keep all strategies that pass standalone gates.

    corr_method : CorrMethod, default "pearson"
        Correlation method used to compute the similarity matrix prior to clustering.
        - "pearson": linear correlation (common default for returns).
        - "spearman": rank correlation (more robust to outliers).

    corr_threshold : float, default 0.8
        Correlation threshold that defines "linked" strategies in clustering.
        Two strategies are connected if corr(i, j) >= corr_threshold.
        Clusters are formed as connected components of this threshold graph.
        - 0.7-0.85 is typical for redundancy reduction.
        - Higher thresholds keep more strategies.

    corr_min_overlap : int, default 60
        Minimum overlapping observations required to compute pairwise correlation.
        Only relevant for time-varying universes with non-synchronous data.

    keep_per_cluster : int, default 2
        Number of strategies to keep in each correlation cluster.
        - 1 means "keep only the single best" per cluster (aggressive pruning).
        - 2-4 often works well for internal variety within a theme.

    rank_metric : RankMetric, default "sharpe"
        Metric used to rank strategies within each cluster.
        Options: "sharpe", "sortino", "nw_tstat", "ann_return"
    """
    enabled: bool = True
    corr_method: CorrMethod = "pearson"
    corr_threshold: float = 0.8
    corr_min_overlap: int = 60
    keep_per_cluster: int = 2
    rank_metric: RankMetric = "sharpe"


@dataclass(frozen=True)
class UniverseConfig:
    """
    Configuration for time-varying universe handling.

    Use this when strategies/instruments have different active periods
    (e.g., due to new issuance or termination dates).

    Attributes
    ----------
    lookback : int, default 126
        Trailing window length used for metrics/correlation (e.g., 126 ~= 6 months).

    rebalance_freq : str, default "W-FRI"
        Rebalance calendar frequency (pandas offset alias).
        Examples: "W-FRI" (weekly Friday), "M" (monthly), "Q" (quarterly).

    require_active_today : bool, default True
        Only consider strategies that exist (have data) on the rebalance date.
        Set to False to include recently terminated strategies.
    """
    lookback: int = 126
    rebalance_freq: str = "W-FRI"
    require_active_today: bool = True
