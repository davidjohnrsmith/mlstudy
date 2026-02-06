"""Unified strategy selector supporting static and time-varying universes.

This module consolidates StrategyPreSelector (static) and UniverseAwarePreSelector
(time-varying) into a unified API.
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

from .metrics import (
    annualized_return,
    annualized_vol,
    sharpe,
    sortino,
    max_drawdown,
    worst_rolling_return,
    newey_west_tstat,
    turnover_proxy,
    positive_year_fraction,
)
from .correlation import (
    corr_matrix,
    pairwise_overlap_corr,
    vol_scale,
)
from .clustering import correlation_clusters
from .gates import StandaloneGates, RedundancyGates, UniverseConfig


def _clean_series(s: pd.Series) -> pd.Series:
    """Clean a return series: dedupe, sort, convert to float."""
    s = s.copy()
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s.astype(float)


@dataclass(frozen=True)
class SelectorReport:
    """Report from strategy selection."""
    metrics: pd.DataFrame
    passed_names: List[str]
    clusters: Optional[Dict[int, List[str]]]
    selected_names: List[str]


def align_returns(
    returns: Dict[str, pd.Series],
    how: Literal["inner", "outer"] = "inner",
    min_obs: int = 50,
) -> pd.DataFrame:
    """
    Align strategy return series into a DataFrame [T x N].

    Parameters
    ----------
    returns : Dict[str, pd.Series]
        Mapping from strategy name to return series.
    how : Literal["inner", "outer"]
        'inner' keeps only overlapping dates.
        'outer' unions all dates (may introduce NaNs).
    min_obs : int
        Drop strategies with fewer than min_obs observations.

    Returns
    -------
    pd.DataFrame
        Aligned returns matrix.
    """
    clean = {}
    for name, s in returns.items():
        s2 = _clean_series(s).dropna()
        if len(s2) >= min_obs:
            clean[name] = s2

    if not clean:
        raise ValueError("No strategies left after cleaning/min_obs filter.")

    df = pd.concat(clean, axis=1, join=how).sort_index()
    return df


def build_returns_panel(backtests: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Combine independently backtested instruments into one panel.

    IMPORTANT: Outside each instrument's life, values should be NaN (missing).
    This is the preferred input format for time-varying universe selection.

    Parameters
    ----------
    backtests : Dict[str, pd.Series]
        Mapping from strategy name to return series.

    Returns
    -------
    pd.DataFrame
        Panel with NaN where instruments are inactive.
    """
    series = {k: _clean_series(v) for k, v in backtests.items()}
    df = pd.concat(series, axis=1).sort_index()
    return df


class StrategySelector:
    """
    Unified pre-selector for strategy return streams.

    Supports two modes:
    1. Static universe: All strategies share the same observation period.
       Use `run()` method.

    2. Time-varying universe: Strategies have different active periods.
       Use `select_at()` method for point-in-time selection.

    Workflow:
      1) Align & clean
      2) Compute standalone metrics
      3) Apply standalone gates
      4) Optionally reduce redundancy via correlation clustering

    Parameters
    ----------
    returns : Dict[str, pd.Series]
        Mapping from strategy name to return series.
    ann_factor : Optional[float]
        Annualization factor. If None, defaults to 252.

    Examples
    --------
    # Static universe
    selector = StrategySelector(returns, ann_factor=252)
    report = selector.run(
        standalone=StandaloneGates(min_sharpe=0.3),
        redundancy=RedundancyGates(corr_threshold=0.8),
    )

    # Time-varying universe
    panel = build_returns_panel(backtests)
    selector = StrategySelector.from_panel(panel, ann_factor=252)
    selected, metrics, clusters = selector.select_at(
        t=pd.Timestamp("2024-06-01"),
        universe_cfg=UniverseConfig(lookback=126),
        standalone=StandaloneGates(min_obs=60),
        redundancy=RedundancyGates(),
    )
    """

    def __init__(
        self,
        returns: Dict[str, pd.Series],
        ann_factor: Optional[float] = None,
    ):
        self.returns = dict(returns)
        self.ann_factor = ann_factor or 252.0
        self._panel: Optional[pd.DataFrame] = None

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, ann_factor: Optional[float] = None) -> "StrategySelector":
        """Create selector from a returns panel (for time-varying universes)."""
        returns = {col: panel[col] for col in panel.columns}
        selector = cls(returns, ann_factor=ann_factor)
        selector._panel = panel.copy().sort_index()
        return selector

    def _get_panel(self) -> pd.DataFrame:
        """Get or build the returns panel."""
        if self._panel is not None:
            return self._panel
        return build_returns_panel(self.returns)

    def compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute performance metrics for each strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Returns matrix [T x N].

        Returns
        -------
        pd.DataFrame
            Metrics for each strategy (rows are strategies).
        """
        ann = self.ann_factor

        rows = []
        for name in df.columns:
            r = df[name].dropna()
            m = {
                "ann_return": annualized_return(r, ann),
                "ann_vol": annualized_vol(r, ann),
                "sharpe": sharpe(r, ann),
                "sortino": sortino(r, ann, mar=0.0),
                "max_drawdown": max_drawdown(r),
                "worst_period": worst_rolling_return(r, window=21),
                "nw_tstat": newey_west_tstat(r, lags=5),
                "obs": int(r.shape[0]),
                "turnover_proxy": turnover_proxy(r),
                "pos_year_frac": positive_year_fraction(r),
            }
            rows.append((name, m))

        out = pd.DataFrame({k: v for k, v in rows}).T
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.sort_index()

    def _compute_metrics_with_gates(
        self,
        df: pd.DataFrame,
        names: List[str],
        gates: StandaloneGates,
    ) -> pd.DataFrame:
        """Compute metrics using gate parameters."""
        ann = gates.ann_factor

        rows = []
        for name in names:
            r = df[name].dropna()
            m = {
                "obs": int(r.shape[0]),
                "sharpe": sharpe(r, ann),
                "sortino": sortino(r, ann, mar=0.0),
                "nw_tstat": newey_west_tstat(r, lags=gates.nw_lags),
                "max_drawdown": max_drawdown(r),
                "worst_period": worst_rolling_return(r, window=gates.worst_period_window),
                "ann_return": annualized_return(r, ann),
                "pos_year_frac": positive_year_fraction(r),
            }
            rows.append((name, m))

        return pd.DataFrame({k: v for k, v in rows}).T

    def apply_standalone_gates(
        self,
        metrics: pd.DataFrame,
        gates: StandaloneGates,
    ) -> List[str]:
        """
        Apply standalone gates to filter strategies.

        Parameters
        ----------
        metrics : pd.DataFrame
            Metrics for each strategy.
        gates : StandaloneGates
            Screening criteria.

        Returns
        -------
        List[str]
            Names of strategies that pass all gates.
        """
        keep = pd.Series(True, index=metrics.index)

        keep &= metrics["obs"] >= gates.min_obs

        if gates.min_sharpe is not None:
            keep &= metrics["sharpe"] >= gates.min_sharpe

        if gates.min_nw_tstat is not None:
            keep &= metrics["nw_tstat"] >= gates.min_nw_tstat

        if gates.max_mdd is not None:
            keep &= metrics["max_drawdown"] >= gates.max_mdd

        if gates.max_worst_period is not None:
            keep &= metrics["worst_period"] >= gates.max_worst_period

        if gates.min_positive_year_frac is not None and "pos_year_frac" in metrics.columns:
            keep &= metrics["pos_year_frac"] >= gates.min_positive_year_frac

        return metrics.index[keep].tolist()

    def reduce_redundancy(
        self,
        df: pd.DataFrame,
        metrics: pd.DataFrame,
        redundancy: RedundancyGates,
        use_pairwise_overlap: bool = False,
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        """
        Reduce redundancy via correlation clustering.

        Parameters
        ----------
        df : pd.DataFrame
            Returns matrix for passed strategies.
        metrics : pd.DataFrame
            Metrics for passed strategies.
        redundancy : RedundancyGates
            Redundancy reduction config.
        use_pairwise_overlap : bool
            If True, use pairwise overlap correlation (for non-synchronous data).
            If False, use standard correlation after dropping NaN rows.

        Returns
        -------
        selected : List[str]
            Selected strategy names.
        clusters : Dict[int, List[str]]
            Cluster assignments.
        """
        if not redundancy.enabled:
            return df.columns.tolist(), {}

        # Compute correlation
        if use_pairwise_overlap:
            corr, _ = pairwise_overlap_corr(df, min_obs=redundancy.corr_min_overlap)
        else:
            scaled = vol_scale(df, target_vol=1.0)
            corr = corr_matrix(scaled, method=redundancy.corr_method)

        clusters = correlation_clusters(corr, threshold=redundancy.corr_threshold)

        # Ranking within cluster
        rank_col = redundancy.rank_metric
        if rank_col not in metrics.columns:
            raise ValueError(f"rank_metric '{rank_col}' not in metrics columns")

        selected: List[str] = []
        for _, names in clusters.items():
            sub = metrics.loc[names].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[rank_col])
            if sub.empty:
                continue
            sub = sub.sort_values(rank_col, ascending=False)
            selected.extend(sub.index[: redundancy.keep_per_cluster].tolist())

        # Preserve deterministic order
        selected = [n for n in df.columns if n in set(selected)]
        return selected, clusters

    # -------------------------
    # Static universe interface
    # -------------------------
    def run(
        self,
        how_align: Literal["inner", "outer"] = "inner",
        min_obs_clean: int = 50,
        standalone: StandaloneGates = StandaloneGates(),
        redundancy: RedundancyGates = RedundancyGates(),
    ) -> SelectorReport:
        """
        Run selection on a static universe.

        Parameters
        ----------
        how_align : Literal["inner", "outer"]
            How to align returns across strategies.
        min_obs_clean : int
            Minimum observations for initial alignment.
        standalone : StandaloneGates
            Standalone screening criteria.
        redundancy : RedundancyGates
            Redundancy reduction config.

        Returns
        -------
        SelectorReport
            Selection results including metrics, passed names, clusters, and final selection.
        """
        df = align_returns(self.returns, how=how_align, min_obs=min_obs_clean)

        metrics = self.compute_metrics(df)
        passed = self.apply_standalone_gates(metrics, standalone)

        df_passed = df[passed].dropna()
        metrics_passed = metrics.loc[passed]

        if redundancy.enabled and df_passed.shape[1] > 1:
            selected, clusters = self.reduce_redundancy(
                df_passed, metrics_passed, redundancy, use_pairwise_overlap=False
            )
        else:
            selected, clusters = passed, {}

        return SelectorReport(
            metrics=metrics,
            passed_names=passed,
            clusters=clusters if redundancy.enabled else None,
            selected_names=selected,
        )

    # -------------------------
    # Time-varying universe interface
    # -------------------------
    def _get_window(self, t: pd.Timestamp, lookback: int) -> pd.DataFrame:
        """Get trailing window ending at t."""
        R = self._get_panel()
        end_loc = R.index.searchsorted(t, side="right") - 1
        if end_loc < 0:
            return R.iloc[0:0]
        start_loc = max(0, end_loc - lookback + 1)
        return R.iloc[start_loc : end_loc + 1]

    def _get_eligible(
        self,
        W: pd.DataFrame,
        t: pd.Timestamp,
        min_obs: int,
        require_active_today: bool,
    ) -> List[str]:
        """Get eligible strategies for selection at time t."""
        if W.empty:
            return []
        obs = W.notna().sum(axis=0)

        if require_active_today:
            active_today = W.loc[t].notna() if t in W.index else W.iloc[-1].notna()
        else:
            active_today = pd.Series(True, index=W.columns)

        elig = obs[(obs >= min_obs) & active_today].index.tolist()
        return elig

    def select_at(
        self,
        t: pd.Timestamp,
        universe_cfg: UniverseConfig = UniverseConfig(),
        standalone: StandaloneGates = StandaloneGates(),
        redundancy: RedundancyGates = RedundancyGates(),
    ) -> Tuple[List[str], pd.DataFrame, Optional[Dict[int, List[str]]]]:
        """
        Select strategies at a specific point in time for time-varying universes.

        At rebalance date t:
          1) Take trailing window [t-lookback+1, t]
          2) Filter to instruments with sufficient history and optionally active at t
          3) Apply standalone gates computed on available window data
          4) Reduce redundancy via pairwise overlap correlation

        Parameters
        ----------
        t : pd.Timestamp
            Selection date.
        universe_cfg : UniverseConfig
            Time-varying universe config.
        standalone : StandaloneGates
            Standalone screening criteria.
        redundancy : RedundancyGates
            Redundancy reduction config.

        Returns
        -------
        selected : List[str]
            Selected strategy names.
        metrics : pd.DataFrame
            Metrics for eligible strategies.
        clusters : Optional[Dict[int, List[str]]]
            Cluster assignments if redundancy enabled.
        """
        W = self._get_window(t, universe_cfg.lookback)
        elig = self._get_eligible(W, t, standalone.min_obs, universe_cfg.require_active_today)
        if not elig:
            return [], pd.DataFrame(), None

        M = self._compute_metrics_with_gates(W, elig, standalone)
        passed = self.apply_standalone_gates(M, standalone)
        if not passed:
            return [], M, None

        if not redundancy.enabled or len(passed) <= 1:
            return passed, M, None

        Wp = W[passed]
        selected, clusters = self.reduce_redundancy(
            Wp, M.loc[passed], redundancy, use_pairwise_overlap=True
        )
        return selected, M, clusters


# Aliases for backward compatibility
StrategyPreSelector = StrategySelector
UniverseAwarePreSelector = StrategySelector
