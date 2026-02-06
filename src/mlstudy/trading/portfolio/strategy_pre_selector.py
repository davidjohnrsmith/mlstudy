from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Literal, Callable

import numpy as np
import pandas as pd


CorrMethod = Literal["pearson", "spearman"]
CovMethod = Literal["sample", "ewma", "diag_shrink", "ledoit_wolf"]


# -----------------------------
# Metrics helpers (generic)
# -----------------------------
def _ann_factor_from_index(idx: pd.Index, default: float = 252.0) -> float:
    # Best effort: if user provides daily business dates, default 252.
    # If you know your frequency (daily/weekly), pass ann_factor explicitly.
    return default


def annualized_return(r: pd.Series, ann_factor: float = 252.0) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * ann_factor)


def annualized_vol(r: pd.Series, ann_factor: float = 252.0) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def sharpe(r: pd.Series, ann_factor: float = 252.0) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float((r.mean() / sd) * np.sqrt(ann_factor))


def max_drawdown(r: pd.Series) -> float:
    """
    Max drawdown computed on a cumulative curve from returns.
    Returns are assumed to be arithmetic (not log). Output is negative (e.g. -0.15).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def worst_period_return(r: pd.Series, window: int = 21) -> float:
    """
    Worst rolling cumulative return over 'window' periods, arithmetic compounding.
    """
    r = r.dropna()
    if len(r) < window:
        return np.nan
    roll = (1.0 + r).rolling(window).apply(np.prod, raw=True) - 1.0
    return float(roll.min())


def downside_deviation(r: pd.Series, ann_factor: float = 252.0, mar: float = 0.0) -> float:
    """
    Downside deviation (Sortino denom), MAR default 0.
    """
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    downside = np.minimum(r - mar, 0.0)
    dd = downside.std(ddof=1)
    return float(dd * np.sqrt(ann_factor))


def sortino(r: pd.Series, ann_factor: float = 252.0, mar: float = 0.0) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    dd = downside_deviation(r, ann_factor=ann_factor, mar=mar)
    if dd == 0 or not np.isfinite(dd):
        return np.nan
    return float((r.mean() - mar) / dd * np.sqrt(ann_factor))


def turnover_proxy(r: pd.Series) -> float:
    """
    Generic turnover proxy for return streams when you don't have trades:
    sum(abs(delta position)) isn't available, so we use roughness of returns.
    If you *do* have turnover, override this with your own metric.
    """
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.diff().abs().mean())


def newey_west_tstat_mean(r: pd.Series, lags: int = 5) -> float:
    """
    Newey-West t-stat for mean of returns (robust to autocorr/heterosk).
    Lightweight implementation; for production consider statsmodels.
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return np.nan
    mu = x.mean()
    u = x - mu

    # gamma_0
    gamma0 = (u @ u) / n
    var = gamma0

    # Bartlett weights
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma = (u[k:] @ u[:-k]) / n
        var += 2.0 * w * gamma

    se = np.sqrt(var / n)
    if se == 0:
        return np.nan
    return float(mu / se)


# -----------------------------
# Data alignment
# -----------------------------
def align_strategies(
    returns: Dict[str, pd.Series],
    how: Literal["inner", "outer"] = "inner",
    min_obs: int = 50,
) -> pd.DataFrame:
    """
    Align strategy return series into a DataFrame [T x N].
    Drops strategies with too few observations post-cleaning.
    """
    clean = {}
    for name, s in returns.items():
        s2 = s.copy()
        s2 = s2[~s2.index.duplicated(keep="last")]
        s2 = s2.sort_index().dropna()
        if len(s2) >= min_obs:
            clean[name] = s2

    if not clean:
        raise ValueError("No strategies left after cleaning/min_obs filter.")

    df = pd.concat(clean, axis=1, join=how).sort_index()
    return df


# -----------------------------
# Correlation, clustering
# -----------------------------
def corr_matrix(df: pd.DataFrame, method: CorrMethod = "pearson") -> pd.DataFrame:
    d = df.dropna()
    if d.shape[0] < 2:
        raise ValueError("Not enough overlapping data to compute correlation.")
    return d.corr(method=method)


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


def hierarchical_cluster(
    corr: pd.DataFrame,
    corr_threshold: float = 0.8,
) -> Dict[int, List[str]]:
    """
    Simple graph-based clustering using a correlation threshold.
    (Avoids scipy dependency. If you want dendrogram clustering, add scipy.)

    Two strategies are linked if corr >= threshold, clusters are connected components.
    """
    names = list(corr.columns)
    C = corr.to_numpy(dtype=float)
    n = len(names)

    # adjacency list
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(C[i, j]) and C[i, j] >= corr_threshold:
                adj[i].append(j)
                adj[j].append(i)

    # connected components
    seen = set()
    clusters: Dict[int, List[str]] = {}
    cid = 0
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        comp = []
        seen.add(i)
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


# -----------------------------
# Selector configuration
# -----------------------------
@dataclass(frozen=True)
class StandaloneGates:
    """
    Standalone (single-strategy) screening criteria.

    This dataclass defines conservative "quality gates" applied to each strategy
    independently, before any portfolio construction or redundancy reduction.
    The intent is to filter out strategies that are structurally weak, fragile,
    or statistically unsupported (e.g., poor risk-adjusted returns, excessive
    tail risk, insufficient history), rather than to aggressively “pick winners”.

    Any threshold can be disabled by setting it to None (except min_history_obs).

    Attributes
    ----------
    min_ann_sharpe : Optional[float], default 0.3
        Minimum required annualized Sharpe ratio for the strategy's return stream.
        Computed as:
            mean(returns) / std(returns) * sqrt(ann_factor)
        Use this as a broad quality screen, not a fine ranking tool.

        Notes:
        - If you are evaluating returns net of transaction costs, apply costs
          before computing Sharpe.
        - For noisier strategies or shorter samples, consider lowering this
          threshold and relying more on stability/t-stat gates.

    min_nw_tstat : Optional[float], default 1.0
        Minimum required Newey–West t-statistic of the mean return.

        This screens for statistical support of a non-zero mean in the presence
        of autocorrelation and heteroskedasticity in returns (common in trading
        strategies).

        Practical guidance:
        - Values around 1.0 are a light filter; 2.0 is stricter.
        - Newey–West lag choice matters; ensure it matches the strategy's
          return autocorrelation horizon (e.g., 5 lags for daily is a common start).

    max_mdd : Optional[float], default -0.25
        Maximum allowed max drawdown (MDD), expressed as a negative number.

        The strategy passes if:
            max_drawdown >= max_mdd

        Example:
        - max_mdd = -0.25 means the strategy must not experience drawdowns
          worse than -25% over the backtest window.

        Notes:
        - MDD depends heavily on sample length and path; it is best used as a
          coarse "tail sanity" gate rather than an optimization target.

    max_worst_1m : Optional[float], default -0.12
        Maximum allowed worst rolling "1-month" return, where 1 month is
        approximated as 21 trading days (daily data).

        The strategy passes if:
            worst_21d_return >= max_worst_1m

        Example:
        - max_worst_1m = -0.12 means the worst 21-day cumulative return must be
          no worse than -12%.

        Notes:
        - This is a more localized tail-risk check than MDD and often better
          aligned with risk limits / stop-out policies.

    min_history_obs : int, default 252
        Minimum number of observations required for the strategy to be considered.

        Example:
        - With daily data, 252 observations is approximately one trading year.

        Notes:
        - This is not optional in this implementation (it guards covariance/correlation
          estimation stability and reduces selection bias from short samples).

    min_positive_year_frac : Optional[float], default 0.5
        Minimum fraction of calendar years with positive cumulative return.

        Computation:
        - Group returns by calendar year and compute each year's cumulative return.
        - pos_year_frac = (# years with return > 0) / (total years)

        Example:
        - 0.5 means at least half of years must be positive.

        Notes:
        - This is a robustness/regime-stability proxy.
        - For shorter histories (e.g., < 3 years), this metric becomes noisy; consider
          disabling it or switching to quarters/months.
    """
    min_ann_sharpe: Optional[float] = 0.3
    min_nw_tstat: Optional[float] = 1.0
    max_mdd: Optional[float] = -0.25
    max_worst_1m: Optional[float] = -0.12
    min_history_obs: int = 252
    min_positive_year_frac: Optional[float] = 0.5


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

        Supported options:
        - "pearson": linear correlation (common default for returns).
        - "spearman": rank correlation (more robust to outliers and non-linear monotonicity).

        Guidance:
        - Use "pearson" for relatively well-behaved return streams.
        - Use "spearman" if you suspect heavy tails, outliers, or non-linear co-movement.

    corr_threshold : float, default 0.8
        Correlation threshold that defines "linked" strategies in the clustering step.

        Interpretation:
        - Two strategies are considered connected if corr(strategy_i, strategy_j)
          >= corr_threshold.
        - Clusters are formed as connected components of this threshold graph.

        Guidance:
        - 0.7–0.85 is typical for redundancy reduction.
        - Higher thresholds keep more strategies; lower thresholds collapse more
          of the universe into fewer clusters.

        Notes:
        - Correlations are typically computed after vol-scaling the strategies so
          that scale differences do not dominate similarity.

    keep_per_cluster : int, default 2
        Number of strategies to keep in each correlation cluster.

        - 1 means "keep only the single best" strategy per cluster (aggressive pruning).
        - 2–4 often works well when clusters are large or when you want some internal
          variety within a theme.

        Guidance:
        - For small universes (e.g., 10–30 strategies), set higher (2–3) or disable
          redundancy filtering and rely on the portfolio optimizer.
        - For large universes (100+), 1–2 is common to reduce noise and stabilize covariance.

    rank_metric : Literal["sharpe", "sortino", "ann_return"], default "sharpe"
        Metric used to rank strategies within each cluster to decide which ones to keep.

        Options:
        - "sharpe": prefer higher risk-adjusted returns.
        - "sortino": prefer higher downside-risk-adjusted returns.
        - "ann_return": prefer higher raw mean returns (least robust, but sometimes useful).

        Guidance:
        - Prefer "sharpe" or "sortino" unless you have a strong reason to prioritize
          raw return (and you have tight risk controls elsewhere).
        - If you compute metrics net of costs, ranking will reflect implementability.
    """
    enabled: bool = True
    corr_method: CorrMethod = "pearson"
    corr_threshold: float = 0.8
    keep_per_cluster: int = 2
    rank_metric: Literal["sharpe", "sortino", "ann_return"] = "sharpe"


@dataclass(frozen=True)
class SelectorReport:
    metrics: pd.DataFrame
    passed_names: List[str]
    clusters: Optional[Dict[int, List[str]]]
    selected_names: List[str]


class StrategyPreSelector:
    """
    Generic pre-selector for strategy return streams.

    Workflow:
      1) Align & clean
      2) Compute standalone metrics
      3) Apply standalone gates
      4) Optionally reduce redundancy via correlation clustering (keep best K per cluster)
    """

    def __init__(
        self,
        returns: Dict[str, pd.Series],
        ann_factor: Optional[float] = None,
    ):
        self.returns = dict(returns)
        self.ann_factor = ann_factor  # if None, default 252 later

    def compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        ann = self.ann_factor or _ann_factor_from_index(df.index, default=252.0)

        rows = []
        for name in df.columns:
            r = df[name].dropna()
            m = {
                "ann_return": annualized_return(r, ann),
                "ann_vol": annualized_vol(r, ann),
                "sharpe": sharpe(r, ann),
                "sortino": sortino(r, ann, mar=0.0),
                "max_drawdown": max_drawdown(r),
                "worst_21d": worst_period_return(r, window=21),
                "nw_tstat_mean": newey_west_tstat_mean(r, lags=5),
                "obs": int(r.shape[0]),
                "turnover_proxy": turnover_proxy(r),
            }

            # robustness: fraction of positive calendar years
            by_year = (1.0 + r).groupby(r.index.year).prod() - 1.0
            if by_year.shape[0] >= 1:
                m["pos_year_frac"] = float((by_year > 0).mean())
            else:
                m["pos_year_frac"] = np.nan

            rows.append((name, m))

        out = pd.DataFrame({k: v for k, v in rows}).T
        # Ensure numeric
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.sort_index()

    def apply_standalone_gates(self, metrics: pd.DataFrame, gates: StandaloneGates) -> List[str]:
        keep = pd.Series(True, index=metrics.index)

        keep &= metrics["obs"] >= gates.min_history_obs

        if gates.min_ann_sharpe is not None:
            keep &= metrics["sharpe"] >= gates.min_ann_sharpe

        if gates.min_nw_tstat is not None:
            keep &= metrics["nw_tstat_mean"] >= gates.min_nw_tstat

        if gates.max_mdd is not None:
            # MDD is negative; require it to be not "too negative"
            keep &= metrics["max_drawdown"] >= gates.max_mdd

        if gates.max_worst_1m is not None:
            keep &= metrics["worst_21d"] >= gates.max_worst_1m

        if gates.min_positive_year_frac is not None:
            keep &= metrics["pos_year_frac"] >= gates.min_positive_year_frac

        return metrics.index[keep].tolist()

    def reduce_redundancy(
        self,
        df: pd.DataFrame,
        metrics: pd.DataFrame,
        redundancy: RedundancyGates,
    ) -> Tuple[List[str], Dict[int, List[str]]]:
        if not redundancy.enabled:
            return df.columns.tolist(), {}

        # Vol-scale before correlation
        scaled = vol_scale(df, target_vol=1.0)
        corr = corr_matrix(scaled, method=redundancy.corr_method)
        clusters = hierarchical_cluster(corr, corr_threshold=redundancy.corr_threshold)

        # ranking within cluster
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

        # preserve deterministic order
        selected = [n for n in df.columns if n in set(selected)]
        return selected, clusters

    def run(
        self,
        how_align: Literal["inner", "outer"] = "inner",
        min_obs_clean: int = 50,
        standalone: StandaloneGates = StandaloneGates(),
        redundancy: RedundancyGates = RedundancyGates(),
    ) -> SelectorReport:
        df = align_strategies(self.returns, how=how_align, min_obs=min_obs_clean)

        metrics = self.compute_metrics(df)
        passed = self.apply_standalone_gates(metrics, standalone)

        df_passed = df[passed].dropna()
        metrics_passed = metrics.loc[passed]

        if redundancy.enabled and df_passed.shape[1] > 1:
            selected, clusters = self.reduce_redundancy(df_passed, metrics_passed, redundancy)
        else:
            selected, clusters = passed, {}

        return SelectorReport(
            metrics=metrics,
            passed_names=passed,
            clusters=clusters if redundancy.enabled else None,
            selected_names=selected,
        )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    idx = pd.date_range("2021-01-01", periods=900, freq="B")
    rng = np.random.default_rng(1)

    returns = {
        "strat_A": pd.Series(0.0002 + 0.010 * rng.standard_normal(len(idx)), index=idx),
        "strat_B": pd.Series(0.0001 + 0.011 * rng.standard_normal(len(idx)), index=idx),
        "strat_C": pd.Series(0.00015 + 0.009 * rng.standard_normal(len(idx)), index=idx),
    }
    # create a highly correlated copy
    returns["strat_A2"] = returns["strat_A"] * 0.9 + 0.001 * rng.standard_normal(len(idx))

    selector = StrategyPreSelector(returns, ann_factor=252)
    report = selector.run(
        standalone=StandaloneGates(
            min_ann_sharpe=0.2,
            min_nw_tstat=0.5,
            max_mdd=-0.35,
            max_worst_1m=-0.20,
            min_history_obs=252,
            min_positive_year_frac=0.4,
        ),
        redundancy=RedundancyGates(
            enabled=True,
            corr_threshold=0.8,
            keep_per_cluster=2,
            rank_metric="sharpe",
        ),
    )

    print("Passed standalone:", report.passed_names)
    print("Selected after redundancy:", report.selected_names)
    print("\nMetrics (top by Sharpe):")
    print(report.metrics.sort_values("sharpe", ascending=False).head(10))
    if report.clusters is not None:
        print("\nClusters:", report.clusters)
