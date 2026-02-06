from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, List

import numpy as np
import pandas as pd


# =========================
# Helpers: metrics
# =========================
def _clean_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s.astype(float)

def ann_sharpe(r: pd.Series, ann_factor: float = 252.0) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return np.nan
    return float((r.mean() / sd) * np.sqrt(ann_factor))

def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def worst_rolling_return(r: pd.Series, window: int = 21) -> float:
    r = r.dropna()
    if len(r) < window:
        return np.nan
    roll = (1.0 + r).rolling(window).apply(np.prod, raw=True) - 1.0
    return float(roll.min())

def nw_tstat_mean(r: pd.Series, lags: int = 5) -> float:
    """
    Lightweight Newey–West t-stat for mean return.
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 10:
        return np.nan
    mu = x.mean()
    u = x - mu
    gamma0 = (u @ u) / n
    var = gamma0
    L = min(lags, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gamma = (u[k:] @ u[:-k]) / n
        var += 2.0 * w * gamma
    se = np.sqrt(var / n)
    if se == 0:
        return np.nan
    return float(mu / se)


# =========================
# Helpers: overlap correlation/cov
# =========================
def pairwise_overlap_corr(df: pd.DataFrame, min_obs: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise correlation computed on overlapping (non-NaN) observations per pair.
    Returns corr matrix and overlap counts.
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

def cov_from_pairwise_corr(df: pd.DataFrame, corr: pd.DataFrame) -> np.ndarray:
    """
    Convert a correlation matrix into covariance using per-column std (computed on df.dropna()).
    NOTE: stds are computed on available data per column (can differ).
    """
    std = df.std(ddof=1).to_numpy(dtype=float)
    C = corr.to_numpy(dtype=float)
    cov = C * std[:, None] * std[None, :]
    return cov

def make_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a symmetric matrix to nearest PSD via eigenvalue clipping.
    """
    A = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


# =========================
# Redundancy: threshold clustering
# =========================
def threshold_clusters(corr: pd.DataFrame, threshold: float = 0.8) -> Dict[int, List[str]]:
    """
    Build clusters as connected components where edge exists if corr >= threshold.
    """
    names = list(corr.columns)
    C = corr.to_numpy(dtype=float)
    n = len(names)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(C[i, j]) and C[i, j] >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    seen = set()
    clusters: Dict[int, List[str]] = {}
    cid = 0
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
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


# =========================
# Config dataclasses
# =========================
CorrMethod = Literal["pearson"]  # placeholder; pairwise corr uses pandas corr
RankMetric = Literal["sharpe", "nw_tstat", "ann_mean"]

@dataclass(frozen=True)
class StandaloneGates:
    """
    Standalone gates applied per rebalance date on a trailing window.
    Set thresholds to None to disable a gate.
    """
    min_obs: int = 60                  # seasoning / minimum history in trailing window
    min_sharpe: Optional[float] = 0.3
    min_nw_tstat: Optional[float] = 1.0
    max_mdd: Optional[float] = -0.25
    max_worst_1m: Optional[float] = -0.12
    ann_factor: float = 252.0
    nw_lags: int = 5
    worst_1m_window: int = 21

@dataclass(frozen=True)
class RedundancyGates:
    enabled: bool = True
    corr_threshold: float = 0.8
    corr_min_overlap: int = 60
    keep_per_cluster: int = 2
    rank_metric: RankMetric = "sharpe"

@dataclass(frozen=True)
class BacktestUniverseConfig:
    """
    How to interpret non-synchronous fly backtests.
    """
    lookback: int = 126                # trailing window length used for metrics/corr (e.g. 6 months)
    rebalance_freq: str = "W-FRI"      # rebalance calendar (weekly Friday)
    require_active_today: bool = True  # only consider flies that exist on rebalance date


# =========================
# Panel builder
# =========================
def build_returns_panel(backtests: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Combine independently backtested instruments into one panel.
    IMPORTANT: outside each instrument's life, values should be NaN (missing).
    """
    series = {k: _clean_series(v) for k, v in backtests.items()}
    df = pd.concat(series, axis=1).sort_index()
    return df


# =========================
# Pre-selector that is issuance-aware
# =========================
class UniverseAwarePreSelector:
    """
    Pre-select instruments (flies) with non-synchronous histories.

    At each rebalance date t:
      1) Take trailing window [t-lookback+1, t]
      2) Filter to instruments that are active at t (optional) and have >= min_obs in window
      3) Apply standalone gates (Sharpe, MDD, etc.) computed on available window data
      4) Reduce redundancy via correlation clustering computed on pairwise overlaps
    """

    def __init__(
        self,
        returns_panel: pd.DataFrame,
        universe_cfg: BacktestUniverseConfig,
        standalone: StandaloneGates,
        redundancy: RedundancyGates,
    ):
        self.R = returns_panel.copy().sort_index()
        self.cfg = universe_cfg
        self.standalone = standalone
        self.redundancy = redundancy

    def _window(self, t: pd.Timestamp) -> pd.DataFrame:
        end_loc = self.R.index.searchsorted(t, side="right") - 1
        if end_loc < 0:
            return self.R.iloc[0:0]
        start_loc = max(0, end_loc - self.cfg.lookback + 1)
        return self.R.iloc[start_loc : end_loc + 1]

    def _eligible(self, W: pd.DataFrame, t: pd.Timestamp) -> List[str]:
        if W.empty:
            return []
        obs = W.notna().sum(axis=0)

        if self.cfg.require_active_today:
            active_today = W.loc[t].notna() if t in W.index else W.iloc[-1].notna()
        else:
            active_today = pd.Series(True, index=W.columns)

        elig = obs[(obs >= self.standalone.min_obs) & active_today].index.tolist()
        return elig

    def _metrics(self, W: pd.DataFrame, names: List[str]) -> pd.DataFrame:
        rows = []
        for n in names:
            r = W[n].dropna()
            m = {
                "obs": int(r.shape[0]),
                "sharpe": ann_sharpe(r, self.standalone.ann_factor),
                "nw_tstat": nw_tstat_mean(r, lags=self.standalone.nw_lags),
                "max_mdd": max_drawdown(r),
                "worst_1m": worst_rolling_return(r, window=self.standalone.worst_1m_window),
                "ann_mean": float(r.mean() * self.standalone.ann_factor) if r.shape[0] else np.nan,
            }
            rows.append((n, m))
        return pd.DataFrame({k: v for k, v in rows}).T

    def _apply_gates(self, M: pd.DataFrame) -> List[str]:
        keep = pd.Series(True, index=M.index)

        if self.standalone.min_sharpe is not None:
            keep &= M["sharpe"] >= self.standalone.min_sharpe
        if self.standalone.min_nw_tstat is not None:
            keep &= M["nw_tstat"] >= self.standalone.min_nw_tstat
        if self.standalone.max_mdd is not None:
            keep &= M["max_mdd"] >= self.standalone.max_mdd
        if self.standalone.max_worst_1m is not None:
            keep &= M["worst_1m"] >= self.standalone.max_worst_1m

        return M.index[keep].tolist()

    def select(self, t: pd.Timestamp) -> Tuple[List[str], pd.DataFrame, Optional[Dict[int, List[str]]]]:
        W = self._window(t)
        elig = self._eligible(W, t)
        if not elig:
            return [], pd.DataFrame(), None

        M = self._metrics(W, elig)
        passed = self._apply_gates(M)
        if not passed:
            return [], M, None

        if not self.redundancy.enabled or len(passed) <= 1:
            return passed, M, None

        Wp = W[passed]
        corr, nobs = pairwise_overlap_corr(Wp, min_obs=self.redundancy.corr_min_overlap)

        clusters = threshold_clusters(corr, threshold=self.redundancy.corr_threshold)

        # pick top K per cluster
        rank_col = self.redundancy.rank_metric
        selected: List[str] = []
        for _, names in clusters.items():
            sub = M.loc[names].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[rank_col])
            sub = sub.sort_values(rank_col, ascending=False)
            selected.extend(sub.index[: self.redundancy.keep_per_cluster].tolist())

        # keep original column order
        selected = [c for c in W.columns if c in set(selected)]
        return selected, M, clusters


# =========================
# Portfolio constructor (dynamic universe)
# =========================
def equal_weight_weights(names: List[str]) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    w = np.ones(len(names), dtype=float)
    w = w / np.sum(np.abs(w))
    return pd.Series(w, index=names, dtype=float)

def risk_parity_weights(cov: np.ndarray, names: List[str], max_iter: int = 3000, step: float = 0.05) -> pd.Series:
    """
    Simple ERC in x>=0 space; outputs gross-normalized weights.
    """
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    x = np.ones(n, dtype=float) / n
    target = 1.0 / n

    def port_vol(w: np.ndarray) -> float:
        v = float(w @ cov @ w)
        return float(np.sqrt(max(v, 0.0)))

    for _ in range(max_iter):
        w = x
        mw = cov @ w
        vol = port_vol(w)
        if vol <= 0:
            break
        rc = (w * mw) / vol
        rc_share = rc / (rc.sum() if rc.sum() != 0 else 1.0)
        err = rc_share - target
        if float(np.max(np.abs(err))) < 1e-6:
            break
        x = x * (1.0 - step * err)
        x = np.maximum(x, 1e-12)
        x /= np.sum(x)

    return pd.Series(x / np.sum(np.abs(x)), index=names, dtype=float)

class UniverseAwarePortfolioBacktester:
    """
    Builds a portfolio through time when instruments (flies) enter/exit.

    Process:
      - on each rebalance date t:
          * pre-select eligible flies using UniverseAwarePreSelector
          * estimate covariance on trailing window for selected flies (pairwise overlap corr -> cov -> PSD)
          * compute weights (equal-weight or risk parity)
      - hold weights until next rebalance
      - portfolio return each day is sum_i w_i(t_hold) * r_i(day), ignoring missing instruments
    """

    def __init__(
        self,
        returns_panel: pd.DataFrame,
        selector: UniverseAwarePreSelector,
        universe_cfg: BacktestUniverseConfig,
        cov_min_overlap: int = 60,
        allocation: Literal["equal", "risk_parity"] = "risk_parity",
        tc_per_turnover: float = 0.0,
    ):
        self.R = returns_panel.copy().sort_index()
        self.selector = selector
        self.cfg = universe_cfg
        self.cov_min_overlap = cov_min_overlap
        self.allocation = allocation
        self.tc_per_turnover = tc_per_turnover

    def _rebalance_dates(self) -> pd.DatetimeIndex:
        # choose rebal dates that exist in the return index (snap to last available <= calendar date)
        cal = pd.date_range(self.R.index.min(), self.R.index.max(), freq=self.cfg.rebalance_freq)
        # snap to actual trading dates in index
        snapped = []
        for t in cal:
            loc = self.R.index.searchsorted(t, side="right") - 1
            if loc >= 0:
                snapped.append(self.R.index[loc])
        return pd.DatetimeIndex(sorted(set(snapped)))

    def run(self) -> Tuple[pd.Series, pd.DataFrame]:
        rebal_dates = self._rebalance_dates()
        if len(rebal_dates) == 0:
            raise ValueError("No rebalance dates found within returns_panel index.")

        weights_hist: Dict[pd.Timestamp, pd.Series] = {}
        port = pd.Series(0.0, index=self.R.index, name="portfolio_return")

        prev_w = pd.Series(dtype=float)

        for k, t in enumerate(rebal_dates):
            selected, metrics, clusters = self.selector.select(t)

            if not selected:
                w = pd.Series(dtype=float)
            else:
                W = self.selector._window(t)[selected]  # trailing window for cov
                corr, nobs = pairwise_overlap_corr(W, min_obs=self.cov_min_overlap)
                # Fill missing correlations (insufficient overlap) with 0 to be conservative,
                # keep diagonal 1. Then PSD-project.
                corr_filled = corr.copy()
                arr = np.array(corr_filled.to_numpy(copy=False), copy=True)  # force writable
                np.fill_diagonal(arr, 1.0)
                corr_filled = corr_filled.copy()
                corr_filled.iloc[:, :] = arr
                corr_filled = corr_filled.fillna(0.0)

                cov = cov_from_pairwise_corr(W, corr_filled)
                cov = make_psd(cov, eps=1e-10)

                if self.allocation == "equal":
                    w = equal_weight_weights(selected)
                else:
                    w = risk_parity_weights(cov, selected)

            # transaction cost on rebalance (turnover in weights)
            if self.tc_per_turnover > 0 and not prev_w.empty:
                all_names = sorted(set(prev_w.index).union(set(w.index)))
                w_prev = prev_w.reindex(all_names).fillna(0.0)
                w_new = w.reindex(all_names).fillna(0.0)
                turnover = float((w_new - w_prev).abs().sum())
                # apply cost on rebalance day
                port.loc[t] -= self.tc_per_turnover * turnover

            weights_hist[t] = w
            prev_w = w

            # apply weights from t (inclusive) until next rebalance (exclusive)
            t_end = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else None
            slice_idx = self.R.loc[t:t_end].index if t_end is None else self.R.loc[t:t_end].index[:-1]

            if w.empty:
                continue

            # portfolio return: ignore missing instruments by treating NaN return as 0 only for that instrument *when not held*
            # Here: if instrument is held but NaN on a day, treat as 0 PnL (you can't trade it / it doesn't exist)
            # Alternative: force w to only include active instruments on t, so NaNs should be rare.
            subR = self.R.loc[slice_idx, w.index]
            port.loc[slice_idx] += (subR.fillna(0.0).to_numpy(dtype=float) @ w.to_numpy(dtype=float))

        weights_df = pd.DataFrame(weights_hist).T.sort_index()
        return port, weights_df


# =========================
# Example of how you'd use it
# =========================
if __name__ == "__main__":
    # backtests: each fly instrument has its own life; outside life it simply has no data
    # Replace with your actual fly return series (pd.Series).
    idx = pd.date_range("2024-01-01", periods=180, freq="B")
    rng = np.random.default_rng(0)

    # Toy illustration: NaNs emulate your issuance windows
    fly_123 = pd.Series(0.0001 + 0.01 * rng.standard_normal(len(idx)), index=idx)              # day1-180
    fly_236 = pd.Series(0.0001 + 0.01 * rng.standard_normal(len(idx)), index=idx)
    fly_236.iloc[60:] = np.nan  # day1-~day90 (roughly)
    fly_234 = pd.Series(np.nan, index=idx)
    fly_234.iloc[60:] = 0.0001 + 0.01 * rng.standard_normal(len(idx) - 60)                     # day91-180

    backtests = {
        "fly(1,2,3)": fly_123,
        "fly(2,3,6)": fly_236,
        "fly(2,3,4)": fly_234,
    }

    R = build_returns_panel(backtests)

    universe_cfg = BacktestUniverseConfig(lookback=63, rebalance_freq="W-FRI", require_active_today=True)

    standalone = StandaloneGates(
        min_obs=40,
        min_sharpe=None,       # example: turn off Sharpe gate initially
        min_nw_tstat=None,
        max_mdd=None,
        max_worst_1m=None,
        ann_factor=252,
    )

    redundancy = RedundancyGates(
        enabled=True,
        corr_threshold=0.8,
        corr_min_overlap=40,
        keep_per_cluster=2,
        rank_metric="sharpe",
    )

    selector = UniverseAwarePreSelector(R, universe_cfg, standalone, redundancy)

    bt = UniverseAwarePortfolioBacktester(
        returns_panel=R,
        selector=selector,
        universe_cfg=universe_cfg,
        cov_min_overlap=40,
        allocation="risk_parity",
        tc_per_turnover=0.0,
    )

    port_ret, w_hist = bt.run()
    print(port_ret.dropna().head())
    print(w_hist.tail())
