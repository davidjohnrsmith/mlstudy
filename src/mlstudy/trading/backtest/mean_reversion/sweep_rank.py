from __future__ import annotations

from dataclasses import dataclass

from .sweep_types import MetricsOnlyResult


class MetricPreferenceRegistry:
    """Direction registry for backtest metric names.

    +1 means higher is better, -1 means lower is better.
    """

    _DIRECTIONS: dict[str, int] = {
        # higher-is-better (+1)
        "total_pnl": +1,
        "final_equity": +1,
        "mean_daily_return": +1,
        "sharpe_ratio": +1,
        "sortino_ratio": +1,
        "max_drawdown": +1,  # <= 0; less negative = better
        "calmar_ratio": +1,
        "avg_holding_period": +1,
        "hit_rate": +1,
        "profit_factor": +1,
        "avg_win": +1,
        "avg_loss": +1,  # negative; less negative = better
        "win_loss_ratio": +1,
        "skewness": +1,
        "var_95": +1,  # negative; less negative = better
        "cvar_95": +1,
        "n_trades": +1,
        "pct_time_in_market": +1,
        # lower-is-better (-1)
        "std_daily_return": -1,
        "max_drawdown_duration": -1,
        "kurtosis": -1,
        "turnover_annual": -1,
    }

    @classmethod
    def direction(cls, name: str) -> int:
        try:
            return cls._DIRECTIONS[name]
        except KeyError:
            raise ValueError(
                f"Unknown metric {name!r}; choose from {sorted(cls._DIRECTIONS)}"
            )


class ParameterPreferenceRegistry:
    """Direction registry for MRBacktestConfig numeric fields.

    +1 means higher is preferred, -1 means lower is preferred.
    """

    _DIRECTIONS: dict[str, int] = {
        # higher-is-preferred (+1)
        "target_notional_ref": +1,
        "entry_z_threshold": +1,
        "expected_yield_pnl_bps_multiplier": +1,
        "entry_cost_premium_yield_bps": +1,
        "tp_cost_premium_yield_bps": +1,
        "sl_cost_premium_yield_bps": +1,
        "tp_quarantine_bars": +1,
        "sl_quarantine_bars": +1,
        "time_quarantine_bars": +1,
        "initial_capital": +1,
        # lower-is-preferred (-1)
        "take_profit_zscore_soft_threshold": -1,
        "take_profit_yield_change_soft_threshold": -1,
        "take_profit_yield_change_hard_threshold": -1,
        "stop_loss_yield_change_hard_threshold": -1,
        "max_holding_bars": -1,
        "max_levels_to_cross": -1,
        "size_haircut": -1,
    }

    @classmethod
    def direction(cls, name: str) -> int:
        try:
            return cls._DIRECTIONS[name]
        except KeyError:
            raise ValueError(
                f"Unknown parameter {name!r}; choose from {sorted(cls._DIRECTIONS)}"
            )


@dataclass(frozen=True)
class RankingPlan:
    primary_metrics: tuple[tuple[str, float], ...] = (("total_pnl", 1.0),)
    tie_metrics: tuple[tuple[str, float], ...] = ()
    primary_params: tuple[tuple[str, float], ...] = ()
    tie_params: tuple[tuple[str, float], ...] = ()


DEFAULT_RANKING_PLAN = RankingPlan()


def _average_ranks(values: list[float]) -> list[float]:
    """Compute 1-based average ranks (ties get average of their positions)."""
    n = len(values)
    if n == 0:
        return []

    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


def _stage_scores(
    results: list[MetricsOnlyResult],
    features: tuple[tuple[str, float], ...],
    source: str,
) -> list[float]:
    """Compute weighted-rank score for a stage.

    ``source`` is either ``"metric"`` or ``"param"``.
    """
    n = len(results)
    if n == 0:
        return []

    scores = [0.0] * n

    for name, weight in features:
        if source == "metric":
            direction = MetricPreferenceRegistry.direction(name)
            raw = [r.metrics_dict()[name] for r in results]
        else:
            direction = ParameterPreferenceRegistry.direction(name)
            raw = [float(getattr(r.scenario.cfg, name)) for r in results]

        oriented = [v * direction for v in raw]
        ranks = _average_ranks(oriented)

        if n <= 1:
            rank01 = [0.0] * n
        else:
            rank01 = [(r - 1) / (n - 1) for r in ranks]

        for i in range(n):
            scores[i] += weight * rank01[i]

    return scores


def rank_scenarios(
    results: list[MetricsOnlyResult],
    plan: RankingPlan | None = None,
) -> list[MetricsOnlyResult]:
    """Rank scenarios according to a multi-stage weighted-rank plan.

    Returns a new list sorted best-first.
    """
    if not results:
        return []

    if plan is None:
        plan = DEFAULT_RANKING_PLAN

    s1 = _stage_scores(results, plan.primary_metrics, "metric")
    s2 = _stage_scores(results, plan.tie_metrics, "metric")
    s3 = _stage_scores(results, plan.primary_params, "param")
    s4 = _stage_scores(results, plan.tie_params, "param")

    decorated = [
        (-s1[i], -s2[i], -s3[i], -s4[i], results[i].scenario_idx, results[i])
        for i in range(len(results))
    ]
    decorated.sort()

    return [item[-1] for item in decorated]
