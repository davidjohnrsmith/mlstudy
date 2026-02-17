from __future__ import annotations

from dataclasses import dataclass

from .sweep_types import MetricsOnlyResult
from ...metrics.metrics_registry import MetricPreferenceRegistry
from ...parameters.parameters_registry import ParameterPreferenceRegistry


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


