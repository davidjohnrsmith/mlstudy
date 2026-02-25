"""Strategy-agnostic sweep ranking.

Provides multi-stage weighted-rank scoring for sweep results.
The ``RankingPlan`` carries a ``param_registry`` so the ranker knows
each strategy's parameter directions without global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from .sweep_types import SweepResultLight
from ...metrics.metrics_registry import MetricPreferenceRegistry

if TYPE_CHECKING:
    from ...parameters.parameters_registry import ParameterPreferenceRegistry


@dataclass(frozen=True)
class RankingPlan:
    primary_metrics: tuple[tuple[str, float], ...] = (("total_pnl", 1.0),)
    tie_metrics: tuple[tuple[str, float], ...] = ()
    primary_params: tuple[tuple[str, float], ...] = ()
    tie_params: tuple[tuple[str, float], ...] = ()
    param_registry: ParameterPreferenceRegistry | None = field(
        default=None, repr=False, compare=False,
    )


class SweepRanker:
    @staticmethod
    def _average_ranks(values: list[float]) -> list[float]:
        """Compute 1-based average ranks (ties get average of their positions)."""
        n = len(values)
        if n == 0:
            return []

        indexed = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n

        i = 0
        while i < n:
            j = i + 1
            vi = values[indexed[i]]
            # Group equal values (NaN == NaN is False, so use math.isnan)
            if vi != vi:  # NaN check
                while j < n and values[indexed[j]] != values[indexed[j]]:
                    j += 1
            else:
                while j < n and values[indexed[j]] == vi:
                    j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j

        return ranks

    @staticmethod
    def _stage_scores(
        results: list[SweepResultLight],
        features: tuple[tuple[str, float], ...],
        source: str,
        param_registry: ParameterPreferenceRegistry | None = None,
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
                raw = [getattr(r.metrics, name) for r in results]
            else:
                if param_registry is None:
                    raise ValueError(
                        f"Cannot rank by parameter {name!r}: no param_registry "
                        f"on the RankingPlan.  Set param_registry when building "
                        f"the plan."
                    )
                direction = param_registry.direction(name)
                raw = [float(getattr(r.scenario.cfg, name)) for r in results]

            oriented = [v * direction for v in raw]
            ranks = SweepRanker._average_ranks(oriented)

            if n <= 1:
                rank01 = [0.0] * n
            else:
                rank01 = [(r - 1) / (n - 1) for r in ranks]

            for i in range(n):
                scores[i] += weight * rank01[i]

        return scores

    @staticmethod
    def rank_scenarios(
        results: list[SweepResultLight],
        plan: RankingPlan,
    ) -> list[SweepResultLight]:
        """Rank scenarios according to a multi-stage weighted-rank plan.

        Returns a new list sorted best-first.
        """
        if not results:
            return []

        reg = plan.param_registry
        s1 = SweepRanker._stage_scores(results, plan.primary_metrics, "metric", reg)
        s2 = SweepRanker._stage_scores(results, plan.tie_metrics, "metric", reg)
        s3 = SweepRanker._stage_scores(results, plan.primary_params, "param", reg)
        s4 = SweepRanker._stage_scores(results, plan.tie_params, "param", reg)

        decorated = [
            (-s1[i], -s2[i], -s3[i], -s4[i], results[i].scenario_idx, results[i])
            for i in range(len(results))
        ]
        decorated.sort()

        return [item[-1] for item in decorated]

    # ------------------------------------------------------------------
    # DataFrame ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _df_stage_scores(
        df: pd.DataFrame,
        features: tuple[tuple[str, float], ...],
        source: str,
        param_registry: ParameterPreferenceRegistry | None = None,
    ) -> list[float]:
        """Compute weighted-rank score for a stage using DataFrame columns."""
        n = len(df)
        if n == 0:
            return []

        scores = [0.0] * n

        for name, weight in features:
            if name not in df.columns:
                continue

            if source == "metric":
                direction = MetricPreferenceRegistry.direction(name)
            else:
                if param_registry is None:
                    raise ValueError(
                        f"Cannot rank by parameter {name!r}: no param_registry "
                        f"on the RankingPlan."
                    )
                direction = param_registry.direction(name)

            raw = df[name].astype(float).tolist()
            oriented = [v * direction for v in raw]
            ranks = SweepRanker._average_ranks(oriented)

            if n <= 1:
                rank01 = [0.0] * n
            else:
                rank01 = [(r - 1) / (n - 1) for r in ranks]

            for i in range(n):
                scores[i] += weight * rank01[i]

        return scores

    @staticmethod
    def rank_dataframe(
        df: pd.DataFrame,
        plan: RankingPlan,
    ) -> pd.DataFrame:
        """Rank DataFrame rows using the multi-stage weighted-rank plan.

        Returns a copy sorted best-first with a ``rank`` column (1-based).
        """
        if df.empty:
            out = df.copy()
            out.insert(0, "rank", pd.Series(dtype=int))
            return out

        reg = plan.param_registry
        s1 = SweepRanker._df_stage_scores(df, plan.primary_metrics, "metric", reg)
        s2 = SweepRanker._df_stage_scores(df, plan.tie_metrics, "metric", reg)
        s3 = SweepRanker._df_stage_scores(df, plan.primary_params, "param", reg)
        s4 = SweepRanker._df_stage_scores(df, plan.tie_params, "param", reg)

        sort_keys = [(-s1[i], -s2[i], -s3[i], -s4[i], i) for i in range(len(df))]
        order = sorted(range(len(df)), key=lambda i: sort_keys[i])

        out = df.iloc[order].reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out
