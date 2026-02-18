from __future__ import annotations

from typing import Union

from mlstudy.trading.backtest.metrics.metrics_enum import Metric

MetricLike = Union[Metric, str]


class MetricPreferenceRegistry:
    """Direction registry for backtest metric names.

    +1 means higher is better, -1 means lower is better.
    Defaults are derived from the ``Metric`` enum.
    """

    _DIRECTIONS: dict[str, int] = {
        **{m.key: m.direction for m in Metric},
    }

    @classmethod
    def direction(cls, name: str) -> int:
        try:
            return cls._DIRECTIONS[name]
        except KeyError:
            raise ValueError(
                f"Unknown metric {name!r}; choose from {sorted(cls._DIRECTIONS)}"
            )

    @classmethod
    def override(cls, name: MetricLike, direction: int) -> None:
        """Override directions for one or more metrics.

        Args:
            overrides: Mapping of metric name to direction (+1 or -1).

        Raises:
            ValueError: If a metric name is unknown or a direction is invalid.
        """
        name = cls._coerce(name)
        if name not in cls._DIRECTIONS:
            raise ValueError(
                f"Unknown metric {name!r}; choose from {sorted(cls._DIRECTIONS)}"
            )
        if direction not in (+1, -1):
            raise ValueError(
                f"Direction must be +1 or -1, got {direction!r} for {name!r}"
            )
        cls._DIRECTIONS.update({name: direction})

    @classmethod
    def override_multi(cls, overrides: dict[MetricLike, int]) -> None:
        """Override directions for one or more metrics.

        Args:
            overrides: Mapping of metric name to direction (+1 or -1).

        Raises:
            ValueError: If a metric name is unknown or a direction is invalid.
        """
        for name, direction in overrides.items():
            cls.override(name, direction)

    @classmethod
    def reset(cls) -> None:
        """Reset all directions to defaults from the ``Metric`` enum."""
        cls._DIRECTIONS = {
            **{m.key: m.direction for m in Metric},
            "final_equity": +1,
        }

    @staticmethod
    def _coerce(metric: MetricLike) -> str:
        if isinstance(metric, Metric):
            return metric.key
        if isinstance(metric, str):
            return Metric.from_key(metric).key  # metric key like "max_drawdown"
        raise TypeError(f"metric must be Metric or str, got {type(metric)}")
