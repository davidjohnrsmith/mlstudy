from __future__ import annotations

from typing import Union

from mlstudy.trading.backtest.parameters.parameters_enum import Parameter

ParameterLike = Union[Parameter, str]


class ParameterPreferenceRegistry:
    """Direction registry for MRBacktestConfig numeric fields.

    +1 means higher is preferred, -1 means lower is preferred.
    Defaults are derived from the ``Parameter`` enum.
    """

    _DIRECTIONS: dict[str, int] = {p.key: p.direction for p in Parameter}

    @classmethod
    def direction(cls, name: str) -> int:
        try:
            d = cls._DIRECTIONS[name]
        except KeyError as e:
            raise ValueError(
                f"Unknown parameter {name!r}; choose from {sorted(cls._DIRECTIONS)}"
            ) from e
        if d not in (+1, -1):
            raise ValueError(f"Invalid direction {d} for parameter {name!r}")
        return d

    @classmethod
    def override(cls, name: ParameterLike, direction: int) -> None:
        """Override direction for a single parameter.

        Args:
            name: Parameter enum member or parameter key string (e.g. "max_holding_bars").
            direction: +1 (higher preferred) or -1 (lower preferred).
        """
        key = cls._coerce(name)
        if key not in cls._DIRECTIONS:
            raise ValueError(
                f"Unknown parameter {key!r}; choose from {sorted(cls._DIRECTIONS)}"
            )
        if direction not in (+1, -1):
            raise ValueError(
                f"Direction must be +1 or -1, got {direction!r} for {key!r}"
            )
        cls._DIRECTIONS[key] = int(direction)

    @classmethod
    def override_multi(cls, overrides: dict[ParameterLike, int]) -> None:
        """Override directions for multiple parameters."""
        for name, direction in overrides.items():
            cls.override(name, direction)

    @classmethod
    def reset(cls) -> None:
        """Reset all directions to defaults from the ``Parameter`` enum."""
        cls._DIRECTIONS = {p.key: p.direction for p in Parameter}

    @staticmethod
    def _coerce(param: ParameterLike) -> str:
        if isinstance(param, Parameter):
            return param.key
        if isinstance(param, str):
            return Parameter.from_key(param).key
        raise TypeError(f"param must be Parameter or str, got {type(param)}")
