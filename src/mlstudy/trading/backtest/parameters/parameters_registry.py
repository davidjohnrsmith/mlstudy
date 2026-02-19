"""Instance-based parameter direction registry.

Each strategy builds its own ``ParameterPreferenceRegistry`` from its
parameter enum.  The instance is stored on the ``RankingPlan`` so the
sweep ranker knows which parameters (and directions) are valid.

Usage::

    from mlstudy.trading.backtest.parameters.parameters_enum import MRParameter, PortfolioParameter

    mr_registry = ParameterPreferenceRegistry(MRParameter)
    port_registry = ParameterPreferenceRegistry(PortfolioParameter)

    mr_registry.direction("target_notional_ref")   # +1
    port_registry.direction("gross_dv01_cap")       # +1

    mr_registry.direction("gross_dv01_cap")         # ValueError — not an MR param
    port_registry.direction("target_notional_ref")  # ValueError — not a portfolio param
"""

from __future__ import annotations

from enum import Enum
from typing import Type


class ParameterPreferenceRegistry:
    """Direction registry scoped to a single strategy's parameters.

    +1 means higher is preferred, -1 means lower is preferred.
    """

    def __init__(self, param_enum: Type[Enum]) -> None:
        self._param_enum = param_enum
        self._directions: dict[str, int] = {
            p.key: p.direction for p in param_enum
        }

    def direction(self, name: str) -> int:
        """Return the preferred direction for *name*.

        Raises ``ValueError`` if *name* is not in this registry's enum.
        """
        try:
            return self._directions[name]
        except KeyError as e:
            raise ValueError(
                f"Unknown parameter {name!r}; choose from {sorted(self._directions)}"
            ) from e

    def is_registered(self, name: str) -> bool:
        """Return True if *name* is known to this registry."""
        return name in self._directions

    def override(self, name: str, direction: int) -> None:
        """Override direction for an already-registered parameter."""
        if name not in self._directions:
            raise ValueError(
                f"Unknown parameter {name!r}; choose from {sorted(self._directions)}"
            )
        if direction not in (+1, -1):
            raise ValueError(
                f"Direction must be +1 or -1, got {direction!r} for {name!r}"
            )
        self._directions[name] = int(direction)

    def override_multi(self, overrides: dict[str, int]) -> None:
        """Override directions for multiple parameters."""
        for name, direction in overrides.items():
            self.override(name, direction)

    def reset(self) -> None:
        """Reset all directions to defaults from the enum."""
        self._directions = {p.key: p.direction for p in self._param_enum}

    def registered(self) -> dict[str, int]:
        """Return a copy of all registered parameters and their directions."""
        return dict(self._directions)
