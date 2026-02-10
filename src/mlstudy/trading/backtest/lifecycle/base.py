"""
backtest/lifecycle/base.py

Lifecycle layer: strategy-specific position management (entry/exit/stop/cooldown)
that produces *targets* (or orders) for the execution engine.

This layer is OPTIONAL:
- If you already have targets/orders upstream, you can skip lifecycle completely.
- If you want the engine to manage stateful rules (MR hysteresis, cooldown, time stops),
  use a PositionManager.

Conventions
- Portfolio/traded objects dimension: M
- Output: target_positions[t, m] (portfolio units)
- Engine later maps portfolio -> legs via W if mapping enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Any

import numpy as np


@dataclass
class PMState:
    """
    Generic per-portfolio state used by lifecycle managers.
    Managers may ignore fields they don't use.

    Shapes:
      pos: (M,) current portfolio position (target held from previous step)
      holding_bars: (M,) bars since entry
      cooldown_bars: (M,) remaining cooldown; >0 means cannot enter new trades
      entry_ref: (M,) optional reference level at entry (e.g., entry z or entry price)
      aux: optional dict for manager-specific extra arrays
    """
    pos: np.ndarray
    holding_bars: np.ndarray
    cooldown_bars: np.ndarray
    entry_ref: np.ndarray
    aux: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class StepOutput:
    """
    Output of one lifecycle step.
    - target_pos: (M,) desired portfolio holdings after decision at bar t
    - reason: (M,) int reason codes (manager-specific enum)
    """
    target_pos: np.ndarray
    reason: np.ndarray


class PositionManager(Protocol):
    """
    A PositionManager produces target positions each bar based on:
    - signals you supply (precomputed arrays)
    - its internal state (holding time, cooldown, entry reference, etc.)

    It does NOT execute trades or compute PnL.
    """

    def reset(self, M: int, *, initial_pos: Optional[np.ndarray] = None) -> PMState: ...

    def step(self, t: int, state: PMState) -> StepOutput: ...
