from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class FlySpec:
    """Specification for a 3-leg butterfly (fly) structure.

    This is a *declarative* config object used by construction and strategy layers.
    It does not contain data; it defines *how* to build a fly (legs + constraints).

    Conventions:
      - Leg order is always (front, belly, back).
      - Weights are applied as: fly = w_front*front + w_belly*belly + w_back*back
      - DV01-neutrality (if enabled) is defined as:
          w_front*dv01_front + w_belly*dv01_belly + w_back*dv01_back = 0

    Notes:
      - Use `normalize` to prevent trivial scaling ambiguity when calibrating weights.
      - If you calibrate weights via OLS/residual methods, the common residual form is:
          fly = belly - (c + beta_front*front + beta_back*back)
        which corresponds to weights (up to intercept handling):
          w_belly=+1, w_front=-beta_front, w_back=-beta_back
    """

    # Identity / metadata
    name: str = "fly"
    meta: Mapping[str, Any] = field(default_factory=dict)

    # Leg identifiers (instrument IDs, tickers, ISINs, etc.)
    # If your code uses LegId from structures/portfolio_types.py, you can swap these to that type.
    front_leg: Any = "front"
    belly_leg: Any = "belly"
    back_leg: Any = "back"

    # Default/static weights (used when method="fixed")
    # Typical equal-weight fly: (1, -2, 1)
    fixed_weights: Tuple[float, float, float] = (1.0, -2.0, 1.0)

    # Construction method
    #   - "fixed": use fixed_weights
    #   - "levels_residual": belly on (front, back) levels residual
    #   - "changes_residual": residual on differences (optionally integrated)
    #   - "optimize_stationarity": grid/optimizer for mean reversion
    method: str = "fixed"

    # Rolling estimation options (used by residual/optimization methods)
    window: Optional[int] = 252
    min_obs: int = 60
    rolling: bool = True

    # Regression options for residual methods
    add_intercept: bool = True
    ridge_lambda: Optional[float] = None
    integrate_changes: bool = True  # for changes_residual

    # Optimization options for stationarity optimization
    objective: str = "half_life"  # "half_life" | "ar1_b" | "adf_pvalue"
    grid_size: int = 21
    bounds_front: Tuple[float, float] = (-3.0, 3.0)
    bounds_back: Tuple[float, float] = (-3.0, 3.0)
    # Fix belly weight to set scale (common choices: +1.0 or -2.0)
    w_belly_fixed: float = 1.0

    # Risk constraints / overlays
    dv01_neutral: bool = False
    dv01_tolerance: float = 1e-6
    # If you "force" dv01 neutrality by adjusting one leg post-hoc
    dv01_adjust_leg: str = "back"  # "front" or "back"

    # Normalization to remove scale ambiguity when calibrating weights
    # Supported:
    #   - None: no normalization
    #   - "belly_fixed": enforce w_belly = w_belly_fixed (recommended)
    #   - "l1": scale so sum(|w|)=1
    #   - "l2": scale so sqrt(sum(w^2))=1
    normalize: Optional[str] = "belly_fixed"

    # Optional additional linear constraints (advanced)
    # Example: {"sum_weights": 0.0} means w_f + w_b + w_k = 0
    # These are declarative; your solver must decide how to enforce them.
    constraints: Mapping[str, float] = field(default_factory=dict)

    def leg_ids(self) -> Tuple[Any, Any, Any]:
        """Return leg identifiers in canonical order."""
        return (self.front_leg, self.belly_leg, self.back_leg)

    def validate(self) -> None:
        """Light validation to catch obvious config issues early."""
        if self.method not in {
            "fixed",
            "levels_residual",
            "changes_residual",
            "optimize_stationarity",
        }:
            raise ValueError(f"Unknown method: {self.method}")

        if self.window is not None and self.window <= 1:
            raise ValueError("window must be > 1 when provided")

        if self.min_obs < 2:
            raise ValueError("min_obs must be >= 2")

        if self.method == "fixed" and len(self.fixed_weights) != 3:
            raise ValueError("fixed_weights must be a 3-tuple")

        if self.normalize not in (None, "belly_fixed", "l1", "l2"):
            raise ValueError(f"Unsupported normalize: {self.normalize}")

        if self.dv01_adjust_leg not in ("front", "back"):
            raise ValueError("dv01_adjust_leg must be 'front' or 'back'")

        if self.grid_size < 3 or self.grid_size % 2 == 0:
            # odd grid is nice because it includes 0 at center when bounds symmetric
            raise ValueError("grid_size should be an odd integer >= 3")

        if self.bounds_front[0] >= self.bounds_front[1]:
            raise ValueError("bounds_front must be (min, max)")

        if self.bounds_back[0] >= self.bounds_back[1]:
            raise ValueError("bounds_back must be (min, max)")

    def normalize_weights(self, w: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply normalization rule (if any) to a weight triple."""
        wf, wb, wk = w

        if self.normalize is None:
            return w

        if self.normalize == "belly_fixed":
            if abs(wb) < 1e-12:
                # fall back: keep as-is to avoid blowup
                return w
            scale = self.w_belly_fixed / wb
            return (wf * scale, wb * scale, wk * scale)

        arr = np.array([wf, wb, wk], dtype=float)

        if self.normalize == "l1":
            denom = float(np.sum(np.abs(arr)))
            return (wf, wb, wk) if denom < 1e-12 else tuple((arr / denom).tolist())  # type: ignore[return-value]

        if self.normalize == "l2":
            denom = float(np.sqrt(np.sum(arr * arr)))
            return (wf, wb, wk) if denom < 1e-12 else tuple((arr / denom).tolist())  # type: ignore[return-value]

        # Should be unreachable due to validate()
        return w
