from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumLeg:
    """One momentum component using (lookback L, skip k) on the level series F_t."""
    lookback: int   # L
    skip: int = 0   # k
    weight: float = 1.0


@dataclass(frozen=True)
class MomentumConfig:
    # Momentum components (ensemble)
    legs: Tuple[MomentumLeg, ...] = (
        MomentumLeg(lookback=60, skip=2, weight=1.0),
    )

    # Vol scaling computed on daily changes of F_t
    vol_window: int = 20          # rolling stdev window on dF
    vol_min_periods: Optional[int] = None

    # Post-processing
    clip_score: float = 3.0       # clip z-score
    smooth_alpha: float = 0.2     # EMA smoothing on score (0=no smoothing)

    # Position rule
    hysteresis: float = 0.5       # threshold theta; 0 => no hysteresis
    allow_flat: bool = True       # if False, stays +/-1 once entered


def _validate_series(x: pd.Series) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError("residual_yield must be a pandas Series")
    if x.index is None:
        raise ValueError("residual_yield must have an index (dates)")
    x = x.astype(float).copy()
    x = x.sort_index()
    return x


def _rolling_vol(dF: pd.Series, window: int, min_periods: Optional[int]) -> pd.Series:
    if min_periods is None:
        min_periods = max(5, int(window * 0.6))
    vol = dF.rolling(window=window, min_periods=min_periods).std()
    return vol.replace(0.0, np.nan)


def _momentum_component(F: pd.Series, lookback: int, skip: int) -> pd.Series:
    # m_t = F_{t-skip} - F_{t-lookback-skip}
    return F.shift(skip) - F.shift(lookback + skip)


def _ema(x: pd.Series, alpha: float) -> pd.Series:
    if alpha <= 0.0:
        return x
    # span mapping: alpha = 2/(span+1)  => span = 2/alpha - 1
    span = max(2.0 / alpha - 1.0, 1.0)
    return x.ewm(span=span, adjust=False, min_periods=1).mean()


def _hysteresis_position(score: pd.Series, theta: float, allow_flat: bool) -> pd.Series:
    """
    Stateful hysteresis:
      enter long if score > +theta
      enter short if score < -theta
      else hold previous (or go flat if allow_flat and previous is 0)
    """
    pos = pd.Series(index=score.index, dtype=float)
    prev = 0.0

    for t, s in score.items():
        if np.isnan(s):
            pos.at[t] = prev
            continue

        if s > theta:
            prev = 1.0
        elif s < -theta:
            prev = -1.0
        else:
            if allow_flat:
                # keep prev as-is (including 0 if currently flat)
                prev = prev
            else:
                # if not allowing flat, force to remain in +/-1 once entered
                prev = prev if prev != 0.0 else (1.0 if s >= 0 else -1.0)

        pos.at[t] = prev

    return pos


def build_momentum_signals(
    residual_yield: pd.Series,
    cfg: MomentumConfig = MomentumConfig(),
) -> pd.DataFrame:
    """
    Build momentum signals from residual yield level series F_t (e.g., residual fly yield in bp).

    Returns DataFrame with:
      F, dF, vol, mom_raw (weighted), score (mom/vol), score_clip, score_smooth, position
    """
    F = _validate_series(residual_yield)

    dF = F.diff()

    vol = _rolling_vol(
        dF,
        window=cfg.vol_window,
        min_periods=cfg.vol_min_periods,
    )

    # Ensemble momentum
    mom_parts = []
    weights = []
    for leg in cfg.legs:
        if leg.lookback <= 0:
            raise ValueError(f"lookback must be > 0, got {leg.lookback}")
        if leg.skip < 0:
            raise ValueError(f"skip must be >= 0, got {leg.skip}")
        mom_parts.append(_momentum_component(F, leg.lookback, leg.skip))
        weights.append(float(leg.weight))

    weights = np.array(weights, dtype=float)
    if np.allclose(weights.sum(), 0.0):
        raise ValueError("Sum of leg weights must be non-zero")

    # Weighted sum (no need to normalize, but you can if you want)
    mom_raw = pd.concat(mom_parts, axis=1).mul(weights, axis=1).sum(axis=1)

    score = mom_raw / vol

    if cfg.clip_score is not None and cfg.clip_score > 0:
        score_clip = score.clip(-cfg.clip_score, cfg.clip_score)
    else:
        score_clip = score

    score_smooth = _ema(score_clip, cfg.smooth_alpha)

    if cfg.hysteresis is not None and cfg.hysteresis > 0:
        position = _hysteresis_position(score_smooth, cfg.hysteresis, cfg.allow_flat)
    else:
        # No hysteresis: simple sign (or continuous position = score_smooth)
        position = np.sign(score_smooth).replace(0.0, np.nan).ffill().fillna(0.0)

    out = pd.DataFrame(
        {
            "F": F,
            "dF": dF,
            "vol": vol,
            "mom_raw": mom_raw,
            "score": score,
            "score_clip": score_clip,
            "score_smooth": score_smooth,
            "position": position,
        }
    )
    return out


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # residual_yield: pd.Series indexed by date
    # residual_yield = ...

    cfg = MomentumConfig(
        legs=(
            MomentumLeg(lookback=20, skip=2, weight=0.5),
            MomentumLeg(lookback=60, skip=2, weight=0.3),
            MomentumLeg(lookback=120, skip=5, weight=0.2),
        ),
        vol_window=20,
        clip_score=3.0,
        smooth_alpha=0.2,
        hysteresis=0.5,
        allow_flat=True,
    )

    # signals = build_momentum_signals(residual_yield, cfg)
    # print(signals.tail())
