from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class MomentumSignalConfig:
    # Momentum score: m_t = F_{t-skip} - F_{t-lookback-skip}, score = m_t / vol(dF)
    lookback: int = 60
    skip: int = 2

    vol_window: int = 20
    vol_min_periods: Optional[int] = None

    # Optional smoothing on score (EWMA); set 0 to disable
    smooth_alpha: float = 0.2

    # Hysteresis thresholds (time to enter / time to exit)
    enter_th: float = 0.8   # theta_in
    exit_th: float = 0.2    # theta_out  (must be < enter_th)

    # Confirmation: require N consecutive days beyond enter_th to enter
    confirm_days: int = 2

    # Position sizing mode
    # - "discrete": positions in {-1, 0, +1}
    # - "continuous": position ~ score / target, clipped
    mode: str = "discrete"
    target_score_for_full: float = 1.5  # only for continuous
    max_pos: float = 1.0                # leverage cap (both modes)

    # If your trade PnL is negatively correlated with dF, set -1.0 to flip
    direction: float = 1.0


# =============================================================================
# Helpers
# =============================================================================

def _validate_fly(F: pd.Series) -> pd.Series:
    if not isinstance(F, pd.Series):
        raise TypeError("fly must be a pandas Series")
    F = F.astype(float).sort_index()
    return F


def _rolling_vol(x: pd.Series, window: int, min_periods: Optional[int]) -> pd.Series:
    if min_periods is None:
        min_periods = max(5, int(window * 0.6))
    vol = x.rolling(window=window, min_periods=min_periods).std()
    return vol.replace(0.0, np.nan)


def _ema(x: pd.Series, alpha: float) -> pd.Series:
    if alpha <= 0:
        return x
    span = max(2.0 / alpha - 1.0, 1.0)  # alpha <-> span mapping
    return x.ewm(span=span, adjust=False, min_periods=1).mean()


def _consecutive_true(x: pd.Series, n: int) -> pd.Series:
    """True if last n values are all True (rolling)."""
    if n <= 1:
        return x.fillna(False)
    return x.fillna(False).rolling(n).apply(lambda a: 1.0 if np.all(a) else 0.0, raw=True).astype(bool)


# =============================================================================
# Core
# =============================================================================

def build_fly_momentum_signal(
    fly: pd.Series,
    cfg: MomentumSignalConfig = MomentumSignalConfig(),
) -> pd.DataFrame:
    """
    Input:
      fly: pd.Series of fly level (e.g., residual fly yield in bp), indexed by date.

    Output columns:
      F, dF, vol, mom, score_raw, score, enter_long, enter_short, exit_long, exit_short, position, action
    """
    F = _validate_fly(fly)

    if cfg.exit_th >= cfg.enter_th:
        raise ValueError("exit_th must be < enter_th for hysteresis")

    dF = F.diff()
    vol = _rolling_vol(dF, window=cfg.vol_window, min_periods=cfg.vol_min_periods)

    # Momentum: m_t = F_{t-skip} - F_{t-lookback-skip}
    mom = F.shift(cfg.skip) - F.shift(cfg.lookback + cfg.skip)

    score_raw = (mom / vol) * float(cfg.direction)
    score = _ema(score_raw, cfg.smooth_alpha)

    # Entry conditions (+ confirmation)
    long_trigger = score > cfg.enter_th
    short_trigger = score < -cfg.enter_th

    enter_long = _consecutive_true(long_trigger, cfg.confirm_days)
    enter_short = _consecutive_true(short_trigger, cfg.confirm_days)

    # Exit conditions (hysteresis)
    exit_long = score < cfg.exit_th
    exit_short = score > -cfg.exit_th

    # Stateful position construction + action labels
    idx = score.index
    pos = pd.Series(index=idx, dtype=float)
    action = pd.Series(index=idx, dtype=object)

    prev = 0.0
    for t in idx:
        s = score.at[t]
        if np.isnan(s):
            pos.at[t] = prev
            action.at[t] = "HOLD"
            continue

        a = "HOLD"
        new = prev

        if cfg.mode == "discrete":
            # Flip logic: flip only on confirmed opposite entry
            if prev <= 0 and enter_long.at[t]:
                new = +cfg.max_pos
                a = "BUY"
            elif prev >= 0 and enter_short.at[t]:
                new = -cfg.max_pos
                a = "SELL"
            else:
                # Exit logic only applies if in position and no flip
                if prev > 0 and exit_long.at[t]:
                    new = 0.0
                    a = "EXIT_LONG"
                elif prev < 0 and exit_short.at[t]:
                    new = 0.0
                    a = "EXIT_SHORT"

        elif cfg.mode == "continuous":
            # Continuous sizing: go towards score/target, but with dead-zone/hysteresis
            # Use entry gate: only take risk if |score| >= exit_th, and increase when >= enter_th
            if abs(s) < cfg.exit_th:
                new = 0.0
                a = "EXIT" if prev != 0 else "HOLD"
            else:
                desired = np.clip(s / cfg.target_score_for_full, -1.0, 1.0) * cfg.max_pos
                # Optional: if inside (exit_th, enter_th), hold prev to reduce churn
                if abs(s) < cfg.enter_th:
                    new = prev
                    a = "HOLD"
                else:
                    new = desired
                    a = "REBAL" if prev != 0 else ("BUY" if new > 0 else "SELL")
        else:
            raise ValueError("mode must be 'discrete' or 'continuous'")

        pos.at[t] = new
        action.at[t] = a
        prev = new

    out = pd.DataFrame(
        {
            "F": F,
            "dF": dF,
            "vol": vol,
            "mom": mom,
            "score_raw": score_raw,
            "score": score,
            "enter_long": enter_long,
            "enter_short": enter_short,
            "exit_long": exit_long,
            "exit_short": exit_short,
            "position": pos,
            "action": action,
        }
    )
    return out


# =============================================================================
# Example
# =============================================================================
if __name__ == "__main__":
    # fly = pd.Series(...)  # residual fly yield series in bp
    cfg = MomentumSignalConfig(
        lookback=60,
        skip=2,
        vol_window=20,
        smooth_alpha=0.2,
        enter_th=0.8,
        exit_th=0.2,
        confirm_days=2,
        mode="discrete",
        max_pos=1.0,
        direction=1.0,  # set -1 if your PnL sign vs dF is inverted
    )
    # sig = build_fly_momentum_signal(fly, cfg)
    # print(sig.tail(20)[["score", "position", "action"]])
