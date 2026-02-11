"""
backtest/execution/timing.py

Execution time conventions:
- Given decision bar t and exec_lag_bars, determine execution bar te.
- Allows plugging different conventions later (e.g., signal at close executes next open),
  but for snapshot-based L2 this is usually just an index shift.
"""

from __future__ import annotations


def resolve_exec_index(t: int, T: int, exec_lag_bars: int) -> int:
    """
    Return execution bar index te = t + exec_lag_bars.
    Clamp by returning T (meaning "no execution possible") if beyond the end.
    """
    te = t + int(exec_lag_bars)
    return te if te < T else T
