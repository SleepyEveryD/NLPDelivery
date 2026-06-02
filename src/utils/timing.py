"""Timing helpers. Watch the 30-second wall, we must.

A stopwatch context manager and a budget guard, these provide -- per-stage breakdowns, they enable.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def stopwatch(sink: dict, key: str):
    # Time a block this does -- into sink[key] the elapsed seconds go.
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = time.perf_counter() - start


class LatencyGuard:
    """Against the budget, warn us this does.

    Exceed the wall we must not -- so before a costly step, the remaining time check we can.
    """

    def __init__(self, budget_s: float = 30.0):
        self.budget_s = budget_s
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def remaining(self) -> float:
        return self.budget_s - self.elapsed()

    def exceeded(self) -> bool:
        # Past the wall, are we?
        return self.remaining() <= 0
