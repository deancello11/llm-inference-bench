"""Small math helpers with no GPU or heavy imports, so they are easy to unit test."""
from __future__ import annotations

import math
from typing import Sequence


def trapezoid_energy_j(times_s: Sequence[float], powers_w: Sequence[float]) -> float:
    """Energy in joules from power readings (watts) taken at known times (seconds).

    Adds up the area under the power curve, one trapezoid per pair of readings.
    """
    total = 0.0
    for i in range(1, len(times_s)):
        total += 0.5 * (powers_w[i] + powers_w[i - 1]) * (times_s[i] - times_s[i - 1])
    return total


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with q from 0 to 100 (for example q=95), blending between neighbors."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def binomial_stderr(acc: float, n: int) -> float:
    """Standard error of an accuracy measured on n questions."""
    return math.sqrt(acc * (1 - acc) / n) if n > 0 else float("nan")


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the best trade-offs, where lower x and higher y are both better.

    A point is dropped if some other point is at least as good on both axes and
    strictly better on one. The result is sorted by x, smallest first.
    """
    idx = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            (xj <= xi and yj >= yi) and (xj < xi or yj > yi)
            for j, (xj, yj) in enumerate(points)
            if j != i
        )
        if not dominated:
            idx.append(i)
    return sorted(idx, key=lambda i: points[i][0])
