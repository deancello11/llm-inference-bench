"""Small pure helpers (no GPU, no heavy deps) so they can be unit tested anywhere."""
from __future__ import annotations

import math
from typing import Sequence


def trapezoid_energy_j(times_s: Sequence[float], powers_w: Sequence[float]) -> float:
    """Integrate sampled power (W) over time (s) with the trapezoid rule -> joules."""
    total = 0.0
    for i in range(1, len(times_s)):
        total += 0.5 * (powers_w[i] + powers_w[i - 1]) * (times_s[i] - times_s[i - 1])
    return total


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
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
    return math.sqrt(acc * (1 - acc) / n) if n > 0 else float("nan")


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points when MINIMISING x and MAXIMISING y.

    A point is dominated if another point has x <= and y >= with at least one strict.
    Returned sorted by x ascending.
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
