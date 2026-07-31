"""Confidence-interval helpers (no extra dependency; used for prevalence reporting)."""

from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p_hat, low, high).

    Robust for small n and proportions near 0 or 1, which is why we use it for
    prevalence rather than the normal approximation.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))
