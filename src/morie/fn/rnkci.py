# morie.fn — function file (hadesllm/morie)
"""Rank-based confidence intervals for location (distribution-free)."""

import numpy as np
from scipy import stats as sp_stats

__all__ = ["rnkci"]


def rnkci(x, ci_level=0.95, axis=0):
    r"""
    Rank-based confidence interval for population median.

    Uses order statistics to construct distribution-free CI
    for the population median without assuming any parametric form.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    elif x.ndim != 1:
        raise ValueError("Input must be 1D or 2D array")

    n = len(x)
    if n < 3:
        raise ValueError("Sample size must be ≥3")

    x_sorted = np.sort(x)

    alpha = 1 - ci_level
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    # Wilcoxon-based CI construction
    d = int(np.ceil(n / 2 - z_crit * np.sqrt(n) / 2))
    d = max(1, min(d, n - 1))

    ci_lower = x_sorted[d - 1]
    ci_upper = x_sorted[n - d]

    point_estimate = np.median(x_sorted)

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_level": float(ci_level),
        "n": int(n),
    }
