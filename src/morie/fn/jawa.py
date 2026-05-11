# morie.fn — function file (hadesllm/morie)
"""Jackknife estimator. 'Utini!' -- Jawa"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def jackknife(x, *, statistic=np.mean) -> DescriptiveResult:
    """Delete-1 jackknife: bias and SE estimation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations")
    theta_full = float(statistic(x))
    theta_i = np.array([float(statistic(np.delete(x, i))) for i in range(n)])
    theta_jack = float(np.mean(theta_i))
    bias = (n - 1) * (theta_jack - theta_full)
    se = float(np.sqrt((n - 1) / n * np.sum((theta_i - theta_jack) ** 2)))
    return DescriptiveResult(
        name="Jackknife",
        value=None,
        extra={
            "estimate": theta_full,
            "bias": bias,
            "se": se,
            "corrected": theta_full - bias,
            "n": n,
        },
    )


jawa = jackknife


def cheatsheet() -> str:
    return "jackknife({}) -> Jackknife estimator. 'Utini!' -- Jawa"
