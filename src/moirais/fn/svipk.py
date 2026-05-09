"""Kernel smoothed ideal point"""

import numpy as np

from ._containers import DescriptiveResult


def ideal_point_kernel(data, *, method="default"):
    """Kernel smoothed ideal point

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="svipk",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


idea = ideal_point_kernel


def cheatsheet() -> str:
    return "ideal_point_kernel({}) -> Kernel smoothed ideal point"
