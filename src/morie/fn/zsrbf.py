"""Multiquadric RBF interpolation"""

import numpy as np

from ._containers import SpatialResult


def rbf_multiquad(data, *, method="default"):
    """Multiquadric RBF interpolation

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="Multiquadric RBF interpolation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


rbf_ = rbf_multiquad


def cheatsheet() -> str:
    return "rbf_multiquad({}) -> Multiquadric RBF interpolation"
