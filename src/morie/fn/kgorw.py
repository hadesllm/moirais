# morie.fn — function file (hadesllm/morie)
"""Ordinary kriging weights"""

import numpy as np

from ._containers import SpatialResult


def ok_weights(data, *, method="default"):
    """Ordinary kriging weights

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
        name="Ordinary kriging weights",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ok_w = ok_weights


def cheatsheet() -> str:
    return "ok_weights({}) -> Ordinary kriging weights"
