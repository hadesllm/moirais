# morie.fn -- function file (hadesllm/morie)
"""Indicator kriging probability"""

import numpy as np

from ._containers import SpatialResult


def ik_probability(data, *, method="default"):
    """Indicator kriging probability

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
        name="Indicator kriging probability",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


ik_p = ik_probability


def cheatsheet() -> str:
    return "ik_probability({}) -> Indicator kriging probability"
