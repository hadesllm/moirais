# moirais.fn — function file (hadesllm/moirais)
"""Diggle-Cressie-Loosmore test"""

import numpy as np

from ._containers import SpatialResult


def diggle_test(data, *, method="default"):
    """Diggle-Cressie-Loosmore test

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
        name="Diggle-Cressie-Loosmore test",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


digg = diggle_test


def cheatsheet() -> str:
    return "diggle_test({}) -> Diggle-Cressie-Loosmore test"
