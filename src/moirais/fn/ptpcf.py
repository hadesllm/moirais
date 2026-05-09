# moirais.fn — function file (hadesllm/moirais)
"""Pair correlation function g(r)"""

import numpy as np

from ._containers import SpatialResult


def pair_corr_fn(data, *, method="default"):
    """Pair correlation function g(r)

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
        name="Pair correlation function g(r)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


pair = pair_corr_fn


def cheatsheet() -> str:
    return "pair_corr_fn({}) -> Pair correlation function g(r)"
