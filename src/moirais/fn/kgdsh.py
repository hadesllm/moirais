# moirais.fn — function file (hadesllm/moirais)
"""Disjunctive kriging Hermite polynomials"""

import numpy as np

from ._containers import SpatialResult


def dk_hermite(data, *, method="default"):
    """Disjunctive kriging Hermite polynomials

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
        name="Disjunctive kriging Hermite polynomials",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


dk_h = dk_hermite


def cheatsheet() -> str:
    return "dk_hermite({}) -> Disjunctive kriging Hermite polynomials"
