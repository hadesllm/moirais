"""Enhanced 2SFCA"""

import numpy as np

from ._containers import SpatialResult


def enhanced_2sfca(data, *, method="default"):
    """Enhanced 2SFCA

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
        name="Enhanced 2SFCA",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


enha = enhanced_2sfca


def cheatsheet() -> str:
    return "enhanced_2sfca({}) -> Enhanced 2SFCA"
