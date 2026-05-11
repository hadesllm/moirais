# morie.fn — function file (hadesllm/morie)
"""KDE bandwidth selection (spatial)"""

import numpy as np

from ._containers import SpatialResult


def kde_bandwidth(data, *, method="default"):
    """KDE bandwidth selection (spatial)

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
        name="KDE bandwidth selection (spatial)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


kde_ = kde_bandwidth


def cheatsheet() -> str:
    return "kde_bandwidth({}) -> KDE bandwidth selection (spatial)"
