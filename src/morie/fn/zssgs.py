"""Sequential Gaussian simulation"""

import numpy as np

from ._containers import SpatialResult


def seq_gauss_sim(data, *, method="default"):
    """Sequential Gaussian simulation

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
        name="Sequential Gaussian simulation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


seq_ = seq_gauss_sim


def cheatsheet() -> str:
    return "seq_gauss_sim({}) -> Sequential Gaussian simulation"
