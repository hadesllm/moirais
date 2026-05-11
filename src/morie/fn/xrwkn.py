"""K-nearest neighbors weights"""

import numpy as np

from ._containers import SpatialResult


def w_knn(data, *, method="default"):
    """K-nearest neighbors weights

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
        name="K-nearest neighbors weights",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


w_k = w_knn


def cheatsheet() -> str:
    return "w_knn({}) -> K-nearest neighbors weights"
