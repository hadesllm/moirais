"""Random non-Gaussian field"""

import numpy as np

from ._containers import SpatialResult


def random_nongauss(data, *, method="default"):
    """Random non-Gaussian field

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
        name="Random non-Gaussian field",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


rand = random_nongauss


def cheatsheet() -> str:
    return "random_nongauss({}) -> Random non-Gaussian field"
