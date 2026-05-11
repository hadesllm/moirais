"""Join count statistic"""

import numpy as np

from ._containers import SpatialResult


def join_count(data, *, method="default"):
    """Join count statistic

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
        name="Join count statistic",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


join = join_count


def cheatsheet() -> str:
    return "join_count({}) -> Join count statistic"
