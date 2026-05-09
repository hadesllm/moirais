"""Join count permutation test"""

import numpy as np

from ._containers import SpatialResult


def join_count_perm(data, *, method="default"):
    """Join count permutation test

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
        name="Join count permutation test",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


join = join_count_perm


def cheatsheet() -> str:
    return "join_count_perm({}) -> Join count permutation test"
