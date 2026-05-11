"""Travel time catchment"""

import numpy as np

from ._containers import SpatialResult


def travel_time_catch(data, *, method="default"):
    """Travel time catchment

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
        name="Travel time catchment",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


trav = travel_time_catch


def cheatsheet() -> str:
    return "travel_time_catch({}) -> Travel time catchment"
