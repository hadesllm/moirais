"""Two-step floating catchment area"""

import numpy as np

from ._containers import SpatialResult


def two_step_fca(data, *, method="default"):
    """Two-step floating catchment area

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
        name="Two-step floating catchment area",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


two_ = two_step_fca


def cheatsheet() -> str:
    return "two_step_fca({}) -> Two-step floating catchment area"
