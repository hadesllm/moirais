"""Three-step FCA"""

import numpy as np

from ._containers import SpatialResult


def three_step_fca(data, *, method="default"):
    """Three-step FCA

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
        name="Three-step FCA",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


thre = three_step_fca


def cheatsheet() -> str:
    return "three_step_fca({}) -> Three-step FCA"
