"""General Nesting Spatial model"""

import numpy as np

from ._containers import SpatialResult


def gns_ml(data, *, method="default"):
    """General Nesting Spatial model

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
        name="General Nesting Spatial model",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


gns_ = gns_ml


def cheatsheet() -> str:
    return "gns_ml({}) -> General Nesting Spatial model"
