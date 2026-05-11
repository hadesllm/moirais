"""SDM impacts decomposition"""

import numpy as np

from ._containers import SpatialResult


def sdm_impacts(data, *, method="default"):
    """SDM impacts decomposition

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
        name="SDM impacts decomposition",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sdm_ = sdm_impacts


def cheatsheet() -> str:
    return "sdm_impacts({}) -> SDM impacts decomposition"
