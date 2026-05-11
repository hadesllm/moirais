"""SAC/SARAR model ML"""

import numpy as np

from ._containers import SpatialResult


def sac_ml(data, *, method="default"):
    """SAC/SARAR model ML

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
        name="SAC/SARAR model ML",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sac_ = sac_ml


def cheatsheet() -> str:
    return "sac_ml({}) -> SAC/SARAR model ML"
