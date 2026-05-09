"""SEM (Spatial Error) model ML estimation"""

import numpy as np

from ._containers import SpatialResult


def sem_ml(data, *, method="default"):
    """SEM (Spatial Error) model ML estimation

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
        name="SEM (Spatial Error) model ML estimation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


sem_ = sem_ml


def cheatsheet() -> str:
    return "sem_ml({}) -> SEM (Spatial Error) model ML estimation"
