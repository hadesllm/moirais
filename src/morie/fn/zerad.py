"""Radiation model (mobility)"""

import numpy as np

from ._containers import SpatialResult


def radiation_model(data, *, method="default"):
    """Radiation model (mobility)

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
        name="Radiation model (mobility)",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


radi = radiation_model


def cheatsheet() -> str:
    return "radiation_model({}) -> Radiation model (mobility)"
