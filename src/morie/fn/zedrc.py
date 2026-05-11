"""Spatial dose-response curve"""

import numpy as np

from ._containers import SpatialResult


def dose_resp_spatial(data, *, method="default"):
    """Spatial dose-response curve

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
        name="Spatial dose-response curve",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


dose = dose_resp_spatial


def cheatsheet() -> str:
    return "dose_resp_spatial({}) -> Spatial dose-response curve"
