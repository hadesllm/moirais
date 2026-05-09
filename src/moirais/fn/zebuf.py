"""Buffer-based exposure assessment"""

import numpy as np

from ._containers import SpatialResult


def buffer_exposure(data, *, method="default"):
    """Buffer-based exposure assessment

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
        name="Buffer-based exposure assessment",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


buff = buffer_exposure


def cheatsheet() -> str:
    return "buffer_exposure({}) -> Buffer-based exposure assessment"
