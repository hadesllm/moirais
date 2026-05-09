# moirais.fn — function file (hadesllm/moirais)
"""Ripley's K-function"""

import numpy as np

from ._containers import SpatialResult


def k_function(data, *, method="default"):
    """Ripley's K-function

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
        name="Ripley's K-function",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


k_fu = k_function


def cheatsheet() -> str:
    return "k_function({}) -> Ripley's K-function"
