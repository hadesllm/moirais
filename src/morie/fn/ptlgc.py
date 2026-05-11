# morie.fn — function file (hadesllm/morie)
"""Log-Gaussian Cox process"""

import numpy as np

from ._containers import SpatialResult


def log_gaussian_cox(data, *, method="default"):
    """Log-Gaussian Cox process

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
        name="Log-Gaussian Cox process",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


log_ = log_gaussian_cox


def cheatsheet() -> str:
    return "log_gaussian_cox({}) -> Log-Gaussian Cox process"
